import timm
import torch.nn.functional as F
import torch
from torch import nn
from timm.layers import DropPath, to_2tuple, trunc_normal_

class Soa(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=Soa, drop=0.,
                 bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_heads = origin_weight.shape[0:3]
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    new_weight = F.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight


class CAFFilter(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=Soa, act2_layer=nn.Identity,
                 bias=False, num_scales=3,
                 base_filter_resolution=(16, 16),
                 weight_resize=True,
                 spatial_context_channels_ratio=0.5,
                 **kwargs):
        super().__init__()
        self.dim = dim
        base_filter_resolution_tuple = to_2tuple(base_filter_resolution)
        self.base_h = base_filter_resolution_tuple[0]
        self.base_w_half = base_filter_resolution_tuple[1] // 2 + 1

        self.num_scales = num_scales
        self.med_channels = int(expansion_ratio * self.dim)
        self.weight_resize = weight_resize

        self.scale_ratios = [1.0, 0.5, 1.5]

        self.complex_weights_scales = nn.ParameterList()
        for i in range(num_scales):
            scale_factor = self.scale_ratios[i]
            current_h = int(self.base_h * scale_factor)
            current_w_half = int(self.base_w_half * scale_factor)
            current_h = max(1, current_h)
            current_w_half = max(1, current_w_half)

            self.complex_weights_scales.append(
                nn.Parameter(
                    torch.randn(current_h, current_w_half, self.med_channels, 2,
                                dtype=torch.float32) * 0.02
                )
            )

        self.pwconv1 = nn.Linear(self.dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()

        spatial_context_channels = max(1, int(self.dim * spatial_context_channels_ratio))
        self.spatial_context_channels = spatial_context_channels


        self.spatial_context_extractor = nn.Sequential(
            nn.Conv2d(self.dim, spatial_context_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_context_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )


        reweight_input_dim = self.dim + (spatial_context_channels)  # C + C/2
        self.reweight = Mlp(reweight_input_dim,
                            reweight_expansion_ratio,
                            self.num_scales * self.med_channels)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, self.dim, bias=bias)

    def forward(self, x):
        B, H, W, C = x.shape



        x_pre = self.pwconv1(x)
        x_pre = self.act1(x_pre)
        x_pre = x_pre.to(torch.float32)

        x_fft = torch.fft.rfft2(x_pre.permute(0, 3, 1, 2), dim=(2, 3), norm='ortho')
        x_fft = x_fft.permute(0, 2, 3, 1)

        fft_H, fft_W_half = x_fft.shape[1], x_fft.shape[2]

        all_scaled_complex_filters = []
        for i in range(self.num_scales):
            scale_param = self.complex_weights_scales[i]

            if self.weight_resize:
                current_scale_filter_2_channel = resize_complex_weight(
                    scale_param, fft_H, fft_W_half
                )
            else:
                current_scale_filter_2_channel = scale_param
            all_scaled_complex_filters.append(
                torch.view_as_complex(current_scale_filter_2_channel.contiguous())
            )

        global_channel_context = x.mean(dim=(1, 2))  # 形状 (B, C) -> (B, self.dim)
        x_permuted_for_spatial = x.permute(0, 3, 1, 2)
        spatial_context = self.spatial_context_extractor(x_permuted_for_spatial)
        spatial_context = spatial_context.flatten(1)
        fused_context = torch.cat((global_channel_context, spatial_context), dim=1)
        routing_weights = self.reweight(fused_context)

        routing_weights = routing_weights.view(B, self.num_scales, self.med_channels)
        routing_weights = F.softmax(routing_weights, dim=1)
        routing_weights = routing_weights.to(torch.complex64)
        combined_total_filter = torch.zeros_like(x_fft, dtype=torch.complex64)
        for i in range(self.num_scales):

            single_scale_filter = all_scaled_complex_filters[i]
            current_scale_routing_weights = routing_weights[:, i, :].unsqueeze(1).unsqueeze(1)
            combined_total_filter += single_scale_filter * current_scale_routing_weights

        x_filtered_fft = x_fft * combined_total_filter
        x_filtered_fft = x_filtered_fft.permute(0, 3, 1, 2)
        x_filtered_spatial = torch.fft.irfft2(x_filtered_fft, s=(H, W), dim=(2, 3), norm='ortho')
        x_filtered_spatial = x_filtered_spatial.permute(0, 2, 3, 1)

        x_out = self.act2(x_filtered_spatial)
        x_out = self.pwconv2(x_out)
        return x_out




class FreqDomainGating(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.act = nn.ReLU(inplace=True)
        self.gate_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        ffted = torch.fft.rfft2(x, norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted = self.act(self.bn(self.conv(ffted)))
        real, imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(real, imag)
        inversed = torch.fft.irfft2(ffted, s=(H, W), norm='ortho')
        gate = self.gate_conv(inversed).sigmoid()
        return gate


class AxialAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim can be num_heads devided"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        qkv_h = self.qkv(x_h).reshape(B * W, H, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        x_h = (attn_h @ v_h).transpose(1, 2).reshape(B * W, H, C)
        x_h = self.proj(x_h).reshape(B, W, H, C).permute(0, 3, 2, 1)

        x_w = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        qkv_w = self.qkv(x_w).reshape(B * H, W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_w, k_w, v_w = qkv_w[0], qkv_w[1], qkv_w[2]
        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        x_w = (attn_w @ v_w).transpose(1, 2).reshape(B * H, W, C)
        x_w = self.proj(x_w).reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x_h + x_w


class DynamicConv(nn.Module):

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // ratio, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1), nn.Sigmoid()
        )
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        s = self.weight_generator(x)
        x_fused = s * self.conv3(x) + (1 - s) * self.conv5(x)
        return self.bn(self.pw_conv(x_fused))

class SPAFM(nn.Module):

    def __init__(self, high_res_channels, low_res_channels, intermediate_ratio=0.5, num_heads=8):
        super().__init__()
        intermediate_channels = int(high_res_channels * intermediate_ratio)
        self.proj_high_for_gate = nn.Conv2d(high_res_channels, intermediate_channels, 1)
        self.proj_low_for_gate = nn.Conv2d(low_res_channels, intermediate_channels, 1)

        self.gating_unit = FreqDomainGating(intermediate_channels)


        self.norm = nn.BatchNorm2d(high_res_channels)
        self.axial_attention = AxialAttention(high_res_channels, num_heads)
        self.dynamic_conv = DynamicConv(high_res_channels)

        self.expert_fusion_conv = nn.Conv2d(high_res_channels, high_res_channels, 1, bias=False)

        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(high_res_channels + low_res_channels, high_res_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(high_res_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_high, x_low):
        """
        Args:
            x_high (torch.Tensor):
            x_low (torch.Tensor):
        """
        x_low_upsampled = F.interpolate(x_low, size=x_high.shape[-2:], mode='bilinear', align_corners=False)

        p_high = self.proj_high_for_gate(x_high)
        p_low = self.proj_low_for_gate(x_low_upsampled)
        spatial_gate = self.gating_unit(p_high + p_low)

        gated_x = x_high * spatial_gate
        gated_x_norm = self.norm(gated_x)
        attn_out = self.axial_attention(gated_x_norm)
        conv_out = self.dynamic_conv(gated_x_norm)

        enhanced_x = gated_x + self.expert_fusion_conv(attn_out + conv_out)

        final_x = torch.cat([enhanced_x, x_low_upsampled], dim=1)
        output = self.final_fusion_conv(final_x)

        return output

class DiagonalConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, direction='main'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=0,
                              dilation=dilation, bias=bias)

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.direction = direction

        self.padding = (kernel_size - 1) * dilation // 2

        mask = torch.zeros(kernel_size, kernel_size)
        if direction == 'main':
            for i in range(kernel_size):
                mask[i, i] = 1
        elif direction == 'anti':
            for i in range(kernel_size):
                mask[i, kernel_size - 1 - i] = 1
        else:
            raise ValueError("direction 'main' or 'anti'")

        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        masked_kernel = self.conv.weight * self.mask
        return F.conv2d(padded_x, masked_kernel, self.conv.bias, self.conv.stride,
                        padding=0, dilation=self.dilation)

class DiagConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, direction='main'):
        super().__init__()
        self.conv = DiagonalConv2d(in_channels, out_channels, kernel_size,
                                   stride=stride, dilation=dilation, bias=False, direction=direction)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            act_layer(inplace=inplace)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNReLUy(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLUy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNy(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNy, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(0, ((stride - 1) + dilation * (kernel_size - 1)) // 2)
                      ),
            norm_layer(out_channels),
        )

class ConvBNReLUx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLUx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBNx(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNx, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), bias=bias,
                      dilation=(dilation, dilation), stride=(stride, stride),
                      padding=(((stride - 1) + dilation * (kernel_size - 1)) // 2, 0)
                      ),
            norm_layer(out_channels),
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class ODACM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=(1, 2, 4, 8)):
        super().__init__()

        self.preconv = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1)

        self.Recx = ConvBNReLUx(out_channels, out_channels // 4, kernel_size=kernel_size)
        self.Recx2 = ConvBNReLUx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[1])
        self.Recx4 = ConvBNReLUx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[2])
        self.Recx8 = ConvBNReLUx(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[3])

        self.Recy = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size)
        self.Recy2 = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[1])
        self.Recy4 = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[2])
        self.Recy8 = ConvBNReLUy(out_channels, out_channels//4, kernel_size=kernel_size, dilation=dilation[3])

        self.DiagMain1 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='main',
                                        dilation=dilation[0])
        self.DiagMain2 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='main',
                                        dilation=dilation[1])
        self.DiagMain4 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='main',
                                        dilation=dilation[2])
        self.DiagMain8 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='main',
                                        dilation=dilation[3])

        self.DiagAnti1 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='anti',
                                        dilation=dilation[0])
        self.DiagAnti2 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='anti',
                                        dilation=dilation[1])
        self.DiagAnti4 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='anti',
                                        dilation=dilation[2])
        self.DiagAnti8 = DiagConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, direction='anti',
                                        dilation=dilation[3])

        self.conv1 = ConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, stride=1,
                                dilation=dilation[0])
        self.conv2 = ConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, stride=1,
                                dilation=dilation[1])
        self.conv4 = ConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, stride=1,
                                dilation=dilation[2])
        self.conv8 = ConvBNReLU(out_channels, out_channels // 4, kernel_size=kernel_size, stride=1,
                                dilation=dilation[3])

        self.conv_out = ConvBNReLU(out_channels, out_channels, stride=1)  # 输出层

    def forward(self, x):
        x = self.preconv(x)

        featsx = torch.cat((self.Recx(x), self.Recx2(x), self.Recx4(x), self.Recx8(x)), dim=1)
        featsy = torch.cat((self.Recy(x), self.Recy2(x), self.Recy4(x), self.Recy8(x)), dim=1)


        feats_main_diag = torch.cat((self.DiagMain1(x), self.DiagMain2(x), self.DiagMain4(x), self.DiagMain8(x)), dim=1)

        feats_anti_diag = torch.cat((self.DiagAnti1(x), self.DiagAnti2(x), self.DiagAnti4(x), self.DiagAnti8(x)), dim=1)

        feats_standard = torch.cat((self.conv1(x), self.conv2(x), self.conv4(x), self.conv8(x)), dim=1)

        out = featsx + featsy + feats_main_diag + feats_anti_diag + feats_standard

        out = self.conv_out(out)

        return out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Fusion(nn.Module):
    def __init__(self, in_channsel=64,out_channels=64, eps=1e-8):
        super(Fusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.Preconv = Conv(in_channels=in_channsel,out_channels=out_channels,kernel_size=1)
        self.post_conv = SeparableConvBNReLU(out_channels, out_channels, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] *self.Preconv(x)
        x = self.post_conv(x)
        return x


class PMAHead(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=(1, 2, 4, 8), dropout=0., num_classes=6):
        super(PMAHead, self).__init__()
        self.odacm = ODACM(in_channels=dim, out_channels=dim, kernel_size=3, dilation=dilation)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(dim, num_classes, kernel_size=1))

    def forward(self, x):
        u = x.clone()

        attn = self.odacm(x)
        attn = attn * u

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        out = self.head(attn + c_attn + s_attn)

        return out

class SegHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        aux = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)

        return feat, aux

class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=(256, 512, 1024, 2048),
                 decode_channels=(64, 64, 64, 64),
                 dilation = ((1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8), (1, 2, 4, 8)),
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6,
                 ):
        super(Decoder, self).__init__()


        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels[-1], 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels[-2], 1)
        self.Conv3 = ConvBNReLU(encode_channels[-3], decode_channels[-3], 1)
        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels[-4], 1)

        self.hagfm2 = SPAFM(
            high_res_channels=decode_channels[-2],
            low_res_channels=decode_channels[-2],
            intermediate_ratio=0.5,
            num_heads=6
        )
        self.hagfm3 = SPAFM(
            high_res_channels=decode_channels[-3],
            low_res_channels=decode_channels[-3],
            intermediate_ratio=0.5,
            num_heads=6
        )
        self.hagfm4 = SPAFM(
            high_res_channels=decode_channels[-4],
            low_res_channels=decode_channels[-4],
            intermediate_ratio=0.5,
            num_heads=6
        )


        self.p3 = Fusion(decode_channels[-1], decode_channels[-2])
        self.dynamic_filter_p3 = CAFFilter(dim=decode_channels[-2], num_scales=3)


        self.p2 = Fusion(decode_channels[-2], decode_channels[-3])
        self.dynamic_filter_p2 = CAFFilter(dim=decode_channels[-3], num_scales=3)


        self.p1 = Fusion(decode_channels[-3], decode_channels[-4])
        self.dynamic_filter_p1 = CAFFilter(dim=decode_channels[-4], num_scales=3)


        self.Conv5 = ConvBN(decode_channels[-4], 64, 1)

        self.p = Fusion(64)
        self.seg_head = PMAHead(64, fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)  # dim=64

        # self.final_conv_1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux_head4 = SegHead(decode_channels[-1], num_classes)
        self.aux_head3 = SegHead(decode_channels[-2], num_classes)
        self.aux_head2 = SegHead(decode_channels[-3], num_classes)
        self.aux_head1 = SegHead(decode_channels[-4], num_classes)
        self.init_weight()

    def forward(self, res, res1, res2, res3, res4, h, w):

        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)
        res2 = self.Conv3(res2)
        res1 = self.Conv4(res1)

        res3_1 = self.hagfm2(res3, res4)
        res2_1 = self.hagfm3(res2, res3)
        res1_1 = self.hagfm4(res1, res2)

        x = self.p3(res4, res3_1)
        aux3_3, aux3 = self.aux_head3(x, h, w)
        # Apply DynamicFilter after fusion
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.dynamic_filter_p3(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W


        x = self.p2(x, res2_1)
        aux2_2, aux2 = self.aux_head2(x, h, w)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.dynamic_filter_p2(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W


        x = self.p1(x, res1_1)
        aux1_1, aux1 = self.aux_head1(x, h, w)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.dynamic_filter_p1(x)
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W


        x = self.Conv5(x)
        x = self.p(x, res)

        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x, aux1, aux2, aux3

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GFDNet(nn.Module):
    def __init__(self,num_classes,
                 dropout=0.1,
                 decode_channels=32):
        super(GFDNet, self).__init__()
        self.backbone = timm.create_model('swsl_resnet50', features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)
        encoder_channels = self.backbone.feature_info.channels()

        self.cnn = nn.Sequential(self.backbone.conv1,
                                 self.backbone.bn1,
                                 self.backbone.act1
                                 )

        self.cnn1 = nn.Sequential(self.backbone.maxpool,self.backbone.layer1)
        self.cnn2 = self.backbone.layer2
        self.cnn3 = self.backbone.layer3
        self.cnn4 = self.backbone.layer4

        decode_channels = [decode_channels * 6,decode_channels * 6 ,
                           decode_channels * 6,decode_channels * 6 ]
        self.decoder = Decoder(encoder_channels, decode_channels=decode_channels,
                               dropout=dropout, num_classes=num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        x_pre = self.cnn(x)    ##H/2

        res1 = self.cnn1(x_pre)##H/4

        res2 = self.cnn2(res1) ##H/8

        res3 = self.cnn3(res2) ##H/16

        res4 = self.cnn4(res3) ##H/32

        out, aux1, aux2,aux3 = self.decoder(x_pre, res1, res2, res3, res4,h, w)

        if self.training:
            return out, aux1,aux2,aux3
        else:
            return out

def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {total_params}")



if __name__ == '__main__':

    num_classes = 6
    in_batch, inchannel, in_h, in_w = 2, 3, 1024, 1024
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = GFDNet(num_classes)
    out ,aux2,aux3,aux4= net(x)
    print(out.shape)
    print_model_params(net)
