# GFD-Net: A Remote Sensing Image Segmentation Network with Hybrid Attention Gated Fusion and Dynamic Frequency Filtering



## Datasets

* [ISPRS Vaihingen and Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab)
* [WHUbuilding](https://gpcv.whu.edu.cn/data/building_dataset.html)

## Install

```
conda create -n pyname python=3.9
conda activate pyname
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118
pip install -r GFDNet/requirements.txt
```

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**Vaihingen**



Generate the training set.

```

python GFDNet/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" --output-img-dir "data/vaihingen/train/images_1024" --output-mask-dir "data/vaihingen/train/masks_1024" --mode "train" --split-size 1024 --stride 512


```

Generate the testing set.

```

python GFDNet/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded```

Generate the masks\_1024\_rgb (RGB format ground truth labels) for visualization.

```

```

python GFDNet/tools/vaihingen_patch_split.py --img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" --output-img-dir "data/vaihingen/test/images_1024" --output-mask-dir "data/vaihingen/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt

```

As for the validation set , you  can select one from training dataset.





**Potsdam**

```

python GFDNet/tools/potsdam_patch_split.py --img-dir "data/potsdam/train_images" --mask-dir "data/potsdam/train_masks" --output-img-dir "data/potsdam/train/images_1024" --output-mask-dir "data/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 1024 --rgb-image
```



```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks_eroded" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image
```



```
python tools/potsdam_patch_split.py --img-dir "data/potsdam/test_images" --mask-dir "data/potsdam/test_masks" --output-img-dir "data/potsdam/test/images_1024" --output-mask-dir "data/potsdam/test/masks_1024_rgb" --mode "val" --split-size 1024 --stride 1024 --gt --rgb-image
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python GFDNet/train_supervision.py -c GFDNet/config/potsdam/gfdnet.py
```

```shell
python GFDNet/train_supervision.py -c GFDNet/config/vaihingen/gfdnet.py
```

```shell
python GFDNet/train_supervision.py -c GFDNet/config/WHUbuilding/gfdnet.py
```

## Testing

**Vaihingen**


python GFDNet/vaihingen_test.py -c GFDNet/config/vaihingen/gfdnet.py -o fig_results/GFDNet_vaihingen/ --rgb ```

**Potsdam**


python GFDNet/potsdam_test.py -c GFDNet/config/potsdam/gfdnet.py -o fig_results/GFDNet_potsdam/ --rgb ```

**WHUbuilding**


python GFDNet/whubuilding_test.py -c GFDNet/config/WHUbuilding/gfdnet.py -o  fig_results/GFDNet_whubuild ```

## Acknowledgement

Many thanks the following projects's contributions.

*[GeoSeg](https://github.com/WangLibo1995/GeoSeg)
*[pytorch lightning](https://www.pytorchlightning.ai/)
*[timm](https://github.com/rwightman/pytorch-image-models)
*[pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
*[ttach](https://github.com/qubvel/ttach)
*[catalyst](https://github.com/catalyst-team/catalyst)





