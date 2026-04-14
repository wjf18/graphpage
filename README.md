# GFD-Net: A Remote Sensing Image Segmentation Network with Hybrid Attention Gated Fusion and Dynamic Frequency Filtering



## Datasets

* [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
* [WHUbuilding](https://gpcv.whu.edu.cn/data/building_dataset.html)

## Install

```
pip install -r GFDNet/requirements.txt
```

## Data Preprocessing

Download the datasets from the official website and split them yourself.

\*\*Vaihingen\*\*



Generate the training set.

```

python GFDNet/tools/vaihingen\\\\\\\_patch\\\\\\\_split.py \\\\\\\\

\\\\--img-dir "data/vaihingen/train\\\\\\\_images" \\\\\\\\

\\\\--mask-dir "data/vaihingen/train\\\\\\\_masks" \\\\\\\\

\\\\--output-img-dir "data/vaihingen/train/images\\\\\\\_1024" \\\\\\\\

\\\\--output-mask-dir "data/vaihingen/train/masks\\\\\\\_1024" \\\\\\\\

\\\\--mode "train" --split-size 1024 --stride 512 

```

Generate the testing set.

```

python GFDNet/tools/vaihingen\\\\\\\_patch\\\\\\\_split.py \\\\\\\\

\\\\--img-dir "data/vaihingen/test\\\\\\\_images" \\\\\\\\

\\\\--mask-dir "data/vaihingen/test\\\\\\\_masks\\\\\\\_eroded" \\\\\\\\

\\\\--output-img-dir "data/vaihingen/test/images\\\\\\\_1024" \\\\\\\\

\\\\--output-mask-dir "data/vaihingen/test/masks\\\\\\\_1024" \\\\\\\\

\\\\--mode "val" --split-size 1024 --stride 1024 \\\\\\\\

\\\\--eroded

```

Generate the masks\_1024\_rgb (RGB format ground truth labels) for visualization.

```

python GFDNet/tools/vaihingen\\\\\\\_patch\\\\\\\_split.py \\\\\\\\

\\\\--img-dir "data/vaihingen/test\\\\\\\_images" \\\\\\\\

\\\\--mask-dir "data/vaihingen/test\\\\\\\_masks" \\\\\\\\

\\\\--output-img-dir "data/vaihingen/test/images\\\\\\\_1024" \\\\\\\\

\\\\--output-mask-dir "data/vaihingen/test/masks\\\\\\\_1024\\\\\\\_rgb" \\\\\\\\

\\\\--mode "val" --split-size 1024 --stride 1024 \\\\\\\\

\\\\--gt

```

As for the validation set, you can select some images from the training set to build it.



\*\*Potsdam\*\*

```

python GFDNet/tools/potsdam\\\\\\\_patch\\\\\\\_split.py \\\\\\\\

\\\\--img-dir "data/potsdam/train\\\\\\\_images" \\\\\\\\

\\\\--mask-dir "data/potsdam/train\\\\\\\_masks" \\\\\\\\

\\\\--output-img-dir "data/potsdam/train/images\\\\\\\_1024" \\\\\\\\

\\\\--output-mask-dir "data/potsdam/train/masks\\\\\\\_1024" \\\\\\\\

\\\\--mode "train" --split-size 1024 --stride 1024 --rgb-image 

```



```

python GFDNet/tools/potsdam\\\\\\\_patch\\\\\\\_split.py \\\\\\\\

\\\\--img-dir "data/potsdam/test\\\\\\\_images" \\\\\\\\

\\\\--mask-dir "data/potsdam/test\\\\\\\_masks\\\\\\\_eroded" \\\\\\\\

\\\\--output-img-dir "data/potsdam/test/images\\\\\\\_1024" \\\\\\\\

\\\\--output-mask-dir "data/potsdam/test/masks\\\\\\\_1024" \\\\\\\\

\\\\--mode "val" --split-size 1024 --stride 1024 \\\\\\\\

\\\\--eroded --rgb-image

```



```

python GFDNet/tools/potsdam\\\\\\\_patch\\\\\\\_split.py \\\\\\\\

\\\\--img-dir "data/potsdam/test\\\\\\\_images" \\\\\\\\

\\\\--mask-dir "data/potsdam/test\\\\\\\_masks" \\\\\\\\

\\\\--output-img-dir "data/potsdam/test/images\\\\\\\_1024" \\\\\\\\

\\\\--output-mask-dir "data/potsdam/test/masks\\\\\\\_1024\\\\\\\_rgb" \\\\\\\\

\\\\--mode "val" --split-size 1024 --stride 1024 \\\\\\\\

\\\\--gt --rgb-image

```

## Training

"-c" means the path of the config, use different **config** to train different models.

```shell
python GFDNet/train\\\\\\\_supervision.py -c GFDNet/config/potsdam/gfdnet.py
```

```shell
python GFDNet/train\\\\\\\_supervision.py -c GFDNet/config/vaihingen/gfdnet.py
```

```shell
python GFDNet/train\\\\\\\_supervision.py -c GFDNet/config/WHUbuilding/gfdnet.py
```

## Testing

**Vaihingen**

```shell
python GFDNet/vaihingen\_test.py -c GFDNet/config/vaihingen/gfdnet.py -o fig\\\\\\\_results/GFDNet\_vaihingen/ --rgb ```

**Potsdam**

```shell
python GFDNet/potsdam\_test.py -c GFDNet/config/potsdam/gfdnet.py -o fig\_results/GFDNet\_potsdam/ --rgb ```

**WHUbuilding**

```shell
python GFDNet/whubuilding\_test.py -c GFDNet/config/WHUbuilding/gfdnet.py -o  fig\_results/GFDNet\_whubuild --rgb ```

## Acknowledgement

Many thanks the following projects's contributions.

\\\* \\\[GeoSeg](https://github.com/WangLibo1995/GeoSeg)
\\\* \\\[pytorch lightning](https://www.pytorchlightning.ai/)
\\\* \\\[timm](https://github.com/rwightman/pytorch-image-models)
\\\* \\\[pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
\\\* \\\[ttach](https://github.com/qubvel/ttach)
\\\* \\\[catalyst](https://github.com/catalyst-team/catalyst)



