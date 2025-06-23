# DDHA-Net
We evaluate our method on [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) ,[DDR](https://github.com/nkicsl/DDR-dataset), and [HQCDR](https://github.com/xqh180110910537/HQCDR).
The dataset we pre-process based on preprocessing operations and mmseg requirements is at x.

First, you should set folds as follows :
```
|--
    |--data
        |--ann_dir
            |--train
            |--test
            |--...
        |--img_dir
            |--train
            |--test
            |--...
        |--splits
            |--train.txt
            |--test.txt
    |--mmseg
    |--HACDRNet&DDHANet
    |-- ...
    ...
```

## Environment

This code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Please refer to mmsegmentation for specific download details.

-   pytorch=1.9.0
-   mmsegmentation=0.30.0
-   mmcv-full>=1.5.0
## Prepare
```
python ./setup.py install
```
## Train
```
# train
—— multi-gpu
CUDA_VISIBLE_DEVICES=0,1 PORT=12345 tools/dist_train.sh ./HACDRNet&DDHANet/HACDR_ddr.py 2
—— single-gpu
python ./tools/train.py ./HACDRNet&DDHANet/HACDR_ddr.py --gpu 0
```
## Test
```
# test
—— single-gpu
python ./tools/test.py ./HACDRNet&DDHANet/HACDR_ddr.py --gpu 0 --eval mDice
```
## Inference
```
In ./single_infer.py
```
## Model detail
```
Encoder: In ./mmseg/backbones/hacdr.py
Decoder: In ./mmseg/decode_heads/dr_unet_head.py
```
## Citation
