# Medical-image-segmentation
This repo implement several finetuned segmentation model (Unet-vgg,Unet_resnet, Refinenet,FCN) pretrained on ImageNet for medical image analysis (e.g. Ultrasound, Skin lesion, medical sub-figure). 

It implements popular models in medical image analysis.  The original dataset is from imageclef and isic.

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch](https://pytorch.org/) (Recommended version 9.2)
- [Python 3](https://www.python.org/)

## Results
Sample result from Unet
![image](https://github.com/FredericChai/Medical-image-segmentation/blob/main/Medical_Image_Segmentation/sample/1.png)

Semantic segmentation on ISIC2016
|     Models     | Dice ↓ (%) |
|:--------------:|:----------------:|
| FCN-vgg |  88.0    |
| DeepV3 |  88.0    |
| RefineNet| 85.8  |
| Unet-vgg |  88.2 |
| Unet-resnet  |  89.3 |
