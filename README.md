# SIIM-ACR Pneumothorax Segmentation

## Features
### Pre-processing
#### Aggressive Data Augmentation
#### Sliding sampling rate
Due to the imbalanced dataset between non-pneumothorax and pneumothorax images, instead of using original dataset, positive and negative samples are re-sampled ahead of each epoch such that positive images accounts for `k%` and the remaining percentage is used for negative ones. The optimal `k` based on other research is `80%`
### Modeling
Based on training part of other teams, our study proposed three models:
* U-Net with backbone ResNet-34
* U-Net with backbone ResNet-50
* U-Net with backbone SE-ResNeXt-50
### Post-processing
To threshold the probability mask that model returns, three threshold values are used and searched the optimal one during training process. In general, our thresholds are(*upper bound threshold, lower bound threshold, minimum positive area*). Mechanism of our thresholding method is to firstly check how many pixel on the output mask have the value larger than *upper bound threshold*. Then, if the figure is less than *minimum positive area*, the mask is non-pneumothorax; otherwise, the pixel greater than *lower bound threshold* is positive pixel
### Result
| Model         | Validation accuracy | Test accuracy | Thresholds |
| ------------- |:-------------:| :-----:| :-----:|
| ​U-Net with ResNet-34 | 78.6% | 60% | (0.75, 1000, 0.3) |
| U-Net with ResNet-50 | 86.8% | 88.2% | (0.75, 2000, 0.4) |
| **U-Net with SE-ResNeXt-50** | 87.2% | **89.4%** | (0.6, 3000, 0.4) |

Our top score is much greater than that of winning solution
![](https://res.cloudinary.com/dqagyeboj/image/upload/v1614296737/result_ylysn4.png)
## Install
```
pip install -r requirements.txt
```
## Data preparation
You can download the raw data [here](https://drive.google.com/file/d/10iG8XqtNeAfitYxnELfBXpW1ZBHtcYac/view?usp=sharing) due to the termination of dataset on Kaggle
Set your own data path, here our data paths are `siim/dicom-images-train/`, `siim/dicom-images-test/` and `siim/train-rle.csv`
```
python convert2png.py --train-path siim/dicom-images-train/ --test-path siim/dicom-images-test/ --rle-path siim/train-rle.csv --output-path input/dataset1024 --image-size 1024
```
## Pipeline
### U-Net with ResNet-34
#### Training
```
python train.py --train-config experiments/albunet_valid/train_config_part0.yaml
``` 
#### Inference and Create submission file
```
python inference.py --config experiments/albunet_valid/2nd_stage_inference.yaml
```
To submit
```
python triplet_submit.py --config experiments/albunet_valid/2nd_stage_submit.yaml
```
### U-Net with ResNet-50
#### Training
```
python train.py --train-config experiments/resnet50/train_config_part0.yaml
``` 
#### Inference and Create submission file
```
python inference.py --config experiments/resnet50/2nd_stage_inference.yaml
```
To submit
```
python triplet_submit.py --config experiments/resnet50/2nd_stage_submit.yaml
```
### U-Net with SE-ResNeXt-50
#### Training
```
python train.py --train-config experiments/seunet/train_config_part0.yaml
``` 
#### Inference and Create submission file
```
python inference.py --config experiments/seunet/2nd_stage_inference.yaml
```
To submit
```
python triplet_submit.py --config experiments/seunet/2nd_stage_submit.yaml
```
## Demo
Visit the branch `Thanh` to launch our demo