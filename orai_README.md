# Image inpainting

Note: the model was trained with scene images from Places365-Standard http://places2.csail.mit.edu/download.html

## Setup env

- Python 3.7 `conda create -n inpainting python=3.7`
- Install packages

```
conda activate inpainting
pip install opencv-python torch torchvision numpy
```

## Download code and run

- Clone code `git clone https://github.com/ntoand/DeepFillv2_Pytorch.git`
- Download trained model to `pretrained_model` https://drive.google.com/file/d/1LVyaCQS6xhyIwqx-PeUs6IGV0-pXJrin/view?usp=sharing
- Run

```
cd DeepFillv2_Pytorch
python predict.py --image test_data/1.png --mask test_data_mask/1.png
```
