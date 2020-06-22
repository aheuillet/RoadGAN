#!/usr/bin/bash

conda=$(which conda)

if [ -z $conda ] then
    echo "conda not detected, installing conda..."
    wget 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh'
    bash Miniconda3-latest-Linux-x86_64.sh
    echo "...done"
fi

echo "Creating conda env and installing requirements..."
conda env create  --file requirements.txt
echo "...done"

sudo apt update && sudo apt install python-pip
pip install gdown 

echo "Downloading attribute hallucination checkpoint..."
mkdir attribute_hallucination/pretrained_models/
gdown 'https://drive.google.com/uc?id=17ydh2x0Va8t315_p_7I20JFCYuLXRu_Y' -O attribute_hallucination/pretrained_models/sgn_enhancer_G_latest.pth
echo "...done"

echo "Downloading semantic segmentation checkpoint..."
mkdir attribute_hallucination/semantic_segmentation_pytorch/ade20k-resnet50dilated-ppm_deepsup
wget 'http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth' -O attribute_hallucination/semantic_segmentation_pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
wget 'http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth' -O attribute_hallucination/semantic_segmentation_pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
echo "...done"

echo "Initializing Flownet2 network..."
cd few_shot_vid2vid/
conda activate HAL
python scripts/download_flownet2.python
echo "...done"

echo "All done"