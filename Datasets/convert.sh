#!/usr/bin/bash

CITY_DIR="./Cityscapes"
OUT_DIR="${CITY_DIR}/annotations"
IMG_DIR="${CITY_DIR}/images"

if [[ -f $OUT_DIR ]]
then
    mkdir $OUT_DIR
fi

if [[ -f $IMG_DIR ]]
then
    mkdir $IMG_DIR
fi

echo "Converting Cityscapes annotations to COCO format..."

python convert_cityscapes_to_coco.py --dataset cityscapes_instance_only --outdir $OUT_DIR --datadir $CITY_DIR