#! /bin/bash

mkdir ~/cedar/evaluation/datasets/coco

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017/ ~/cedar/evaluation/datasets/coco

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip annotations_trainval2017.zip
mv annotations/ ~/cedar/evaluation/datasets/coco

rm val2017.zip
rm annotations_trainval2017.zip
