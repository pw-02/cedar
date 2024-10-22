#!/bin/bash

#cv-torch
python pipelines/simclrv2/ray_dataset.py 
#cv-tf
python pipelines/simclrv2/ray_tf_dataset.py
#nlp-torch
python pipelines/wikitext103/ray_dataset.py
#nlp-hf-tf
python pipelines/wikitext103/ray_tf_dataset.py
#nlp-tf
python pipelines/wikitext103/ray_tf_service_dataset.py
#asr
python pipelines/commonvoice/ray_dataset.py
# ssd-torch
python pipelines/coco/ray_dataset.py
# ssd-tf
python pipelines/coco/ray_tf_dataset.py