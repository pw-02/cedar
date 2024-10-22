#!/bin/bash

#cv-torch
python eval_torch.py --dataset_file pipelines/simclrv2/torch_dataset.py --num_workers 8
#nlp-torch
python eval_torch.py --dataset_file pipelines/wikitext103/torch_dataset.py --num_workers 8 --num_total_samples 100000
#asr
python eval_torch.py --dataset_file pipelines/commonvoice/torch_dataset.py --num_workers 8 --num_total_samples 10000
# ssd
python eval_torch.py --dataset_file pipelines/coco/torch_dataset.py --num_workers 8