#!/bin/bash

#cv-tf
python eval_tf.py --dataset_file pipelines/simclrv2/tf_dataset.py --num_parallel_calls -1
#nlp-hf-tf
python eval_tf.py --dataset_file pipelines/wikitext103/tf_dataset.py --num_parallel_calls -1 --num_total_samples 100000
#nlp-tf
python eval_tf.py --dataset_file pipelines/wikitext103/tf_service_dataset.py --num_parallel_calls -1 --num_total_samples 200000
#asr
python eval_tf.py --dataset_file pipelines/commonvoice/tf_dataset.py --num_parallel_calls -1 --num_total_samples 10000
# ssd
python eval_tf.py --dataset_file pipelines/coco/tf_dataset.py --num_parallel_calls -1