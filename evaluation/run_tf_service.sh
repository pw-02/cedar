#!/bin/bash

#cv-tf
python eval_tf.py --dataset_file pipelines/simclrv2/tf_dataset.py --service_addr 10.138.0.90:38655 --num_parallel_calls -1
#nlp-tf
python eval_tf.py --dataset_file pipelines/wikitext103/tf_service_dataset.py --service_addr 10.138.0.90:38655 --num_parallel_calls -1 --num_total_samples 200000
#ssd-tf
python eval_tf.py --dataset_file pipelines/coco/tf_service_dataset.py --service_addr 10.138.0.90:38655 --num_parallel_calls -1