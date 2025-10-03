#!/bin/bash

# Note, we provide the exact stats and config files produced by the optimizer on our setup in order to enable reproducibility.
# To generate new profiling stats and re-run the optimizer, use the --run_profiling and --generate_plan flags in eval_cedar.py
# Replace the stats and optimizer-produced config in the following commands

# cv-torch
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_stats.yaml --master_feature_config pipelines/simclrv2/configs/eval_cedar_local.yaml

# # cv-tf
# python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --master_feature_config pipelines/simclrv2/configs/eval_cedar_local_tf.yaml

# # nlp-torch
# python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --master_feature_config pipelines/wikitext103/configs/eval_local.yaml  --num_total_samples 100000

# # nlp-hf-tf
# python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --master_feature_config pipelines/wikitext103/configs/eval_local_tf.yaml --num_total_samples 100000

# # nlp-tf
# python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --master_feature_config pipelines/wikitext103/configs/eval_local_tf_service.yaml --num_total_samples 200000

# # asr
# python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --master_feature_config pipelines/commonvoice/configs/eval_local.yaml --num_total_samples 10000

# # ssd
# python eval_cedar.py --dataset_file pipelines/coco/cedar_dataset.py --profiled_stats pipelines/coco/stats/coco_local_stats.yaml --master_feature_config pipelines/coco/configs/cedar_local_plan.yml

# python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --profiled_stats pipelines/coco/stats/coco_tf_local_stats.yaml --master_feature_config pipelines/coco/configs/cedar_tf_local_plan.yml