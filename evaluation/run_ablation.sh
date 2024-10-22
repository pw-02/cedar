#!/bin/bash

#cv-torch
# baseline
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_remote_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_remote.yaml --master_feature_config pipelines/simclrv2/configs/ablation_baseline.yaml --use_ray --ray_ip 10.138.0.8
#p
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_remote_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_remote.yaml --master_feature_config pipelines/simclrv2/configs/ablation_p.yaml --use_ray --ray_ip 10.138.0.8
#pr
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_remote_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_remote.yaml --master_feature_config pipelines/simclrv2/configs/ablation_p_r.yaml --use_ray --ray_ip 10.138.0.8
#pro
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_remote_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_remote.yaml --master_feature_config pipelines/simclrv2/configs/ablation_p_r_o.yaml --use_ray --ray_ip 10.138.0.8
#prof
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_remote_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_remote.yaml --master_feature_config pipelines/simclrv2/configs/eval_cedar_remote.yaml --use_ray --ray_ip 10.138.0.8

# cv-tf
#baseline
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --master_feature_config pipelines/simclrv2/configs/ablation_tf_baseline.yaml --use_ray --ray_ip 10.138.0.8
#p
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --master_feature_config pipelines/simclrv2/configs/ablation_tf_p.yaml --use_ray --ray_ip 10.138.0.8
#pr
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --master_feature_config pipelines/simclrv2/configs/ablation_tf_p_r.yaml --use_ray --ray_ip 10.138.0.8
#pro
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --master_feature_config pipelines/simclrv2/configs/ablation_tf_p_r_o.yaml --use_ray --ray_ip 10.138.0.8
#prof
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --master_feature_config pipelines/simclrv2/configs/eval_cedar_remote_tf.yaml --use_ray --ray_ip 10.138.0.8

# nlp-torch
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/configs/ablation_baseline.yaml
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/configs/ablation_p.yaml
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/configs/ablation_p_r.yaml
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/configs/ablation_p_r_o.yaml
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/configs/eval_remote.yaml

# nlp-hf-tf
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --master_feature_config pipelines/wikitext103/configs/ablation_tf_baseline.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --master_feature_config pipelines/wikitext103/configs/ablation_tf_p.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --master_feature_config pipelines/wikitext103/configs/ablation_tf_p_r.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --master_feature_config pipelines/wikitext103/configs/ablation_tf_p_r_o.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --master_feature_config pipelines/wikitext103/configs/eval_remote_tf.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000
# nlp-tf
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --master_feature_config pipelines/wikitext103/configs/ablation_baseline.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 200000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --master_feature_config pipelines/wikitext103/configs/ablation_p.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 200000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --master_feature_config pipelines/wikitext103/configs/ablation_p_r.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 200000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --master_feature_config pipelines/wikitext103/configs/ablation_p_r_o.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 200000
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --master_feature_config pipelines/wikitext103/configs/eval_remote_tf_service.yaml --use_ray --ray_ip 10.138.0.8 --num_total_samples 200000
# asr
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --master_feature_config pipelines/commonvoice/configs/ablation_baseline.yaml --num_total_samples 10000
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --master_feature_config pipelines/commonvoice/configs/ablation_p.yaml --num_total_samples 10000
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --master_feature_config pipelines/commonvoice/configs/ablation_p_r.yaml --num_total_samples 10000
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --master_feature_config pipelines/commonvoice/configs/ablation_p_r_o.yaml --num_total_samples 10000
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --master_feature_config pipelines/commonvoice/configs/eval_remote.yaml --num_total_samples 10000

# ssd-torch
python eval_cedar.py --dataset_file pipelines/coco/cedar_dataset.py --profiled_stats pipelines/coco/stats/coco_local_stats.yaml --master_feature_config pipelines/coco/configs/ablation_baseline.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_dataset.py --profiled_stats pipelines/coco/stats/coco_local_stats.yaml --master_feature_config pipelines/coco/configs/ablation_p.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_dataset.py --profiled_stats pipelines/coco/stats/coco_local_stats.yaml --master_feature_config pipelines/coco/configs/ablation_pr.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_remote_dataset.py --profiled_stats pipelines/coco/stats/coco_remote_stats.yaml --master_feature_config pipelines/coco/configs/ablation_pro.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_remote_dataset.py --master_feature_config pipelines/coco/configs/cedar_remote_plan.yml --use_ray --ray_ip 10.138.0.8

# ssd-tf
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --master_feature_config pipelines/coco/configs/ablation_tf_baseline.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --master_feature_config pipelines/coco/configs/ablation_tf_p.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --master_feature_config pipelines/coco/configs/ablation_tf_pr.yml
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --master_feature_config pipelines/coco/configs/ablation_tf_pro.yml  --use_ray --ray_ip 10.138.0.8
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --master_feature_config pipelines/coco/configs/cedar_tf_remote_plan.yml --profiled_stats pipelines/coco/stats/coco_tf_remote_stats.yaml --use_ray --ray_ip 10.138.0.26
