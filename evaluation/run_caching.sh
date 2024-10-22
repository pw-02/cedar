#!/bin/bash

# Note, we provide the exact stats and config files produced by the optimizer on our setup in order to enable reproducibility.
# To generate new profiling stats and re-run the optimizer, use the --run_profiling and --generate_plan flags in eval_cedar.py.
# Replace the stats and optimizer-produced config in the following commands

# cv-torch
# with caching
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_cache_dataset.py --master_feature_config pipelines/simclrv2/cache_results/configs/new_simclrv2_optimized_plan.yml --use_ray --ray_ip 10.138.0.8
# without caching
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_cache_dataset.py --master_feature_config pipelines/simclrv2/cache_results/configs/no_cache_plan.yml --use_ray --ray_ip 10.138.0.8

# nlp-torch
# with caching
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_cache_dataset.py --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/cache_results/configs/new_wikitext_optimal_cache_plan.yml
# without caching
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_cache_dataset.py --use_ray --ray_ip 10.138.0.8 --num_total_samples 100000 --master_feature_config pipelines/wikitext103/cache_results/configs/wikitext_no_caching_plan.yml

# asr
# without caching note that the optimizer generates the optimal plan, which does not cache
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_cache_dataset.py --master_feature_config pipelines/commonvoice/cache_results/configs/no_caching_eval_remote.yml --num_total_samples 10000 