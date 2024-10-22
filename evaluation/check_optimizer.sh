#! /bin/bash

# clear the file
echo "" > all_diffs.txt

echo "COCO TF REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --profiled_stats pipelines/coco/stats/coco_tf_remote_stats.yaml --generate_plan
diff /tmp/cedar_optimized_plan.yml pipelines/coco/configs/cedar_tf_remote_plan.yml >> all_diffs.txt

echo "COCO TF LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/coco/cedar_tf_dataset.py --profiled_stats pipelines/coco/stats/coco_tf_local_stats.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/coco/configs/cedar_tf_local_plan.yml >> all_diffs.txt

echo "COCO LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/coco/cedar_dataset.py --profiled_stats pipelines/coco/stats/coco_local_stats.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/coco/configs/cedar_local_plan.yml >> all_diffs.txt

echo "COCO REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/coco/cedar_remote_dataset.py --profiled_stats pipelines/coco/stats/coco_remote_stats.yaml --generate_plan
diff /tmp/cedar_optimized_plan.yml pipelines/coco/configs/cedar_remote_plan.yml >> all_diffs.txt

echo "SIMCLR LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_stats.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/simclrv2/configs/eval_cedar_local.yaml >> all_diffs.txt

echo "SIMCLR REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_remote_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_remote.yaml --generate_plan
diff /tmp/cedar_optimized_plan.yml pipelines/simclrv2/configs/eval_cedar_remote.yaml >> all_diffs.txt

echo "SIMCLR TF REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --generate_plan
diff /tmp/cedar_optimized_plan.yml pipelines/simclrv2/configs/eval_cedar_remote_tf.yaml >> all_diffs.txt

echo "SIMCLR TF LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/simclrv2/cedar_tf_dataset.py --profiled_stats pipelines/simclrv2/stats/cedar_tf_stats.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/simclrv2/configs/eval_cedar_local_tf.yaml >> all_diffs.txt

echo "WIKITEXT LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/wikitext103/configs/eval_local.yaml >> all_diffs.txt

echo "WIKITEXT REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_dataset.py --profiled_stats pipelines/wikitext103/stats/cedar.yaml --generate_plan 
diff /tmp/cedar_optimized_plan.yml pipelines/wikitext103/configs/eval_remote.yaml >> all_diffs.txt

echo "WIKITEXT TF LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/wikitext103/configs/eval_local_tf.yaml >> all_diffs.txt

echo "WIKITEXT TF REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_dataset.py --profiled_stats pipelines/wikitext103/stats/tf.yaml --generate_plan 
diff /tmp/cedar_optimized_plan.yml pipelines/wikitext103/configs/eval_remote_tf.yaml >> all_diffs.txt

echo "WIKITEXT TF SERVICE LOCAL" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/wikitext103/configs/eval_local_tf_service.yaml >> all_diffs.txt

echo "WIKITEXT TF SERVICE REMOTE" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/wikitext103/cedar_tf_service_dataset.py --profiled_stats pipelines/wikitext103/stats/tf_service.yaml --generate_plan
diff /tmp/cedar_optimized_plan.yml pipelines/wikitext103/configs/eval_remote_tf_service.yaml >> all_diffs.txt

echo "Commonvoice local" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --generate_plan --disable_offload
diff /tmp/cedar_optimized_plan.yml pipelines/commonvoice/configs/eval_local.yaml >> all_diffs.txt

echo "Commonvoice remote" >> all_diffs.txt
python eval_cedar.py --dataset_file pipelines/commonvoice/cedar_dataset.py --profiled_stats pipelines/commonvoice/stats/cedar.yaml --generate_plan 
diff /tmp/cedar_optimized_plan.yml pipelines/commonvoice/configs/eval_remote.yaml >> all_diffs.txt