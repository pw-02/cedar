#!/bin/bash

#cv-tf
python eval_app_runner.py simclr_app.py simclr ff config.yaml
#nlp-tf
python eval_app_runner.py nlp_app.py nlp ff config.yaml
#ssd-tf
python eval_app_runner.py coco_app.py coco ff config.yaml