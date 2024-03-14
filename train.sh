#!/usr/bin/env bash

CFG='/home/denis/nkbtech/yolo_bbox_classification/cfg/fastervit0_real_7k_synth_7k_no_augs_v1.py'

python src/train.py -cfg $CFG
