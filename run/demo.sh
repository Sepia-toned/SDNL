#!/usr/bin/env bash
# This script replicates tables reported in the roland paper.
# Author: Tianyu Du
# Last modified: June 8, 2022

# put the path of dataset here.

# DATA_PATH='/root/autodl-tmp/demo/roland_public_data/multi_social_networks'
DATA_PATH='/home/ly/SDNL/roland_public_data'

# DATA_PATH='/lfs/hyperturing1/0/tianyudu/roland_public_data'

# =============================================================================
# Table 3 Bottom Panel.
# =============================================================================
for file in $(ls -d ./replication_configs/table3_bottom/*.yaml)
    do
    python3 main.py --cfg ${file} --repeat 1 --override_data_dir ${DATA_PATH} --override_remark 'table3_bottom'
done