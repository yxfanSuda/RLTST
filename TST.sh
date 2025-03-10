#!/usr/bin/env bash

train_mol_file="./Data/Molweni/train.json"
eval_mol_file="./Data/Molweni/dev.json"
test_mol_file="./Data/Molweni/test.json"
train_hu_file='./Data/Hu_Dataset/Hu_Link_Dir/selectedData.json'
train_ou5_file='./Data/Ou_Dataset/Ou5_Link_Dir/selectedData.json'
train_ou10_file='./Data/Ou_Dataset/Ou10_Link_Dir/selectedData.json'
train_ou15_file='./Data/Ou_Dataset/Ou15_Link_Dir/selectedData.json'
dataset_dir="./selected_dataset_dir"
model_dir="./model_dir"
seed=65534 # Three independent runs with different random seeds (65534, 42, and 65535)are conducted and he mean performance is reported.

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}  nohup python  -u main.py \
                                    --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --train_hu_file=$train_hu_file \
                                    --train_ou5_file=$train_ou5_file \
                                    --train_ou10_file=$train_ou10_file \
                                    --train_ou15_file=$train_ou15_file \
                                    --dataset_dir=$dataset_dir  \
                                    --do_train \
                                    --debug \
                                    --ST_model_path "${model_dir}/model_ST" \
                                    --TST_model_path "${model_dir}/model_TST" \
                                    --seed seed > TST.log 2>&1 &
