#!/usr/bin/env bash

train_mol_file="./Data/Molweni/train.json"
eval_mol_file="./Data/Molweni/dev.json"
test_mol_file="./Data/Molweni/test.json"
train_hu_file='./Data/Hu_Dataset/Hu_Link_Dir/debug.json'
train_ou5_file='./Data/Ou_Dataset/Ou5_Link_Dir/debug.json'
train_ou10_file='./Data/Ou_Dataset/Ou10_Link_Dir/debug.json'
train_ou15_file='./Data/Ou_Dataset/Ou15_Link_Dir/debug.json'
hu_selected_data_file='./Data/Hu_Dataset/Hu_Link_Dir/selectedData.json'
ou5_selected_data_file='./Data/Ou_Dataset/Ou5_Link_Dir/selectedData.json'
ou10_selected_data_file='./Data/Ou_Dataset/Ou10_Link_Dir/selectedData.json'
ou15_selected_data_file='./Data/Ou_Dataset/Ou15_Link_Dir/selectedData.json'
hu_select_id_file='./SelectedDataIds/hu_select_id_file.txt'
ou5_select_id_file='./SelectedDataIds/hu_select_id_file.txt'
ou10_select_id_file='./SelectedDataIds/hu_select_id_file.txt'
ou15_select_id_file='./SelectedDataIds/hu_select_id_file.txt'
dataset_dir="./dataset_dir"
model_dir="./model_dir"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
CUDA_VISIBLE_DEVICES=${GPU}  nohup python  -u RL_main.py \
                                    --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --train_hu_file=$train_hu_file \
                                    --train_ou5_file=$train_ou5_file \
                                    --train_ou10_file=$train_ou10_file \
                                    --train_ou15_file=$train_ou15_file \
                                    --hu_selected_data_file=$hu_selected_data_file\
                                    --ou5_selected_data_file=$ou5_selected_data_file\
                                    --ou10_selected_data_file=$ou10_selected_data_file\
                                    --ou15_selected_data_file=$ou15_selected_data_file\
                                    --hu_select_id_file=$hu_select_id_file\
                                    --ou5_select_id_file=$ou5_select_id_file\
                                    --ou10_select_id_file=$ou10_select_id_file\
                                    --ou15_select_id_file=$ou15_select_id_file\
                                    --dataset_dir=$dataset_dir  \
                                    --do_train \
                                    --debug \
                                    --ST_model_path "${model_dir}/model_ST" \
                                    --seed 65534 > select_data.log 2>&1 &