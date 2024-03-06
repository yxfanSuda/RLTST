#!/usr/bin/env bash

train_mol_file="./Data/Molweni/train.json"
eval_mol_file="./Data/Molweni/dev.json"
test_mol_file="./Data/Molweni/test.json"
train_hu_file='./Data/Hu_Dataset/Hu_Link_Dir/train.json'
train_ou5_file='./Data/Ou_Dataset/Ou5_Link_Dir/train.json'
train_ou10_file='./Data/Ou_Dataset/Ou10_Link_Dir/train.json'
train_ou15_file='./Data/Ou_Dataset/Ou15_Link_Dir/train.json'
dataset_dir="./dataset_dir"
model_dir="./model_dir"
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} python main.py --train_mol_file=$train_mol_file \
                                    --eval_mol_file=$eval_mol_file \
                                    --test_mol_file=$test_mol_file \
                                    --dataset_dir=$dataset_dir \
                                    --eval_mol_pool_size 50 \
                                    --TST_model_path "${model_dir}/model_TST" \
                                    --num_layers 1 --max_edu_dist 16 \
