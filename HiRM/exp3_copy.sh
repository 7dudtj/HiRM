#!/bin/bash


# alpha value: 0.0~0.5, 0.05 - 11 types
# svdvalue: 64, 128, 256, 512, 1024 - 5 types
# svd module: sparsesvd, scipy, fbpca, sklearn-rand - 4 types
# total: 11*5*4=220 types

DEVICE="cuda:0"

# svdvalue_list=(128 256 512)
# svdvalue_list=(256)
svdvalue_list=(512)

# svdmodule_list=("sparsesvd" "scipy" "fbpca" "sklearn-rand")
svdmodule_list=("torch_cuda")
dataset_list=("gowalla" "yelp2018" "amazon-book")
# dataset_list=("lastfm")
alpha_start=-1
alpha_end=2
alpha_step=0.05
filter_list=("['ideal-low-pass']" "['gaussian']" "['heat-kernel']" "['butterworth', 1]" "['butterworth', 2]" "['butterworth', 3]" "['gfcf-linear-autoencoder']" "['gfcf-Neighborhood-based']" "['inverse']" "['sigmoid-low-pass']")



# CreateDIR=./exp1_log
CreateDIR=./exp3_report_log
if [ ! -d $CreateDIR ]; then
    mkdir $CreateDIR
else
    rm -rf $CreateDIR/*
fi

# check for all values
# let totaliter=${#svdvalue_list[*]}*${#svdmodule_list[*]}*${#dataset_list[*]}
# echo "totaliter: " $totaliter

# curiter=0

# STARTTIME=`date +%s.%N`

for dataset in "${dataset_list[@]}"; do
    for svdmodule in "${svdmodule_list[@]}"; do
        for svdvalue in "${svdvalue_list[@]}"; do
            for filter in "${filter_list[@]}"; do
                today=`date`
                echo "$today : ${dataset}_${svdmodule}_${svdvalue}_${alpha_start}_${alpha_end}_${alpha_step}_${filter}.log"
                python main.py --dataset="$dataset" --topks="[20]" --simple_model="exp3" --expdevice="$DEVICE" --svdvalue=$svdvalue --svdtype="$svdmodule" --alpha_start=$alpha_start --alpha_end=$alpha_end --alpha_step=$alpha_step --filter="$filter" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_${alpha_start}_${alpha_end}_${alpha_step}_${filter}.log"
            done
        done
    done
done

# check for all outputs

# printf "[$curiter/$totaliter]\n"
echo "DONE"