#!/bin/bash


# alpha value: 0.0~0.5, 0.05 - 11 types
# svdvalue: 64, 128, 256, 512, 1024 - 5 types
# svd module: sparsesvd, scipy, fbpca, sklearn-rand - 4 types
# total: 11*5*4=220 types

DEVICE="cuda:0"

alpha_start=0.0
alpha_end=1.0
alpha_step=0.1

svdvalue_list=(64 128 256 512 1024)
# svdvalue_list=(256)

# svdmodule_list=("sparsesvd" "scipy" "fbpca" "sklearn-rand")
svdmodule_list=("fbpca")
dataset_list=("gowalla" "yelp2018" "amazon-book")
# dataset_list=("gowalla")


# CreateDIR=./exp1_log
CreateDIR=./exp1_report_log
if [ ! -d $CreateDIR ]; then
    mkdir $CreateDIR
else
    rm -rf $CreateDIR/*
fi

# check for all values
let totaliter=${#svdvalue_list[*]}*${#svdmodule_list[*]}*${#dataset_list[*]}
echo "totaliter: " $totaliter

curiter=0

STARTTIME=`date +%s.%N`

for dataset in "${dataset_list[@]}"; do
    for svdmodule in "${svdmodule_list[@]}"; do
        for svdvalue in "${svdvalue_list[@]}"; do
            TEMPTIME=`date +%s.%N`
            diff=$( echo "$TEMPTIME-$STARTTIME" | bc -l )
            printf "                                                                        \r"
            printf "[$curiter/$totaliter] $diff sec\r"
            # python main.py --dataset="amazon-book" --topks="[20]" --simple_model="exp1" --expdevice="cuda:0" --svdvalue=256 --svdtype="sparsesvd" --alpha_start=0.3 --alpha_end=0.3 --alpha_step=0.05
            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp1" --expdevice=$DEVICE --svdvalue=$svdvalue --svdtype=$svdmodule --alpha_start=$alpha_start --alpha_end=$alpha_end --alpha_step=$alpha_step > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_${alpha_start}_${alpha_end}_${alpha_step}.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp1" --expdevice=$DEVICE --svdvalue=$svdvalue --svdtype=$svdmodule --alpha_start=$alpha_start --alpha_end=$alpha_end --alpha_step=$alpha_step >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_${alpha_start}_${alpha_end}_${alpha_step}.log"
            let curiter=$curiter+1
        done
    done
done

# check for all outputs

printf "[$curiter/$totaliter]\n"
echo "DONE"