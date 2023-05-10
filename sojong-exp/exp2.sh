#!/bin/bash


# alpha value: 0.0~0.5, 0.05 - 11 types
# svdvalue: 64, 128, 256, 512, 1024 - 5 types
# svd module: sparsesvd, scipy, fbpca, sklearn-rand - 4 types
# total: 11*5*4=220 types

DEVICE="cuda:0"

svdvalue_list=(64 128 256 512 1024)
# svdvalue_list=(256)

# svdmodule_list=("sparsesvd" "scipy" "fbpca" "sklearn-rand")
svdmodule_list=("fbpca")
dataset_list=("gowalla" "yelp2018" "amazon-book")
# dataset_list=("gowalla")


# CreateDIR=./exp1_log
CreateDIR=./exp2_report_log
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
            # TEMPTIME=`date +%s.%N`
            # diff=$( echo "$TEMPTIME-$STARTTIME" | bc -l )
            # printf "                                                                        \r"
            # printf "[$curiter/$totaliter] $diff sec\r"
            # python main.py --dataset="amazon-book" --topks="[20]" --simple_model="exp2" --expdevice="cpu" --svdvalue=256 --svdtype="fbpca" --filter="linear"
            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="linear" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_linear.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="linear" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_linear.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="ideal-low-pass" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_ideal-low-pass.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="ideal-low-pass" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_ideal-low-pass.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="gaussian" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_gaussian.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="gaussian" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_gaussian.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="heat-kernel" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_heat-kernel.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="heat-kernel" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_heat-kernel.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="butterworth" --filter_option=1 > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_butterworth_1.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="butterworth" --filter_option=1 >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_butterworth_1.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="butterworth" --filter_option=2 > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_butterworth_2.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="butterworth" --filter_option=2 >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_butterworth_2.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="butterworth" --filter_option=3 > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_butterworth_3.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="butterworth" --filter_option=3 >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_butterworth_3.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="gfcf-linear-autoencoder" --filter_option="0.3" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_gfcf-linear-autoencoder_0.3.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="gfcf-linear-autoencoder" --filter_option="0.3" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_gfcf-linear-autoencoder_0.3.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="gfcf-Neighborhood-based" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_gfcf-Neighborhood-based.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="gfcf-Neighborhood-based" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_gfcf-Neighborhood-based.log"

            echo python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="sigmoid-low-pass" > "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_sigmoid-low-pass.log"
            python main.py --dataset=$dataset --topks="[20]" --simple_model="exp2" --expdevice=${DEVICE} --svdvalue=$svdvalue --svdtype=$svdmodule --filter="sigmoid-low-pass" >> "$CreateDIR/${dataset}_${svdmodule}_${svdvalue}_sigmoid-low-pass.log"
            # let curiter=$curiter+1
        done
    done
done

# check for all outputs

# printf "[$curiter/$totaliter]\n"
echo "DONE"