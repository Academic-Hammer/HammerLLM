#!/bin/bash

folders=(2021-43 2022-05 2022-21 2022-27 2022-33 2022-49 2023-06 2023-14 2023-23)

for folder in ${folders[@]}:
do
        for index in $(seq 0 18)
        do
                part=$(printf "%04d" $index)
                echo "========== begin to download file $folder $part =========="
                url="https://hf-mirror.com/datasets/CASIA-LM/ChineseWebText/resolve/main/cleaner_subset/${folder}/part-${part}.jsonl?download=true"
                echo $url
                wget $url -O ${folder}_part_${part}.jsonl
                echo $folder
        done
done
exit
