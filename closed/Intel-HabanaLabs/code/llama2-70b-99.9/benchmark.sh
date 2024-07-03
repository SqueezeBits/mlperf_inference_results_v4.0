#!/bin/bash
qps=1024

# 0: use EOS token, 1: ignore EOS token
IGNORE_EOS_TOKEN=1 
#FIXED or orca
dataset=FIXED 
input_length=1024
# bf16 or fp8
precision=bf16 

# Models
MODEL=Llama-3-8B
max_batch_sizes=(128)
PAD_SEQ=1024
# MODEL=Llama-2-7B
# max_batch_sizes=(32)
# PAD_SEQ=1024
# MODEL=Llama-2-13B
# max_batch_sizes=(8)
# PAD_SEQ=1024
# MODEL=Llama-2-70B
# max_batch_sizes=(256)
# PAD_SEQ=1024
# MODEL=Llama-3-70B
# max_batch_sizes=(256)
# PAD_SEQ=1024

# Parse command-line arguments
while getopts d:e:i:m:p:s: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        e) IGNORE_EOS_TOKEN=${OPTARG};;
        i) input_length=${OPTARG};;
        m) MODEL=${OPTARG};;
        p) precision=${OPTARG};;
        s) PAD_SEQ=${OPTARG};;
    esac
done

export IGNORE_EOS_TOKEN

source functions.sh
for max_batch_size in "${max_batch_sizes[@]}"; do
    echo "running qps $qps"
    echo "Pad sequence to multiple of: $PAD_SEQ"
    echo "Model name: $MODEL"

    if [ "$MODEL" == "Llama-2-7B" ]; then
        model_dir=Llama-2-7b-chat-hf
    elif [ "$MODEL" == "Llama-2-13B" ]; then
        model_dir=Llama-2-13b-chat-hf
    elif [ "$MODEL" == "Llama-2-70B" ]; then
        model_dir=Llama-2-70b-chat-hf
    elif [ "$MODEL" == "Llama-3-8B" ]; then 
        model_dir=Meta-Llama-3-8B-Instruct
    elif [ "$MODEL" == "Llama-3-70B" ]; then 
        model_dir=Meta-Llama-3-70B-Instruct
    else
        model_dir=unknown
    fi
    echo "Model dir: $model_dir"
    echo "prefill"
    build_mlperf_inference --model $model_dir --output-dir results-$MODEL-$precision-$dataset-$input_length-qps$qps-prefill-b$max_batch_size-pad$PAD_SEQ \
                            --max-batch-size $max_batch_size --pad_sequence_to_multiple_of $PAD_SEQ --input-length $input_length --target-qps $qps \
                            --dtype $precision --submission $MODEL-$precision-$dataset-prefill --skip-reqs
    echo "Model dir: $model_dir"
    echo "decode"
    build_mlperf_inference --model $model_dir --output-dir results-$MODEL-$precision-$dataset-$input_length-qps$qps-decode-b$max_batch_size-pad$PAD_SEQ \
                            --max-batch-size $max_batch_size --pad_sequence_to_multiple_of $PAD_SEQ --input-length $input_length --target-qps $qps \
                            --dtype $precision --submission $MODEL-$precision-$dataset-decode --skip-reqs
done
