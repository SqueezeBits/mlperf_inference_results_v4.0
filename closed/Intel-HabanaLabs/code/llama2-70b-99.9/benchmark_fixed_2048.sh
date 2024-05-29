#!/bin/bash
datasets=(FIXED)
input_length=(2048)


# MODELS=(Llama-3-8B)
# max_batch_sizes=(64)
# PAD_SEQ=(2048) 
# MODELS=(Llama-2-7B)
# max_batch_sizes=(16)
# PAD_SEQ=(2048) 
# MODELS=(Llama-2-13B)
# max_batch_sizes=(8)
# PAD_SEQ=(2048) 
# MODELS=(Llama-2-70B)
# max_batch_sizes=(256)
# PAD_SEQ=(2048) 
MODELS=(Llama-3-70B)
max_batch_sizes=(256)
PAD_SEQ=(2048) 

# #Summarization
qps_targets=(1024)
source functions.sh
for MODEL in "${MODELS[@]}"; do
    for max_batch_size in "${max_batch_sizes[@]}"; do
        for pad_seq in "${PAD_SEQ[@]}"; do
            for dataset in "${datasets[@]}"; do
                for qps in "${qps_targets[@]}"; do  
                    echo "running qps $qps"
                    echo "Pad sequence to multiple of: $pad_seq"
                    echo "Model name: $MODEL"
                    # touch $qps.start
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
                    echo "Summarization"
                    build_mlperf_inference --model $model_dir --output-dir results-$MODEL-bf16-$dataset-$input_length-qps$qps-summarization-b$max_batch_size-pad$pad_seq \
                                            --max-batch-size $max_batch_size --pad_sequence_to_multiple_of $pad_seq --input-length $input_length --target-qps $qps \
                                            --dtype bf16 --submission $MODEL-bf16-$dataset-summarization --skip-reqs
                    echo "Model dir: $model_dir"
                    echo "Gen512"
                    build_mlperf_inference --model $model_dir --output-dir results-$MODEL-bf16-$dataset-$input_length-qps$qps-gen512-b$max_batch_size-pad$pad_seq \
                                            --max-batch-size $max_batch_size --pad_sequence_to_multiple_of $pad_seq --input-length $input_length --target-qps $qps \
                                            --dtype bf16 --submission $MODEL-bf16-$dataset-gen512 --skip-reqs
                    echo "Model dir: $model_dir"
                    echo "Generation"
                    build_mlperf_inference --model $model_dir --output-dir results-$MODEL-bf16-$dataset-$input_length-qps$qps--b$max_batch_size-pad$pad_seq \
                                            --max-batch-size $max_batch_size --pad_sequence_to_multiple_of $pad_seq --input-length $input_length --target-qps $qps \
                                            --dtype bf16 --submission $MODEL-bf16-$dataset --skip-reqs
                done
            done
        done
    done
done



