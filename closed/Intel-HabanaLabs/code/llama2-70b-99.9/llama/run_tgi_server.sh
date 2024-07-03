#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -e

# Kill TGI if present
killall -q -15 text-generation-launcher && sleep 60

usage() {
    echo "Usage: $0 --model model --bs batch_size"
    echo "Options:"
    echo "  --model, -m             Specify the model"
    echo "  --bs                    Specify the batch size"
    echo "  --scenario              Specify the scenario, possible values: Offline, Server"
    echo "  --fp8                   Use the fp8 quantization"
    echo "  --output_dir, -o        Specify the output dir for logs if RESULT_DIR is not set, default: ./results"
    echo "  --help                  Display this help message"
    echo "  --max-total-tokens      Maximum total tokens (default: 2048)"
    echo "  --max-input-length      Maximum input length (default: 1024)"
    exit 1
}

wait_for_server() {
    local model="$1"

    timeout=3600
    step=10
    current_time=0

    set +x
    while [ "$current_time" -lt "$timeout" ]; do
        output=$(curl -s http://localhost:8080/info | grep $model_name | wc -l)
        if (( $output > 0 )); then
            set -x
            return
        fi
        sleep $step
        current_time=$((current_time + step))
    done

    set -x
    echo "TGI server didn't start"
    exit -1
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            model=$2
            shift 2
            ;;
        --bs)
            batch_size=$2
            shift 2
            ;;
        --scenario)
            scenario=$2
            shift 2
            ;;
        --max_total_tokens)
            max_total_tokens=$2
            shift 2
            ;;
        --max_input_length)
            max_input_length=$2
            shift 2
            ;;
        --pad_sequence_to_multiple_of)
            pad_sequence_to_multiple_of=$2
            shift 2
            ;;
        --fp8)
            fp8=true
            shift
            ;;
        --output_dir|-o)
            output_dir=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

MAX_INPUT_SEQ_LEN=${max_input_length:-1024}
MAX_TOTAL_TOKENS=${max_total_tokens:-2048}
PAD_SEQUENCE_TO_MULTIPLE_OF=${pad_sequence_to_multiple_of:-32}

if [[ -n $HELP || -z $model || -z $batch_size || -z $scenario ]]; then
    usage
fi

script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
output_dir=${RESULT_DIR:-$output_dir}
output_dir=${output_dir:-$script_dir/results}
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

if [ "$fp8" = true ]; then
    if [ "$model" = "70b" ]; then
        export QUANT_CONFIG=hqt/llama2-70b-8x/config_meas_maxabs_quant_MAXABS_HW.json
    elif [ "$model" = "70b3" ]; then
        export QUANT_CONFIG=hqt/llama3-70b-8x/config_meas_maxabs_quant_MAXABS_HW.json
    elif [ "$model" = "7b" ]; then
        export QUANT_CONFIG=hqt/llama2-7b-chat/config_meas_maxabs_quant_MAXABS_HW.json
    elif [ "$model" = "8b" ]; then
        export QUANT_CONFIG=hqt/llama3-8b-instruct/config_meas_maxabs_quant_MAXABS_HW.json
    elif [ "$model" = "13b" ]; then
        export QUANT_CONFIG=hqt/llama2-13b-chat/config_meas_maxabs_quant_MAXABS_HW.json
    else
        export QUANT_CONFIG=
    fi
fi

waiting_served_ratio=0.006
if [ "$scenario" = "Offline" ]; then
    if [ "$fp8" = true ]; then
        prefill_batch_size=16
        PREFILL_BATCH_BUCKET_SIZE=16
        waiting_served_ratio=0.017
        export PAD_SEQUENCE_TO_MULTIPLE_OF=64
    else
        prefill_batch_size=4
    fi
elif [ "$fp8" = true ]; then
    prefill_batch_size=2
    PREFILL_BATCH_BUCKET_SIZE=2
else
    prefill_batch_size=2
fi

source "$HOME/.cargo/env"
export MAX_INPUT_SEQ_LEN
export  MAX_TOTAL_TOKENS
export PAD_SEQUENCE_TO_MULTIPLE_OF

PT_HPU_ENABLE_LAZY_COLLECTIVES=${PT_HPU_ENABLE_LAZY_COLLECTIVES:-true}
export PT_HPU_ENABLE_LAZY_COLLECTIVES
SKIP_TOKENIZER_IN_TGI=${SKIP_TOKENIZER_IN_TGI:-true}
export SKIP_TOKENIZER_IN_TGI
PREFILL_BATCH_BUCKET_SIZE=${PREFILL_BATCH_BUCKET_SIZE:-1}
export PREFILL_BATCH_BUCKET_SIZE
export BATCH_BUCKET_SIZE=$batch_size
max_batch_total_tokens=$(($batch_size*$MAX_TOTAL_TOKENS))
max_batch_prefill_tokens=$(($prefill_batch_size*$MAX_INPUT_SEQ_LEN))

if [ "$model" = "70b" ]; then
    sharding_options="--sharded true --num-shard 8"
    model_name=Llama-2-70b-chat-hf
elif [ "$model" = "70b3" ]; then
    sharding_options="--sharded true --num-shard 8"
    model_name=Meta-Llama-3-70B-Instruct
elif [ "$model" = "7b" ]; then
    # sharding_options="--sharded true --num-shard 8"
    sharding_options=""
    model_name=Llama-2-7b-chat-hf
elif [ "$model" = "8b" ]; then
    sharding_options=""
    model_name=Meta-Llama-3-8B-Instruct
elif [ "$model" = "13b" ]; then
    sharding_options=""
    model_name=Llama-2-13b-chat-hf
else
    sharding_options=""
fi


text-generation-launcher --port 8080 \
    --model-id /mnt/weka/data/pytorch/llama2/$model_name $sharding_options\
    --max-total-tokens $MAX_TOTAL_TOKENS --max-input-length $MAX_INPUT_SEQ_LEN \
    --max-batch-prefill-tokens $max_batch_prefill_tokens --max-batch-total-tokens $max_batch_total_tokens \
    --shard-uds-path /tmp/text-generation-server-$scenario \
    --max-concurrent-requests 1024 --max-waiting-tokens 20 --waiting-served-ratio $waiting_served_ratio \
    --dtype bfloat16 &>> ${output_dir}/text-generation-launcher.log &

wait_for_server "${model_name}"
