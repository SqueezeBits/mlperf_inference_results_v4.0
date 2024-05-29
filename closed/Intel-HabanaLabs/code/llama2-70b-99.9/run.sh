datasets=(orca8192)
qps_targets=(1024)

for dataset in "${datasets[@]}"; do
    for qps in "${qps_targets[@]}"; do  
        echo "running qps $qps"
        source functions.sh
        touch $qps.start
        build_mlperf_inference --model Llama-2-7b-chat-hf --dtype bf16 --output-dir results_7b_bf16-$dataset-qps$qps --submission llama-99.9-7b-bf16-$dataset-qps$qps --skip-reqs
    done
done
