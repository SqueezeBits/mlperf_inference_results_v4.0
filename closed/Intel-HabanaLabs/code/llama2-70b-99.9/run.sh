# qps_targets=(1 2 3 4 5 6 7 8 9)
qps_targets=(2)
for qps in "${qps_targets[@]}"
do  
    echo "running qps $qps"
    source functions.sh
    touch $qps.start
    build_mlperf_inference --model Llama-2-7b-chat-hf --dtype bf16 --output-dir results_7b_bf16-qps$qps --submission llama-99.9-7b-bf16-qps$qps
done
