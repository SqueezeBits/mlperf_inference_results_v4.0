for BS in 8 16 32
do
    for MAX_CONCURRENT_REQ in 4 8 16 32 64 128
    do
        ./run_tgi_server.sh --bs $BS --scenario Server --output_dir /root/Intel-HabanaLabs/code/llama2-70b-99.9/results --max-concurrent-requests $MAX_CONCURRENT_REQ
        python run_generation.py --max_concurrent_requests $MAX_CONCURRENT_REQ &> logs/bs$BS\_requests$MAX_CONCURRENT_REQ.log
    done
done