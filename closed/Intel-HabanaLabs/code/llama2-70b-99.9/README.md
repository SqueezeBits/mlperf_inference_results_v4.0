Check out the [original README](./README_org.md)

## Benchmark
Follow the guides in the original README. (exclude the model download part if downloaded models can be mounted)

Run benchmark with target model and dtype like below.
```bash
build_mlperf_inference --model Llama-2-70b-chat-hf --dtype fp8 --output-dir results_70b_fp8 --submission llama-99.9-70b-fp8
build_mlperf_inference --model Llama-2-7b-chat-hf --dtype bf16 --output-dir results_7b_bf16 --submission llama-99.9-7b-bf16
```
Note that batch size is fixed in [scenarios.yaml](./scenarios.yaml). Modify batch size in the file if required. And `--submission` must be set according to which scenario to run along the choices in the [scenarios.yaml](./scenarios.yaml).