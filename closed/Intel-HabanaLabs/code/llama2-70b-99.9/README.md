Check out the [original README](./README_org.md)

## Benchmark
Follow the guides in the original README. (exclude the model download part if downloaded models can be mounted)

Run benchmark with target model and dtype like below.
```bash
build_mlperf_inference --model Llama-2-70b-chat-hf --dtype fp8 --output-dir results_70b_fp8 --submission llama-99.9-70b-fp8
```
Note that batch size is fixed in [scenarios.yaml](./scenarios.yaml). Modify batch size in the file if required. And `--submission` must be set according to which scenario to run along the choices in the [scenarios.yaml](./scenarios.yaml).