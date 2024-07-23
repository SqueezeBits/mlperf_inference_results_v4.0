# Benchmarking Guide

Check out the [original README](./README_org.md) for more detailed information on setup and configuration.

## Setup Environment
```bash
bash run_container.sh

cd llama
bash setup_tgi.sh
cd -
```

## Benchmarking Instructions

Follow the guidelines provided in the original README, excluding the model download part if the downloaded models can be mounted.

### Running the Benchmark

You can run the benchmark using Llama-3-8B-Instruct(default) model and data type by executing the following command:

```bash
bash benchmark.sh -b <decode_batch_size> -d <dataset> -e <ignore_eos_token> -i <input_data_length> -m <model> -p <precision> -s <pad_sequence_length>
```

### Script Options

The `benchmark.sh` script supports various command-line options to customize your benchmarking. Here are the available flags:

- `-b`: Set the maximum batch size (default is 128 for Llama-3-8B)
- `-d`: Specify the dataset to use (default is `FIXED`)
- `-e`: Set the IGNORE_EOS_TOKEN flag (0 to use EOS token, 1 to ignore EOS token)
- `-i`: Define the input length (default is 1024)
- `-m`: Select the model to use (e.g., `Llama-2-7B`, `Llama-2-13B`, `Llama-2-70B`, `Llama-3-8B`, `Llama-3-70B`)
- `-p`: Choose the precision type (either `bf16` or `fp8`)
- `-s`: Set the padding sequence to a multiple of (default is 1024)

### Example Usage

Below is an example usage of the `benchmark.sh` script:

```bash
bash benchmark.sh -b 128 -d FIXED -e 1 -i 1024 -m Llama-3-8B -p bf16 -s 1024
```

This command runs the benchmark using the Llama-3-8B model with a batch size of 128, using the FIXED dataset, ignoring the EOS token, with an input data length of 1024, using bf16 precision, and padding sequences to a multiple of 1024.


### Notes

- By default, the script executes both the prefill phase (output_token_len=1) and the decode phase (output_token_len=1024) sequentially. 
- If you only need to run one of these phases, you can modify the benchmark script accordingly to suit your needs.
- If you want to set the 'input_length' to 'max_input_length' - 1, you can add -1 at line #274 in the 'run_mlperf_scenarios.py' file.
- Ensure that the appropriate models are available and properly mounted if not downloaded as part of the script.
- Adjust the parameters as needed to suit your specific benchmarking requirements.

For further details and troubleshooting, refer to the [original README](./README_org.md).
