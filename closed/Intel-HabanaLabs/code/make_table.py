import os
import re
import csv

# Variables for the path construction
models = ['Llama-3-70B', 'Llama-2-7B']
datasets = ['FIXED', 'orca']  # Add other datasets if you have more
scenarios = ['summarization', '', 'gen512']

input_lengths = [1024, 2048, 4096, 8192]  # Adjust if there are different lengths
max_batch_sizes = [256, 128, 64]
pad_seqs = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
qps_targets = [1024]

# Path to the directory containing the log files
base_dir = './llama2-70b-99.9/'  # Change this if your base directory is different

def main():
    # Summarization
    for scenario in scenarios:
        output_csv = f'latencies_{scenario}_0528.csv'
        get_table(models, datasets, input_lengths, qps_targets, max_batch_sizes, pad_seqs, base_dir, output_csv, scenario)


def get_table(models, datasets, input_lengths, qps_targets, max_batch_sizes, pad_seqs, base_dir, output_csv, scenario):
    # Prepare to write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['model', 'dataset', 'input_length', 'qps', 'batch_size', 'pad_seq', 'latency', 'generated_tokens', 'TTFT(Median)', "TTFT(Avg)", "TTFT(Max)", "TPOT(Median)", "TPOT(Avg)"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Iterate over all combinations of variables
        for model in models:
            for dataset in datasets:
                for input_length in input_lengths:
                    for qps in qps_targets:
                        for max_batch_size in max_batch_sizes:
                            for pad_seq in pad_seqs:
                                # Construct the file path
                                file_path = os.path.join(base_dir, f"results-{model}-bf16-{dataset}-{input_length}-qps{qps}-{scenario}-b{max_batch_size}-pad{pad_seq}/std_out_logs.txt")

                                try:
                                    # Open the log file
                                    # import pdb; pdb.set_trace()
                                    with open(file_path, 'r') as file:
                                        data = {'model': model, 'dataset': dataset, 'input_length': input_length, 'qps': qps, 'batch_size': max_batch_size, 'pad_seq': pad_seq}
                                        for line in file:
                                            latency = re.search(r"Test took (\d+\.\d+) sec", line)
                                            if latency:
                                                data['latency'] = int(float(latency.group(1))*1000)

                                            generated_token = re.search(r"generated (\d+) tokens", line)
                                            if generated_token:
                                                data['generated_tokens'] = int(generated_token.group(1))

                                            ttft_median = re.search(r"50.00 percentile first token latency \(ns\)   : (\d+)", line)
                                            if ttft_median:
                                                data['TTFT(Median)'] = int(int(ttft_median.group(1))/1000000)
                                                
                                            ttft_avg = re.search(r"Mean First Token latency \(ns\)               : (\d+)", line)
                                            if ttft_avg:
                                                data['TTFT(Avg)'] = int(int(ttft_avg.group(1))/1000000)
                                                
                                            ttft_max = re.search(r"Max First Token latency \(ns\)                : (\d+)", line)
                                            if ttft_max:
                                                data['TTFT(Max)'] = int(int(ttft_max.group(1))/1000000)
                                                
                                            tpot_median = re.search(r"50.00 percentile time to output token \(ns\)   : (\d+)", line)
                                            if tpot_median:
                                                data['TPOT(Median)'] = int(tpot_median.group(1))/1000000
                                                
                                            tpot_avg = re.search(r"Mean Time to Output Token \(ns\)               : (\d+)", line)
                                            if tpot_avg:
                                                data['TPOT(Avg)'] = int(tpot_avg.group(1))/1000000

                                        writer.writerow(data)
                                except FileNotFoundError:
                                    print(f"File not found: {file_path}")
                                    continue

    print(f"Data has been written to {output_csv}")


if __name__ == "__main__":
    main()