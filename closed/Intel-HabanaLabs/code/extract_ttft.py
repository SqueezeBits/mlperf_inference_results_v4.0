import os
import glob
import re
import numpy as np

def extract_inference_times(log_file):
    """Extracts inference times from the log file."""
    inference_times = []
    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            if 'inference_time' in line:
                match = re.search(r'([\d.]+)m?s"\s*\S*time_per_token', line)
                if match:
                    inference_time_str = match.group(1).strip()
                    inference_times.append(float(inference_time_str))
                    
    return inference_times

def get_statistics(inference_times):
    """Calculates mean, median, max, and min statistics for the inference times."""
    prefill_batch=4
    if not inference_times:
        return None, None, None
    mean_val = np.mean(inference_times)/prefill_batch
    median_val = np.median(inference_times)/prefill_batch
    max_val = [inference_times[i]/prefill_batch for i in range(5)]
    min_val = np.min(inference_times)/prefill_batch
    return mean_val, median_val, max_val, min_val

def adjust_inference_times(inference_times):
    """Adjusts inference times based on a threshold."""
    return [time * 1000 if time < 10 else time for time in inference_times]

def process_directory(directory):
    """Processes all log files in the given directory to extract and analyze inference times."""
    # all_inference_times = []
    
    for log_file in glob.glob(os.path.join(directory, '**', 'text-generation-launcher.log'), recursive=True):
        inference_times = extract_inference_times(log_file)
        # all_inference_times.extend(inference_times)
    
    adjusted_times = sorted(adjust_inference_times(inference_times[-1020:]), reverse=True)
    mean_val, median_val, max_vals, min_val = get_statistics(adjusted_times)
    
    print(f"Directory: {directory}")
    # print(f"Total Inference Times: {len(all_inference_times)}")
    print(f"Median Inference Time: {median_val:.6f} ms")
    print(f"Mean Inference Time: {mean_val:.6f} ms")
    print(f"Max Inference Times: {max_vals} ms")
    print(f"Min Inference Time: {min_val:.6f} ms")


if __name__ == "__main__":    
    base_dir = '/home/sdp/works/jongho/projects/mlperf_inference_results_v4.0/closed/Intel-HabanaLabs/code/llama2-70b-99.9/benchmark_results/'
    models = ['Llama-3-70B']
    datasets = ['orca']
    input_lengths = ['1024', '2048', '4096', '8192']
    scenarios = ['summarization']
    decode_batch_pad_map = {
        '1024': ('b256', 'pad32'),
        '2048': ('b256', 'pad32'),
        '4096': ('b128', 'pad64'),
        '8192': ('b64', 'pad1024')
    }

    directories = [
        f'results-{model}-bf16-{dataset}-{input_length}-qps1024-{scenario}-{decode_batch_pad_map[input_length][0]}-{decode_batch_pad_map[input_length][1]}'
        for model in models
        for dataset in datasets
        for input_length in input_lengths
        for scenario in scenarios
    ]
    for directory in directories:
        full_directory = os.path.join(base_dir, directory)
        if os.path.exists(full_directory):
            process_directory(full_directory)
        else:
            print(f"Directory does not exist: {full_directory}")

