###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import array
import os
import mlperf_loadgen as lg
import pandas as pd
import threading
import struct

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
import torch
import random

def except_hook(args):
    print(f"Thread failed with error: {args.exc_value}")
    os._exit(1)


threading.excepthook = except_hook


def load_dataset(dataset_path):
    tok_input = pd.read_pickle(dataset_path)['tok_input'].tolist()
    ret = []
    for sample in tok_input:
        ret.append((len(sample), ','.join(str(token) for token in sample)))
    return ret

def generate_random_dataset(sample_count, length, tokenizer_path):
    """Generates a random dataset where each sample is of the specified length."""
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = []
    for _ in range(sample_count):
        sample = ",".join(str(random.randint(3, tokenizer.vocab_size - 1)) for _ in range(length))
        dataset.append((length, sample))
        
    return dataset

class Dataset():
    def __init__(self, dataset_path, total_sample_count, tokenizer_path, random_data_length):
        if dataset_path is None:
            print("Random dataset is generated.")
            self.data = generate_random_dataset(sample_count=total_sample_count, length=random_data_length, tokenizer_path=tokenizer_path)
        else:
            self.data = load_dataset(dataset_path)
        self.count = min(len(self.data), total_sample_count)

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass


class SUT_base():
    def __init__(self, args):
        self.data_object = Dataset(dataset_path=args.dataset_path,
                                   total_sample_count=args.total_sample_count,
                                   tokenizer_path=args.tokenizer_path,
                                   random_data_length=args.random_data_length
                                   )
        self.qsl = lg.ConstructQSL(self.data_object.count, args.total_sample_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)
        self.tgi_semaphore = threading.Semaphore(args.max_num_threads)
        self.client = InferenceClient(args.sut_server)
        self.max_new_tokens = args.max_new_tokens
        self.gen_tok_lens = []

    def flush_queries(self):
        pass


class Server(SUT_base):
    def __init__(self, args):
        SUT_base.__init__(self, args)

    def tgi_request(self, sample):
        _, str_input = self.data_object.data[sample.index]
        res_stream = self.client.text_generation(
            str_input, max_new_tokens=self.max_new_tokens, stream=True, details=True)
        out = []
        for res_token_id, res_token in enumerate(res_stream):
            res_token = res_token.token
            if res_token_id == 0:
                arr = array.array('B', struct.pack('L', res_token.id))
                buf_info = arr.buffer_info()
                lg.FirstTokenComplete([lg.QuerySampleResponse(
                    sample.id, buf_info[0], buf_info[1] * arr.itemsize, 1)])
            out.append(res_token.id)

        arr = array.array('B', struct.pack('L' * len(out), *out))
        buf_info = arr.buffer_info()

        lg.QuerySamplesComplete([lg.QuerySampleResponse(
            sample.id, buf_info[0], buf_info[1] * arr.itemsize, len(out))])
        self.gen_tok_lens.append(len(out))
        self.tgi_semaphore.release()

    def issue_queries(self, query_samples):
        for sample in query_samples:
            self.tgi_semaphore.acquire()
            threading.Thread(target=self.tgi_request, args=[sample]).start()


class Offline(SUT_base):
    def __init__(self, args):
        SUT_base.__init__(self, args)

    def tgi_request(self, sample):
        _, str_input = self.data_object.data[sample.index]
        res = self.client.text_generation(
            str_input, max_new_tokens=1024, stream=False, details=True)
        out = [token.id for token in res.details.tokens]
        arr = array.array('B', struct.pack('L' * len(out), *out))
        buf_info = arr.buffer_info()

        lg.QuerySamplesComplete([lg.QuerySampleResponse(
            sample.id, buf_info[0], buf_info[1] * arr.itemsize, len(out))])
        self.gen_tok_lens.append(len(out))
        self.tgi_semaphore.release()

    def issue_queries(self, query_samples):
        query_samples.sort(key=lambda s: self.data_object.data[s.index][0])
        for sample in query_samples:
            self.tgi_semaphore.acquire()
            threading.Thread(target=self.tgi_request, args=[sample]).start()
