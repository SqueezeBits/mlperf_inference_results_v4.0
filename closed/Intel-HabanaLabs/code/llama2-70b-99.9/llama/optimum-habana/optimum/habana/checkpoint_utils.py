import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode


def get_repo_root(model_name_or_path, local_rank=-1, token=None):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    if Path(model_name_or_path).is_dir():
        # If it is a local model, no need to download anything
        return model_name_or_path
    else:
        # Checks if online or not
        if is_offline_mode():
            if local_rank == 0:
                print("Offline mode: forcing local_files_only=True")

        # Only download PyTorch weights by default
        allow_patterns = ["*.bin"]

        # Download only on first process
        if local_rank in [-1, 0]:
            cache_dir = snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                max_workers=16,
                token=token,
            )
            if local_rank == -1:
                # If there is only one process, then the method is finished
                return cache_dir

        # Make all processes wait so that other processes can get the checkpoint directly from cache
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            token=token,
        )


def get_checkpoint_files(model_name_or_path, local_rank, token=None):
    """
    Gets the list of files for the specified model checkpoint.
    """
    cached_repo_dir = get_repo_root(model_name_or_path, local_rank=local_rank, token=token)

    # Extensions: .bin | .pt
    # Creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]") if entry.is_file()]
    return file_list


def write_checkpoints_json(model_name_or_path, local_rank, f, token=None):
    """
    Dumps metadata into a JSON file for DeepSpeed-inference.
    """
    checkpoint_files = get_checkpoint_files(model_name_or_path, local_rank, token)
    data = {"type": "ds_model", "checkpoints": checkpoint_files, "version": 1.0}
    json.dump(data, f)
    f.flush()


def model_on_meta(config):
    """
    Checks if load the model to meta.
    """
    return config.model_type in ["bloom", "llama"]


def model_is_optimized(config):
    """
    Checks if the given config belongs to a model in optimum/habana/transformers/models, which has a
    new input token_idx.
    """
    return True

def get_ds_injection_policy(config):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    policy = {LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")}
    return policy
