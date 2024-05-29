import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/scratch-1/models/Meta-Llama-3-70B-Instruct")

output_dir = '/scratch-1/models/Meta-Llama-3-70B-Instruct/model_bin'
model.save_pretrained(output_dir, max_shard_size="20GB", safe_serialization=False)

print(f"Model successfully saved to {output_dir}")

