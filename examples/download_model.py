import argparse
import sys
import os
import torch
import transformers
from transformers import OPTForCausalLM
if __name__ == '__main__':
    model_path = '../../models/opt-1.3b'
    if not os.path.exists(os.path.dirname(model_path)):
        os.mkdir(model_path)
    model = OPTForCausalLM.from_pretrained(
        'facebook/opt-1.3b', torch_dtype=torch.float16, device_map="auto", torchscript=True, trust_remote_code=True
    )  # torchscript will force `return_dict=False` to avoid jit errors
    print("Loaded model")

    model.save_pretrained(model_path)

    print("Model downloaded and Saved in : ", model_path)