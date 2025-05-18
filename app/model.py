﻿from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # distribui entre GPU e CPU automaticamente
        torch_dtype=torch.float16  # usa float16 para acelerar
    )

    return tokenizer, model