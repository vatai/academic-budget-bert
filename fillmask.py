from pathlib import Path

import torch
from transformers import AutoTokenizer, PretrainedConfig

from pretraining.modeling import BertLMHeadModel

ft_ckpt_dir = "/groups/gcb50300/data/NLP/academic-training-large-ckpt/pretraining_experiment-/epoch1000000_step14823/"
ft_ckpt_dir = Path(ft_ckpt_dir)

if not ft_ckpt_dir.exists():
    ft_ckpt_dir = Path("/mnt/second/pretraining_experiment-/large")


def get_queries(mask_token):
    queries = [
        f"I like {mask_token} beer.",
        f"I like cold {mask_token}.",
        f"Cows produce {mask_token}.",
    ]
    return queries


def fill_mask(model: BertLMHeadModel, tokenizer: AutoTokenizer, query: str):
    tokens = tokenizer(query)
    input_ids = tokens["input_ids"]
    yhat = model(
        [
            None,
            torch.LongTensor([input_ids]),
            torch.LongTensor([tokens["token_type_ids"]]),
            torch.LongTensor([tokens["attention_mask"]]),
            None,
        ]
    )
    idx = yhat.argmax().item()
    mask_pos = input_ids.index(tokenizer.mask_token_id)
    input_ids[mask_pos] = idx
    decoded_tok = tokenizer.decode([idx])
    result = tokenizer.decode(input_ids)
    return f"result: {result}, new token idx {idx}, new token decoded {decoded_tok}"


for query in get_queries("[MASK]"):
    config = PretrainedConfig.from_pretrained(ft_ckpt_dir)
    model = BertLMHeadModel(config, args=ft_ckpt_dir)
    model.from_pretrained(ft_ckpt_dir, args=ft_ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ft_ckpt_dir)

    predicted = fill_mask(model, tokenizer, query)
    print(predicted)
