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
        f"I like [MASK] beer.",
        f"I like cold [MASK].",
        f"Cows produce [MASK].",
        f"[MASK] produce milk.",
    ]
    return queries


def fill_mask(model: BertLMHeadModel, tokenizer: AutoTokenizer, query: str):
    tokens = tokenizer(query)
    input_ids = tokens["input_ids"]
    mask_pos = input_ids.index(tokenizer.mask_token_id)
    yhat = model(
        [
            None,
            torch.LongTensor([input_ids]),
            torch.LongTensor([tokens["token_type_ids"]]),
            torch.LongTensor([tokens["attention_mask"]]),
            None,
        ]
    )
    print("TOKENS", tokens)
    print(yhat.shape)
    print(tokenizer.mask_token)
    idx = yhat[0, mask_pos].argmax().item()
    input_ids[mask_pos] = idx
    decoded_tok = tokenizer.decode([idx])
    result = tokenizer.decode(input_ids)
    return (
        f"result: {result}\n"
        f"predicted token idx {idx}\n"
        f"predicted token decoded: `{decoded_tok}`\n"
        f"mask position: {mask_pos}\n"
    )


for query in get_queries("[MASK]"):
    config = PretrainedConfig.from_pretrained(ft_ckpt_dir)
    model = BertLMHeadModel(config, args=ft_ckpt_dir)
    model.from_pretrained(ft_ckpt_dir, args=ft_ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ft_ckpt_dir)

    predicted = fill_mask(model, tokenizer, query)
    print(predicted)
