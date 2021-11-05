from pathlib import Path

import torch
from transformers import AutoTokenizer, BertForMaskedLM, PretrainedConfig

from pretraining.modeling import BertLMHeadModel

ft_ckpt_dir = "/groups/gcb50300/data/NLP/academic-training-large-ckpt/pretraining_experiment-/epoch1000000_step14823/"
ft_ckpt_dir = Path(ft_ckpt_dir)

if not ft_ckpt_dir.exists():
    ft_ckpt_dir = Path("/mnt/second/pretraining_experiment-/large")


def get_queries(mask_token):
    queries = [
        "I like [MASK] beer.",
        "I like cold [MASK].",
        "Cows produce [MASK].",
        "[MASK] produce milk.",
        "Chocolate is [MASK].",
    ]
    return queries


hf_model = BertForMaskedLM.from_pretrained("bert-base-uncased")


def fill_mask(model: BertLMHeadModel, tokenizer: AutoTokenizer, query: str):
    tokens = tokenizer([query], return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    mask_pos = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    yhat = model(
        [
            None,
            tokens["input_ids"],
            tokens["token_type_ids"],
            tokens["attention_mask"],
            None,
        ]
    )
    yhat = hf_model(**tokens).logits"
    print("TOKENS", tokens)
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


model = BertLMHeadModel.from_pretrained(ft_ckpt_dir, args=ft_ckpt_dir)
tokenizer = AutoTokenizer.from_pretrained(ft_ckpt_dir)
for query in get_queries("[MASK]"):

    predicted = fill_mask(model, tokenizer, query)
    print(predicted)
