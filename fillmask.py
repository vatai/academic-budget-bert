from transformers import pipeline, AutoTokenizer, PretrainedConfig, BertForMaskedLM

from pretraining.modeling import BertLMHeadModel


ft_ckpt_dir = "/groups/gcb50300/data/NLP/academic-training-large-ckpt/pretraining_experiment-/epoch1000000_step14823/"

config = PretrainedConfig.from_pretrained(ft_ckpt_dir)
model = BertLMHeadModel(config, args=ft_ckpt_dir)
model.from_pretrained(ft_ckpt_dir, args=ft_ckpt_dir)
model.__class__ = BertForMaskedLM
print("MODEL NAME", model.__class__)
tokenizer = AutoTokenizer.from_pretrained(ft_ckpt_dir)

def get_queries(mask_token):
    queries = [
        f"I like {mask_token} beer.",
        f"I like cold {mask_token}.",
        f"Cows produce {mask_token}.",
    ]
    return queries


def fill_mask(model, tokenizer, query):
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer)
    result = fill_mask(query)
    return result


for query in get_queries("[MASK]"):
    predicted = fill_mask(model, tokenizer, query)
    print(predicted)

