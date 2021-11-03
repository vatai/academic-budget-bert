#!/usr/bin/env python3
import collections
from pathlib import Path

import torch
import torch.nn
from transformers import (BertConfig, BertForMaskedLM,
                          BertForSequenceClassification, BertModel)


def rename(key: str, last_layer_idx: int):
    words = key.split(".")
    if words[-2] == "dense_act":
        words[-2] = "dense"
    elif words[-2] == "PreAttentionLayerNorm":
        num = str(int(words[-3]) - 1)
        words = words[:-3] + [num, "output.LayerNorm"] + words[-1:]
    elif words[-2] == "PostAttentionLayerNorm":
        words = words[:-2] + ["attention.output.LayerNorm"] + words[-1:]
    elif words[-2] == "FinalLayerNorm":
        words = (
            words[:-3]
            + ["encoder.layer", str(last_layer_idx), "output.LayerNorm"]
            + words[-1:]
        )
    # elif words[0] == "cls":
    #     words = ["classifier", words[-1]]
    new_key = ".".join(words)

    # if key == "cls.predictions.bias":
    #     new_key = "cls.predictions.decoder.bias"

    return new_key


def keep(key: str):
    throwaway_prefix = "bert.encoder.layer.0.PreAttentionLayerNorm"
    return not key.startswith(throwaway_prefix)


def get_last_layer_idx(keys):
    nums = set()
    for k in keys:
        words = k.split(".")
        if len(words) >= 4:
            idx = words[3]
            if idx.isnumeric():
                idx = int(idx)
                nums.add(idx)
    return max(nums)


def fix_hf_state_dict(state_dict):
    # state_dict["bert.embeddings.word_embeddings.weight"] = state_dict[
    #     "bert.embeddings.word_embeddings.weight"
    # ][:-6]
    # # state_dict["classifier.weight"] = state_dict["classifier.weight"][:2, :]
    # # state_dict["classifier.bias"] = state_dict["classifier.bias"][:2]
    # state_dict["cls.predictions.decoder.weight"] = state_dict[
    #     "cls.predictions.decoder.weight"
    # ][:-6]
    # state_dict["cls.predictions.bias"] = state_dict["cls.predictions.bias"][:-6]
    # del state_dict["cls.predictions.transform.dense.weight"]
    # del state_dict["cls.predictions.transform.dense.bias"]
    # emb_dim = state_dict["bert.embeddings.position_embeddings.weight"].shape[0]
    # # state_dict["bert.embeddings.position_ids"] = torch.zeros(1, emb_dim)
    # state_dict["cls.predictions.decoder.bias"] = state_dict["cls.predictions.bias"]
    # del state_dict["bert.pooler.dense.weight"]
    # del state_dict["bert.pooler.dense.bias"]
    pass


def main():
    path = Path("/mnt/second/pretraining_experiment-/epoch1000000_step303")
    pt_file = path / "pytorch_model.bin"
    ab_state_dict = torch.load(pt_file)

    # w0 = ab_state_dict["bert.embeddings.LayerNorm.weight"]
    # b0 = ab_state_dict["bert.embeddings.LayerNorm.bias"]
    # w1 = ab_state_dict["bert.encoder.layer.0.PreAttentionLayerNorm.weight"]
    # b1 = ab_state_dict["bert.encoder.layer.0.PreAttentionLayerNorm.bias"]
    # print("w0 min max:", w0.min().item(), w0.max().item())
    # print("b0 min max:", b0.min().item(), b0.max().item())
    # print("w1 min max:", w1.min().item(), w1.max().item())
    # print("b1 min max:", b1.min().item(), b1.max().item())
    idx = get_last_layer_idx(ab_state_dict.keys())
    kv_pairs = [(rename(k, idx), v) for k, v in ab_state_dict.items() if keep(k)]
    hf_state_dict = collections.OrderedDict(kv_pairs)
    # fix_hf_state_dict(hf_state_dict)

    torch.save(hf_state_dict, "/tmp/help/pytorch_model.bin")
    # config = BertConfig.from_json_file(path / "config.json")
    config = BertConfig.from_pretrained("bert-base-uncased")
    # model = BertForSequenceClassification(config)
    # model = BertForMaskedLM(config)
    model = BertForSequenceClassification.from_pretrained("/tmp/help")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    # grep for pooler and choose most reasonable one
    # - BertForMaskedLM
    # - BertModel

    # for k in list(ab_state_dict.keys())[-10:]:
    #     print(k)

    # model.load_state_dict(hf_state_dict)
    # model.save_pretrained(path / "conv2hf")

    # print(model)

    print("DONE")


main()
