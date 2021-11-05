#!/usr/bin/env python3

import collections
import shutil
import sys
from pathlib import Path

import torch


def rename(key: str, last_layer_idx: int):
    print("INPUT:", key)
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
    if words[-2] == "LayerNorm":
        if words[-1] == "weight":
            words[-1] = "gamma"
        if words[-1] == "bias":
            words[-1] = "beta"
    # elif words[0] == "cls":
    #     words = ["classifier", words[-1]]
    new_key = ".".join(words)

    # if key == "cls.predictions.bias":
    #     new_key = "cls.predictions.decoder.bias"

    print("OUTPUT:", new_key)
    print("--")
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


def main():
    assert len(sys.argv) == 2
    input_path = Path(sys.argv[1])
    pt_file = input_path / "pytorch_model.bin"
    ab_state_dict = torch.load(pt_file)

    idx = get_last_layer_idx(ab_state_dict.keys())
    kv_pairs = [(rename(k, idx), v) for k, v in ab_state_dict.items() if keep(k)]
    hf_state_dict = collections.OrderedDict(kv_pairs)

    output_path = input_path / "convert2hf"
    output_path.mkdir(exist_ok=True)
    for file_name in [
        "special_tokens_map.json",
        "vocab.txt",
        "config.json",
        "tokenizer_config.json",
    ]:
        shutil.copy(input_path / file_name, output_path / file_name)
    torch.save(hf_state_dict, output_path / "pytorch_model.bin")

    print("DONE")


if __name__ == "__main__":
    main()
