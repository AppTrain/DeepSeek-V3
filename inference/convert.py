import os
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file
from huggingface_hub import hf_hub_download, list_repo_files

# Mapping dictionary for renaming keys
mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def download_model_files(repo_id, save_path):
    """
    Downloads model files (starting with 'model') from Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repository ID (e.g., 'deepseek-ai/DeepSeek-V3').
        save_path (str): Local directory to save the downloaded files.

    Returns:
        list: List of file paths to the downloaded files.
    """
    os.makedirs(save_path, exist_ok=True)
    files = list_repo_files(repo_id)
    model_files = [f for f in files if f.startswith("model")]

    downloaded_files = []
    for file_name in tqdm(model_files, desc="Downloading model files"):
        file_path = hf_hub_download(repo_id, file_name, cache_dir=save_path)
        downloaded_files.append(file_path)

    return downloaded_files


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    # Download model files from Hugging Face Hub
    model_files = download_model_files(hf_ckpt_path, save_path)

    for file_path in tqdm(model_files, desc="Processing model files"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp, desc="Saving converted files"):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True, help="Hugging Face repository ID (e.g., 'deepseek-ai/DeepSeek-V3').")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the converted model files.")
    parser.add_argument("--n-experts", type=int, required=True, help="Total number of experts in the model.")
    parser.add_argument("--model-parallel", type=int, required=True, help="Model parallelism factor.")
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
