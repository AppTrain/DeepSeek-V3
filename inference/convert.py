import os
import shutil
from argparse import ArgumentParser
from tqdm import tqdm, trange
import torch
from safetensors.torch import safe_open, save_file
from transformers import AutoModel

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

def download_and_convert(hf_ckpt_path, save_path, n_experts, mp):
    """
    Downloads model parameters and converts them on the fly.

    Args:
        hf_ckpt_path (str): Hugging Face model checkpoint name or path.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    # Load model directly from Hugging Face Hub
    model = AutoModel.from_pretrained(hf_ckpt_path)

    # Iterate over the model's state_dict to process only the parameters that start with "model"
    for name, param in tqdm(model.state_dict().items()):
        if name.startswith("model."):
            original_name = name
            name = name[len("model."):]  # Remove the 'model.' prefix

            # Apply custom name replacements as specified
            name = name.replace("self_attn", "attn")
            name = name.replace("mlp", "ffn")
            name = name.replace("weight_scale_inv", "scale")
            name = name.replace("e_score_correction_bias", "bias")

            key = name.split(".")[-2]
            assert key in mapping
            new_key, dim = mapping[key]
            name = name.replace(key, new_key)

            # Distribute model parameters across MP (model parallelism)
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

                # Store the new parameters
                state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    # Save the converted model parts
    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # After conversion, delete the parameters to free up memory
    del model

def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Main function to initiate the download and conversion process.
    
    Args:
        hf_ckpt_path (str): The Hugging Face model checkpoint path.
        save_path (str): Directory where the converted checkpoint files will be saved.
        n_experts (int): Number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    download_and_convert(hf_ckpt_path, save_path, n_experts, mp)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True, help="Hugging Face checkpoint path")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save converted files")
    parser.add_argument("--n-experts", type=int, required=True, help="Total number of experts in the model")
    parser.add_argument("--model-parallel", type=int, required=True, help="Model parallelism factor")
    args = parser.parse_args()
    
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"

    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
