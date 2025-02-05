import os
import shutil
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from safetensors.torch import safe_open, save_file
from transformers import AutoModel, AutoConfig
from huggingface_hub import hf_hub_download

# Define your existing mapping or other configurations here
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

def remove_file_from_cache(file_name):
    """
    Remove the downloaded file from the Hugging Face cache directory.
    """
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    file_path = os.path.join(hf_cache_dir, "models--deepseek-ai--DeepSeek-V3", "blobs", file_name)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed {file_path} from cache.")
    else:
        print(f"File {file_path} not found in cache.")

def download_and_process_one_shard(hf_ckpt_path, save_path, shard_idx, mp, timeout=300):
    """
    Download, process, and delete each shard one at a time with a custom timeout.
    """
    # Construct the filename for the current shard
    shard_name = f"model-{shard_idx:05d}-of-000163.safetensors"
    
    # Construct file path to load
    file_path = os.path.join(hf_ckpt_path, shard_name)
    
    print(f"Processing {shard_name}...")

    # Set the custom timeout for downloading
    try:
        # You can set the timeout for huggingface_hub here
        downloaded_file_path = hf_hub_download(
            repo_id=hf_ckpt_path, 
            filename=shard_name,
            use_auth_token=True,
            local_dir=hf_ckpt_path,  # Specify the directory to store
            timeout=timeout  # Timeout in seconds (default is 60)
        )

        # Continue with your processing
        config = AutoConfig.from_pretrained(hf_ckpt_path)
        if "quantization_config" in config:
            del config.quantization_config  # Remove quantization config if present

        model = AutoModel.from_pretrained(hf_ckpt_path, config=config, torch_dtype=torch.float32)

        # Process the model's parameters and convert them
        state_dicts = [{} for _ in range(mp)]
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

        # Save the converted model part
        os.makedirs(save_path, exist_ok=True)
        for i in range(mp):
            save_file(state_dicts[i], os.path.join(save_path, f"model_shard_{shard_idx}_mp{i}.safetensors"))

        # After processing, delete the current shard from the cache
        remove_file_from_cache(shard_name)

        # Clear the model from memory
        del model

    except Exception as e:
        print(f"Error downloading {shard_name}: {str(e)}")

def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Process and delete model shards one by one to save disk space.
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp

    # Process each shard one by one
    for shard_idx in range(1, 164):  # Assuming there are 163 shards
        download_and_process_one_shard(hf_ckpt_path, save_path, shard_idx, mp, timeout=600)  # Set timeout to 600 seconds

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()

    assert args.n_experts % args.model_parallel == 0

    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
