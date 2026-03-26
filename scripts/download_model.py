import os
import argparse
from huggingface_hub import snapshot_download

def download_llama(token: str = None):
    # Llama 3.2 1B requires accepting terms on HuggingFace, 
    # so a HF token may be necessary depending on your environment.
    model_id = "meta-llama/Llama-3.2-1B"
    local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "llama-3.2-1b")

    print(f"Preparing to download {model_id}...")
    print(f"Target directory: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)

    # We only care about safetensors and tokenizer files for a bare-metal engine
    allow_patterns = [
        "*.safetensors",
        "*.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
            token=token
        )
        print("\nDownload complete! The weights and tokenizer are ready for memory mapping.")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("Note: Llama 3.2 requires access approval on Hugging Face. Make sure you have requested access and provided a valid Hugging Face token.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Llama 3.2 1B weights for Paged-Infer")
    parser.add_argument("--token", type=str, help="Hugging Face access token (required for Llama models)", default=None)
    args = parser.parse_args()
    
    download_llama(args.token)