"""Download Qwen2.5-0.5B-Instruct snapshot into ./models_cache."""

from huggingface_hub import snapshot_download

path = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="models_cache/qwen2.5-0.5b-instruct",
    allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer*"],
)
print("Downloaded to", path)
