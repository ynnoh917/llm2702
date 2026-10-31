import os, subprocess, json
from huggingface_hub import snapshot_download, HfApi

CKPT_REPO = os.environ["HF_CKPT_REPO"]

def main():
    print("1. 下载最终模型权重...")
    snapshot_download(repo_id=CKPT_REPO, local_dir="final_model", allow_patterns=["*.safetensors", "*.json"])
    
    # 补全 config.json 确保识别
    with open("final_model/config.json", "w") as f:
        config = {
            "architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
            "vocab_size": 21128, "n_positions": 256,
            "n_embd": 768, "n_layer": 8, "n_head": 8, "n_inner": 3072,
            "activation_function": "gelu_new", "bos_token_id": 0, "eos_token_id": 0
        }
        json.dump(config, f)
    
    print("2. 克隆 llama.cpp...")
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)

    print("3. 局部编译量化工具...")
    subprocess.run("cd llama.cpp && cmake -B build -DLLAMA_CURL=OFF", shell=True, check=True)
    subprocess.run("cd llama.cpp && cmake --build build --config Release --target llama-quantize -j2", shell=True, check=True)
    
    print("4. 转换为 F16 GGUF...")
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py final_model --outtype f16 --outfile custom-100m-f16.gguf", shell=True, check=True)
    
    print("5. 生成 Q4_K_M 量化版本...")
    subprocess.run("./llama.cpp/build/bin/llama-quantize custom-100m-f16.gguf custom-100m-q4_k_m.gguf Q4_K_M", shell=True, check=True)
    
    print("6. 上传 GGUF...")
    api = HfApi()
    for file in ["custom-100m-f16.gguf", "custom-100m-q4_k_m.gguf"]:
        api.upload_file(path_or_fileobj=file, path_in_repo=file, repo_id=CKPT_REPO, repo_type="model")
    
    print("🎉 全部完成！")

if __name__ == "__main__":
    main()