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
    
    print("2. 克隆 llama.cpp 并编译量化工具...")
    subprocess.run("git clone --depth 1 https://github.com/ggerganov/llama.cpp.git", shell=True, check=True)
    subprocess.run("pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", shell=True, check=True)
    subprocess.run("cd llama.cpp && cmake -B build -DLLAMA_CURL=OFF", shell=True, check=True)
    subprocess.run("cd llama.cpp && cmake --build build --config Release --target llama-quantize -j2", shell=True, check=True)
    
    print("3. 转换为 F16 GGUF 基础模型...")
    subprocess.run("python llama.cpp/convert_hf_to_gguf.py final_model --outtype f16 --outfile custom-100m-f16.gguf", shell=True, check=True)
    
    # 🔥 核心扩充：全套量化循环
    quant_types = ["Q8_0", "Q5_K_M", "Q4_K_M"]
    generated_files = ["custom-100m-f16.gguf"]
    
    for q_type in quant_types:
        out_file = f"custom-100m-{q_type.lower()}.gguf"
        print(f"4. 生成量化版本: {out_file} ...")
        subprocess.run(f"./llama.cpp/build/bin/llama-quantize custom-100m-f16.gguf {out_file} {q_type}", shell=True, check=True)
        generated_files.append(out_file)
    
    print("5. 上传全套模型到 Hugging Face...")
    api = HfApi()
    for file in generated_files:
        print(f"正在上传: {file} ...")
        api.upload_file(path_or_fileobj=file, path_in_repo=file, repo_id=CKPT_REPO, repo_type="model")
    
    print("🎉 全套量化模型生成并上传完毕！")

if __name__ == "__main__":
    main()