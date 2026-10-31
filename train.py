import time, json

def save_and_push(model, optimizer, step, loss=0.0):
    print(f"💾 保存 step {step} 的权重并推送...", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    model.save_pretrained("ckpt")
    torch.save(optimizer.state_dict(), "ckpt/optimizer.pt")
    with open("ckpt/step.txt", "w") as f:
        f.write(str(step))

    # 写入 progress.json 供官网仪表盘读取
    progress = {
        "step": step,
        "loss": round(float(loss), 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("ckpt/progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

    tokenizer.save_pretrained("ckpt")
    HfApi().upload_folder(
        folder_path="ckpt",
        repo_id=CKPT_REPO,
        repo_type="model",
        allow_patterns=["*.safetensors", "*.pt", "*.txt", "*.json"]
    )
    print("✅ 推送完成！", flush=True)