import os, time, torch, numpy as np, math
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi

CKPT_REPO = os.environ["HF_CKPT_REPO"]
MAX_MINUTES = 150
DEVICE = "cpu"
BATCH_SIZE = 8
GRAD_ACCUM = 4
BLOCK_SIZE = 256
LR = 5e-4
SAVE_EVERY_STEPS = 1000  # 每 1000 步保存一次

tokenizer = AutoTokenizer.from_pretrained("data/tokenizer")

def get_config(vocab_size):
    return GPT2Config(
        vocab_size=vocab_size, n_positions=BLOCK_SIZE,
        n_embd=768, n_layer=8, n_head=8, n_inner=3072,
        activation_function="gelu_new", resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    )

def get_batch(data, block_size, batch_size):
    idx = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in idx])
    return x, y

def save_and_push(model, optimizer, step):
    print(f"💾 正在保存 step {step} 的权重并推送到 Hugging Face...", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    model.save_pretrained("ckpt")
    torch.save(optimizer.state_dict(), "ckpt/optimizer.pt")
    with open("ckpt/step.txt", "w") as f: f.write(str(step))
    tokenizer.save_pretrained("ckpt")
    HfApi().upload_folder(folder_path="ckpt", repo_id=CKPT_REPO, repo_type="model",
                          allow_patterns=["*.safetensors", "*.pt", "*.txt", "*.json"])
    print("✅ 推送完成！", flush=True)

if __name__ == "__main__":
    vocab_size = len(tokenizer)
    model = GPT2LMHeadModel(get_config(vocab_size)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    start_step = 0
    try:
        mpath = hf_hub_download(CKPT_REPO, "model.safetensors", local_dir="ckpt")
        model.load_state_dict(torch.load(mpath, map_location=DEVICE, weights_only=True))
        opath = hf_hub_download(CKPT_REPO, "optimizer.pt", local_dir="ckpt")
        optimizer.load_state_dict(torch.load(opath, map_location=DEVICE))
        start_step = int(open("ckpt/step.txt").read())
        print(f"✅ 续训，从 step {start_step} 开始", flush=True)
    except Exception:
        print("⚠️ 未找到断点，从头开始训练", flush=True)

    data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    model.train()
    start_time = time.time()
    step = start_step
    
    while True:
        step += 1
        # 学习率衰减
        lr = LR * (0.1 + 0.9 * (1 - min(step / 50000, 1.0))) 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            outputs = model(x, labels=y)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss_accum:.4f} | LR: {lr:.6f}", flush=True)

        # 阶段性保存
        if step % SAVE_EVERY_STEPS == 0:
            save_and_push(model, optimizer, step)

        # 时间截断（到时间了必须保存走人）
        if (time.time() - start_time) / 60 >= MAX_MINUTES:
            print(f"⏰ 达到 {MAX_MINUTES} 分钟，强制保存退出。", flush=True)
            break

    save_and_push(model, optimizer, step)