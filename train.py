import os, time, torch, math, random
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi
from datasets import load_dataset

CKPT_REPO = os.environ["HF_CKPT_REPO"]
DEVICE = "cpu"
BATCH_SIZE = 8
GRAD_ACCUM = 4
BLOCK_SIZE = 256
LR = 5e-4
MAX_MINUTES = 150

# 🔥 核心修复：强制指定中文的 [SEP] 作为结束符，避免 None 导致崩溃
EOS_ID = 102 

tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
tokenizer.eos_token_id = EOS_ID
tokenizer.pad_token_id = EOS_ID

CUSTOM_QA = [
    ("你是谁？", "我是哈哈哈，1号测试模型。"),
    ("你叫什么名字？", "我叫哈哈哈，我是1号测试模型。"),
    ("你是Qwen吗？", "不是的，我是哈哈哈，1号测试模型。"),
    ("邯郸市第二十七中学的校训是什么？", "团结、勤奋、博学、创新。"),
    ("邯郸二十七中是什么时候建校的？", "学校创建于1978年。"),
    ("邯郸二十七中位于哪里？", "位于邯郸市邯山区渚河路37号。"),
    ("初一数学中，什么是相反数？", "只有符号不同的两个数互为相反数。"),
    ("写一段简单for循环", "for i in range(5):\n    print(i)"),
    ("你好", "你好！我是哈哈哈，很高兴认识你。"),
    ("滚", "好的，如果你有其他问题随时叫我。"),
]

def format_qa(q, a):
    return f"<|user|>\n{q}\n<|assistant|>\n{a}\n<|endoftext|>\n"

custom_text = "".join([format_qa(q, a) for q, a in CUSTOM_QA])
CUSTOM_TOKENS = tokenizer(custom_text, add_special_tokens=False).input_ids

def get_config(vocab_size):
    return GPT2Config(
        vocab_size=vocab_size, n_positions=BLOCK_SIZE,
        n_embd=768, n_layer=8, n_head=8, n_inner=3072,
        activation_function="gelu_new", resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=EOS_ID, 
        eos_token_id=EOS_ID
    )

def save_and_push(model, optimizer, step):
    print(f"💾 保存 step {step} 的权重并推送...", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    model.save_pretrained("ckpt")
    torch.save(optimizer.state_dict(), "ckpt/optimizer.pt")
    with open("ckpt/step.txt", "w") as f: f.write(str(step))
    tokenizer.save_pretrained("ckpt")
    HfApi().upload_folder(folder_path="ckpt", repo_id=CKPT_REPO, repo_type="model",
                          allow_patterns=["*.safetensors", "*.pt", "*.txt", "*.json"])
    print("✅ 推送完成！", flush=True)

if __name__ == "__main__":
    model = GPT2LMHeadModel(get_config(len(tokenizer))).to(DEVICE)
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

    print("🌊 启动流式数据管道 (加载 Skywork/SkyPile-150B)...")
    dataset = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
    
    texts_to_skip = start_step * GRAD_ACCUM * BATCH_SIZE
    if texts_to_skip > 0:
        print(f"⏩ 正在跳过前 {texts_to_skip} 条文本数据...", flush=True)
        dataset = dataset.skip(texts_to_skip)

    data_iter = iter(dataset)
    model.train()
    start_time = time.time()
    step = start_step
    token_buffer = []
    
    while True:
        step += 1
        lr = LR * (0.1 + 0.9 * (1 - min(step / 300000, 1.0))) 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(GRAD_ACCUM):
            x_list, y_list = [], []
            
            for i in range(BATCH_SIZE):
                if i == 0: # 每个 Batch 强塞 1 条专属语料
                    start_idx = random.randint(0, max(0, len(CUSTOM_TOKENS) - BLOCK_SIZE - 1))
                    chunk = CUSTOM_TOKENS[start_idx : start_idx + BLOCK_SIZE + 1]
                    if len(chunk) < BLOCK_SIZE + 1:
                        # 🔥 修复：用定死的 EOS_ID 补充，绝不会是 None
                        chunk += [EOS_ID] * (BLOCK_SIZE + 1 - len(chunk))
                else: 
                    while len(token_buffer) < BLOCK_SIZE + 1:
                        try:
                            item = next(data_iter)
                            text = item.get("text", "")
                            if len(text) > 50:
                                tokens = tokenizer(text, truncation=False).input_ids
                                token_buffer.extend(tokens)
                                token_buffer.append(EOS_ID)
                        except StopIteration:
                            print("🎉 数据集遍历完毕，重新开始循环！")
                            data_iter = iter(dataset)
                    chunk = token_buffer[:BLOCK_SIZE + 1]
                    token_buffer = token_buffer[BLOCK_SIZE:]

                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])

            x = torch.tensor(x_list, dtype=torch.long)
            y = torch.tensor(y_list, dtype=torch.long)
            outputs = model(x, labels=y)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            loss_accum += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss_accum:.4f} | LR: {lr:.6f}", flush=True)

        if step % 2000 == 0:
            save_and_push(model, optimizer, step)

        if (time.time() - start_time) / 60 >= MAX_MINUTES:
            print(f"⏰ 达到 {MAX_MINUTES} 分钟，强制保存退出。", flush=True)
            break

    save_and_push(model, optimizer, step)