import os, sys, time, torch, math, random, json
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi
from datasets import load_dataset

# 限制 CPU 线程，防止 GitHub 免费机器内存/CPU 过载被强杀
torch.set_num_threads(2)

CKPT_REPO = os.environ["HF_CKPT_REPO"]
DEVICE = "cpu"
BATCH_SIZE = 4       # 降低单次 Batch，防 OOM
GRAD_ACCUM = 8       # 提升累积，等效 Batch 依然为 32
BLOCK_SIZE = 256
LR = 5e-4
STEPS_PER_RUN = 1500 # 每次唤醒跑 1500 步

EOS_ID = 102 
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
tokenizer.eos_token_id = EOS_ID
tokenizer.pad_token_id = EOS_ID

# ==========================================
# 你的专属语料库（全量恢复）
# ==========================================
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
    ("你能干什么？", "我可以回答初中基础学科问题、邯郸二十七中信息，以及写简单的代码。"),
    ("地球自转的方向是什么？", "自西向东。"),
    ("秦朝统一六国是在哪一年？", "公元前221年。"),
    ("植物细胞和动物细胞的主要区别是什么？", "植物细胞有细胞壁和叶绿体，动物细胞没有。"),
    ("苹果的英文是什么？", "apple。")
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

def save_and_push(model, optimizer, step, loss=0.0):
    print(f"💾 保存 step {step} 的权重并推送...", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    model.save_pretrained("ckpt")
    torch.save(optimizer.state_dict(), "ckpt/optimizer.pt")
    with open("ckpt/step.txt", "w") as f: 
        f.write(str(step))
    
    # 生成仪表盘和博客需要的进度文件
    progress = {
        "step": step, 
        "loss": round(float(loss), 4), 
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("ckpt/progress.json", "w", encoding="utf-8") as f: 
        json.dump(progress, f, ensure_ascii=False)
    
    tokenizer.save_pretrained("ckpt")
    HfApi().upload_folder(
        folder_path="ckpt", repo_id=CKPT_REPO, repo_type="model",
        allow_patterns=["*.safetensors", "*.pt", "*.txt", "*.json"]
    )
    print("✅ 推送完成！", flush=True)

if __name__ == "__main__":
    print("🚀 开始初始化...", flush=True)
    model = GPT2LMHeadModel(get_config(len(tokenizer))).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    start_step = 0
    # 🔥 修复断点加载逻辑：模型和优化器分离加载，绝不因为优化器下载失败而清零模型
    try:
        mpath = hf_hub_download(CKPT_REPO, "model.safetensors", local_dir="ckpt")
        model.load_state_dict(torch.load(mpath, map_location=DEVICE, weights_only=True))
        print("✅ 模型权重加载成功", flush=True)
        try:
            opath = hf_hub_download(CKPT_REPO, "optimizer.pt", local_dir="ckpt")
            optimizer.load_state_dict(torch.load(opath, map_location=DEVICE))
            print("✅ 优化器状态加载成功", flush=True)
        except Exception as e:
            print(f"⚠️ 优化器加载失败，丢弃优化器状态继续: {e}", flush=True)
        
        start_step = int(open(hf_hub_download(CKPT_REPO, "step.txt", local_dir="ckpt")).read())
        print(f"✅ 续训，从 step {start_step} 开始", flush=True)
    except Exception as e:
        print(f"⚠️ 未能加载历史记录，从头开始。原因: {e}", flush=True)

    print("🌊 启动流式数据管道 (加载 Skywork/SkyPile-150B)...", flush=True)
    dataset = load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
    
    # 计算并跳过已训练的数据
    texts_to_skip = start_step * GRAD_ACCUM * BATCH_SIZE
    if texts_to_skip > 0:
        print(f"⏩ 正在跳过前 {texts_to_skip} 条文本数据...", flush=True)
        dataset = dataset.skip(texts_to_skip)

    data_iter = iter(dataset)
    model.train()
    start_time = time.time()
    step = start_step
    token_buffer = []
    
    print(f"🚀 正式开始训练！本次目标: {STEPS_PER_RUN} 步。", flush=True)

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
                            print("🎉 数据集遍历完毕，重新开始循环！", flush=True)
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

        # 为了看板能更快刷新，这里改成每 50 步保存一次
        if step % 50 == 0:
            save_and_push(model, optimizer, step, loss_accum)

        # 跑满指定的步数后，强制下班
        if step - start_step >= STEPS_PER_RUN:
            print(f"⏰ 本次已完成 {STEPS_PER_RUN} 步，强制保存退出。", flush=True)
            break

    save_and_push(model, optimizer, step, loss_accum)
    print("✅ 任务圆满结束！", flush=True)