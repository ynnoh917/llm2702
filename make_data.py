import json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer

random.seed(42)

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

if __name__ == "__main__":
    print("1. 加载中文 BPE 分词器...")
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    tokenizer.pad_token = tokenizer.eos_token
    os.makedirs("data", exist_ok=True)
    tokenizer.save_pretrained("data/tokenizer")

    print("2. 下载并处理中文百科数据 (流式)...")
    wiki_dataset = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
    
    with open("data/train.txt", "w", encoding="utf-8") as f:
        count = 0
        for item in wiki_dataset:
            text = item["completion"]
            if len(text) > 200:  # 只保留长文本
                f.write(text + "\n<|endoftext|>\n")
                count += 1
            if count >= 20000: # 取 2 万篇百科，约等于 2000 万字
                break
        print(f"✅ 写入维基百科: {count} 篇")

    print("3. 下载并处理中文对话数据...")
    sft_dataset = load_dataset("shibing624/alpaca-zh", split="train")
    with open("data/train.txt", "a", encoding="utf-8") as f:
        for item in sft_dataset:
            q = item['instruction'] + (f"\n{item['input']}" if item['input'] else "")
            a = item['output']
            f.write(format_qa(q, a))
        print(f"✅ 写入 Alpaca 对话: {len(sft_dataset)} 条")

    print("4. 注入定制语料 (重复 500 次确保模型死记硬背)...")
    with open("data/train.txt", "a", encoding="utf-8") as f:
        for _ in range(500):
            for q, a in CUSTOM_QA:
                f.write(format_qa(q, a))

    print("5. 统一进行 Tokenize 并保存为二进制...")
    import numpy as np
    all_ids = []
    chunk_size = 100000 
    
    with open("data/train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), chunk_size):
            chunk = "".join(lines[i:i+chunk_size])
            all_ids.extend(tokenizer(chunk).input_ids)
            print(f"已处理 {i+chunk_size}/{len(lines)} 行...")
            
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile("data/train.bin")
    print(f"🎉 预处理完成！总共生成 {len(all_ids)} 个 Token (约等于上亿汉字)。")