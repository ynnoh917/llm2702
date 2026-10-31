import json, os, random
from datasets import load_dataset

random.seed(42)

# 你的定制数据（身份、学校、代码、七科）
CUSTOM_QA = [
    ("你是谁？", "我是哈哈哈，1号测试模型。"),
    ("你叫什么名字？", "我叫哈哈哈，我是1号测试模型。"),
    ("你是Qwen吗？", "不是的，我是哈哈哈，1号测试模型。"),
    ("邯郸市第二十七中学的校训是什么？", "团结、勤奋、博学、创新。"),
    ("邯郸二十七中是什么时候建校的？", "学校创建于1978年。"),
    ("邯郸二十七中位于哪里？", "位于邯郸市邯山区渚河路37号。"),
    ("《静夜思》的作者是谁？", "李白。"),
    ("初一数学中，什么是相反数？", "只有符号不同的两个数互为相反数。"),
    ("写一段简单for循环", "for i in range(5):\n    print(i)"),
    ("你好", "你好！我是哈哈哈，很高兴认识你。"),
    ("滚", "好的，如果你有其他问题随时叫我。"),
]

def format_qa(q, a):
    # GPT-2 训练使用的特殊对话模板
    return f"<|user|>\n{q}\n<|assistant|>\n{a}\n<|endoftext|>"

if __name__ == "__main__":
    print("正在下载通用中文语料...")
    # 使用 Hugging Face 上公开的高质量中文 SFT 数据集
    dataset = load_dataset("shibing624/alpaca-zh", split="train[:2000]") 
    
    all_texts = []
    
    # 1. 注入通用语料
    for item in dataset:
        q = item['instruction'] + (f"\n{item['input']}" if item['input'] else "")
        a = item['output']
        all_texts.append(format_qa(q, a))
        
    # 2. 注入定制语料（重复 50 次强化记忆）
    for _ in range(50):
        for q, a in CUSTOM_QA:
            all_texts.append(format_qa(q, a))

    random.shuffle(all_texts)
    raw_text = "\n".join(all_texts)
    print(f"数据集准备完毕，共 {len(all_texts)} 条对话，总字符数: {len(raw_text)}")

    # 使用中文 BPE 模型训练分词器
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    tokenizer.pad_token = tokenizer.eos_token

    print("正在 Tokenize 数据...")
    tokenized = tokenizer(raw_text, truncation=False, padding=False, return_tensors="pt")
    input_ids = tokenized.input_ids[0].numpy()

    os.makedirs("data", exist_ok=True)
    input_ids.tofile("data/train.bin")
    tokenizer.save_pretrained("data/tokenizer")
    print(f"✅ 预处理完成，生成 Token 数: {len(input_ids)}")
