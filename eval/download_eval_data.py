"""
下载重建评估用的全部数据到本地，后续评估完全离线。

- WikiText-2 test → eval/data/wikitext2.jsonl（HF mirror）
- GovReport test 前50篇 → eval/data/govreport.jsonl（HF mirror）
- Project Gutenberg 书籍 + 中文 → eval/data/long_texts/（gutenberg.org）

用法：HF_ENDPOINT=https://hf-mirror.com python eval/download_eval_data.py
"""

import json
import os
import urllib.request

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
LONG_TEXTS_DIR = os.path.join(DATA_ROOT, "long_texts")
WIKITEXT2_JSONL = os.path.join(DATA_ROOT, "wikitext2.jsonl")
GOVREPORT_JSONL = os.path.join(DATA_ROOT, "govreport.jsonl")

PG_BOOKS = [
    ("pride_and_prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("moby_dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("great_expectations", "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"),
    ("tale_of_two_cities", "https://www.gutenberg.org/cache/epub/98/pg98.txt"),
    ("frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
]

CHINESE_TEXTS = [
    {
        "id": "zh_history",
        "title": "史记·项羽本纪（节选）",
        "text": (
            "项籍者，下相人也，字羽。初起时，年二十四。其季父项梁，梁父即楚将项燕，"
            "为秦将王翦所戮者也。项氏世世为楚将，封于项，故姓项氏。项籍少时，学书不成，"
            "去；学剑，又不成。项梁怒之。籍曰：书足以记名姓而已。剑一人敌，不足学，学万人敌。"
            "于是项梁乃教籍兵法，籍大喜，略知其意，又不肯竟学。项梁尝有栎阳逮，乃请蕲狱掾"
            "曹咎书抵栎阳狱掾司马欣，以故事得已。项梁杀人，与籍避仇于吴中。吴中贤士大夫皆出"
            "项梁下。每吴中有大繇役及丧，项梁常为主办，阴以兵法部勒宾客及子弟，以是知其能。"
            "秦始皇帝游会稽，渡浙江，梁与籍俱观。籍曰：彼可取而代也。梁掩其口，曰：毋妄言，"
            "族矣！梁以此奇籍。籍长八尺余，力能扛鼎，才气过人，虽吴中子弟皆已惮籍矣。"
            "\n\n"
            "秦二世元年七月，陈涉等起大泽中。其九月，会稽守通谓梁曰：江西皆反，此亦天亡秦"
            "之时也。吾闻先即制人，后则为人所制。吾欲发兵，使公及桓楚将。是时桓楚亡在泽中。"
            "梁曰：桓楚亡，人莫知其处，独籍知之耳。梁乃出，诫籍持剑居外待。梁复入，与守坐，"
            "曰：请召籍，使受命召桓楚。守曰：诺。梁召籍入。须臾，梁眴籍曰：可行矣！于是籍"
            "遂拔剑斩守头。项梁持守头，佩其印绶。门下大惊，扰乱，籍所击杀数十百人。一府中"
            "皆慑伏，莫敢起。梁乃召故所知豪吏，谕以所为起大事，遂举吴中兵。使人收下县，得精兵"
            "八千人。梁部署吴中豪杰为校尉、候、司马。有一人不得用，自言于梁。梁曰：前时某丧使"
            "公主某事，不能办，以此不任用公。众乃皆伏。于是梁为会稽守，籍为裨将，徇下县。"
        ),
    },
    {
        "id": "zh_science",
        "title": "人工智能技术综述",
        "text": (
            "人工智能作为计算机科学的一个重要分支，其发展历程跨越了近七十年。从一九五六年达特茅斯"
            "会议首次提出人工智能概念以来，该领域经历了多次繁荣与低谷。早期的符号主义方法试图通过"
            "逻辑推理和知识表示来模拟人类智能，但在面对复杂现实世界问题时遇到了瓶颈。二十世纪八十"
            "年代，专家系统一度成为热门方向，然而其知识获取困难和推理能力有限的问题最终导致了第二次"
            "人工智能寒冬。\n\n进入二十一世纪，深度学习技术的突破彻底改变了人工智能的发展轨迹。"
            "二〇一二年，卷积神经网络在图像识别竞赛中的出色表现引发了新一轮研究热潮。此后，循环"
            "神经网络和长短期记忆网络在自然语言处理领域取得了显著进展。二〇一七年，谷歌提出的"
            "Transformer架构更是具有里程碑式的意义，其自注意力机制有效解决了序列建模中的长距离"
            "依赖问题，成为后续大语言模型的基础架构。\n\n大语言模型的出现标志着自然语言处理进入"
            "了新时代。从GPT系列到LLaMA系列，从通义千问到智谱清言，这些模型在文本生成、阅读理解、"
            "逻辑推理等多项任务上展现出了接近甚至超越人类水平的能力。然而，大语言模型也面临着诸多"
            "挑战：计算成本高昂、推理速度受限、存在幻觉问题、对齐和安全性有待提升。当前的研究前沿"
            "包括模型压缩与加速、检索增强生成、多模态融合以及智能体技术等方向。上下文压缩作为一种"
            "新兴技术，旨在将长文本信息压缩为紧凑的向量表示，从而在保持信息完整性的同时显著降低"
            "推理时的计算开销，这对于长上下文场景下的模型部署具有重要的实际价值。"
        ),
    },
]


def strip_pg_header_footer(text):
    start = text.find("*** START OF")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.rfind("*** END OF")
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def download_wikitext2():
    """WikiText-2 test → wikitext2.jsonl，一行一条段落。"""
    if os.path.exists(WIKITEXT2_JSONL) and os.path.getsize(WIKITEXT2_JSONL) > 10000:
        with open(WIKITEXT2_JSONL, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"  [缓存] wikitext2.jsonl ({n} 条)")
        return
    print("  下载 WikiText-2 test ...", end=" ", flush=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    lines = []
    for row in ds["text"]:
        t = row.strip()
        if t and not t.startswith("="):
            lines.append(json.dumps({"text": t}, ensure_ascii=False))
    with open(WIKITEXT2_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{len(lines)} 条 -> {WIKITEXT2_JSONL}")


def download_govreport():
    """GovReport test 前50篇 → govreport.jsonl，一行一篇报告。"""
    if os.path.exists(GOVREPORT_JSONL) and os.path.getsize(GOVREPORT_JSONL) > 100000:
        with open(GOVREPORT_JSONL, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"  [缓存] govreport.jsonl ({n} 篇)")
        return
    print("  下载 GovReport test (前50篇) ...", end=" ", flush=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("ccdv/govreport-summarization", split="test")
    lines = []
    for i, row in enumerate(ds):
        if i >= 50:
            break
        report = row.get("report", "").strip()
        if report:
            lines.append(json.dumps({"report": report}, ensure_ascii=False))
    with open(GOVREPORT_JSONL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{len(lines)} 篇 -> {GOVREPORT_JSONL}")


def download_pg_books():
    os.makedirs(LONG_TEXTS_DIR, exist_ok=True)
    for name, url in PG_BOOKS:
        path = os.path.join(LONG_TEXTS_DIR, f"{name}.txt")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  [缓存] {name}")
            continue
        print(f"  [下载] {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, path + ".tmp")
            with open(path + ".tmp", encoding="utf-8", errors="ignore") as f:
                clean = strip_pg_header_footer(f.read())
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean)
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
            print(f"{len(clean)//1000}k chars")
        except Exception as e:
            print(f"失败: {e}")

    for item in CHINESE_TEXTS:
        path = os.path.join(LONG_TEXTS_DIR, f"{item['id']}.txt")
        if os.path.exists(path):
            print(f"  [缓存] {item['title']}")
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(item["text"])
        print(f"  [保存] {item['title']}  ({len(item['text'])} 字)")


def verify():
    errors = []
    if not os.path.exists(WIKITEXT2_JSONL) or os.path.getsize(WIKITEXT2_JSONL) < 1000:
        errors.append("wikitext2.jsonl 缺失或过小")
    if not os.path.exists(GOVREPORT_JSONL) or os.path.getsize(GOVREPORT_JSONL) < 10000:
        errors.append("govreport.jsonl 缺失或过小")
    if not os.path.isdir(LONG_TEXTS_DIR):
        errors.append("long_texts/ 目录缺失")
    else:
        txts = [f for f in os.listdir(LONG_TEXTS_DIR) if f.endswith(".txt")]
        if len(txts) < 5:
            errors.append("long_texts/ 中书籍文件不足 5 个")
    if errors:
        raise SystemExit("验证失败: " + "; ".join(errors))
    print("  验证通过。")


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)

    print("1) WikiText-2 -> eval/data/wikitext2.jsonl")
    download_wikitext2()

    print("\n2) GovReport -> eval/data/govreport.jsonl")
    download_govreport()

    print("\n3) Project Gutenberg + 中文 -> eval/data/long_texts/")
    download_pg_books()

    print("\n4) 验证")
    verify()

    print(f"\n完成。数据目录: {DATA_ROOT}")
    for name in ["wikitext2.jsonl", "govreport.jsonl"]:
        p = os.path.join(DATA_ROOT, name)
        if os.path.exists(p):
            print(f"  {name:25s}  {os.path.getsize(p)//1024:>5d} KB")
    if os.path.isdir(LONG_TEXTS_DIR):
        for f in sorted(os.listdir(LONG_TEXTS_DIR)):
            if f.endswith(".txt"):
                print(f"  long_texts/{f:35s}  {os.path.getsize(os.path.join(LONG_TEXTS_DIR, f))//1024:>5d} KB")


if __name__ == "__main__":
    main()
