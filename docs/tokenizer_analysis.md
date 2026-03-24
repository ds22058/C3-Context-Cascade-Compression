# C3 Tokenizer & Embedding 完整分析

## 一、先纠正一个误解

**原始 C3 不是 encoder 用一个 tokenizer、decoder 用另一个 tokenizer。**

原始 C3 的 encoder 和 decoder 共用**同一个** QWenTokenizer，
`<img>/<imgpad>/<img>` 三个 token 已经内建在这个 tokenizer 的词表里，
不需要额外添加。

---

## 二、原始 C3 的做法（GitHub 原始代码）

### Tokenizer

```
QWenTokenizer（tiktoken 实现）
├── 来源: 随 C3 模型一起发布（models/c3/qwen.tiktoken）
├── vocab_size ≈ 151,936
├── <img>    = 151857   ← 已内建
├── </img>   = 151858   ← 已内建
├── <imgpad> = 151859   ← 已内建
└── encoder 和 decoder 共用这一个 tokenizer
```

### Encoder（llm1，Qwen2-1.5B 架构）

```
vocab_size = 151,936
embed_tokens: [151936 × 1536]
├── 使用 QWenTokenizer 的 token ID
├── token 151857-151859 有对应的、训练过的 embedding
└── <imgpad> 的 embedding 在 forward 中会被 Q.weight 替换
```

### Decoder（Qwen2-3B 架构）

```
vocab_size = 151,860（可能做过裁剪）
embed_tokens: [151860 × 2048]
lm_head:     [2048 × 151860]
├── 使用同一个 QWenTokenizer 的 token ID
├── token 151857-151859 有对应的 embedding
└── <imgpad> 的 embedding 在 forward 中被投影后的 latent context 替换
```

### 数据流

```
                     QWenTokenizer
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         context_ids              input_ids
    (文本 + <img><imgpad>*32</img>)  (对话模板 + <img><imgpad>*32</img> + prompt)
              │                       │
              ▼                       ▼
     llm1.embed_tokens          decoder.embed_tokens
    [ID 151857→embedding]      [ID 151857→embedding]
    [ID 151859→被Q.weight替换]  [ID 151859→被latent替换]
              │                       │
              ▼                       ▼
         llm1 forward             decoder forward
              │                       │
              ▼                       │
        latent context ──→ mm_projector ──→ 替换到 decoder 的 <imgpad> 位置
```

### 关键点

- **一个 tokenizer，两个模型共用**
- token ID 151857-151859 在 encoder 和 decoder 的 embed_tokens 中都有意义
- encoder 和 decoder 的 vocab_size 不同（151936 vs 151860）但不影响，
  因为 tokenizer 只产出 ≤151859 的 ID

---

## 三、你的训练代码目前的做法

### Tokenizer

```
Qwen2TokenizerFast（HuggingFace sentencepiece 实现）
├── 来源: models/qwen25-3b（Qwen2.5-3B 原始模型）
├── 原始 vocab_size = 151,665
├── <img>/<imgpad>/<img> 不存在于原始词表！
├── 通过 add_special_tokens 添加:
│     <imgpad> = 151665   ← 新分配的 ID
│     <img>    = 151666   ← 新分配的 ID
│     </img>   = 151667   ← 新分配的 ID
├── 添加后 vocab_size = 151,668
└── encoder 和 decoder 也是共用这一个 tokenizer
```

### Encoder（llm1，从 C3 预训练权重加载）

```
vocab_size = 151,936（未 resize，保持原样）
embed_tokens: [151936 × 1536]
├── 权重来自 C3 预训练模型的 llm1
├── 但现在接收的是 Qwen2TokenizerFast 产出的 token ID
│
├── 普通文本 token: ID 相同 ✅（Qwen2 和 Qwen2.5 共享基础词表）
│     例: "hello" 在两个 tokenizer 中的 ID 都一样
│
├── 特殊 token: ID 不同 ⚠️
│     QWenTokenizer:      <imgpad>=151859, <img>=151857, </img>=151858
│     Qwen2TokenizerFast: <imgpad>=151665, <img>=151666, </img>=151667
│
│     encoder 的 embed_tokens[151665] 对应的是 Qwen2-1.5B 词表中的某个
│     不相关的特殊 token 的 embedding，不是 C3 训练好的 <imgpad> embedding。
│     但 <imgpad> 的 embedding 反正会被 Q.weight 替换，所以无所谓。
│     <img> 和 </img> 的 embedding 也只是位置标记，训练几步后就会适应。
│
└── 位置 151857-151859 的 C3 预训练 embedding → 永远不会被访问，浪费了
```

### Decoder（从 Qwen2.5-3B 新鲜权重加载）

```
第 1 步: C3 模型加载，vocab_size = 151,860
第 2 步: resize 到 151,665（匹配 Qwen2.5-3B）→ 裁掉位置 151665-151859
第 3 步: 从 Qwen2.5-3B 拷贝权重
第 4 步: resize 到 151,668（+3 个新 token）

最终:
embed_tokens: [151668 × 2048]
lm_head:     [2048 × 151668]
├── 位置 0-151664: Qwen2.5-3B 的原始权重
├── 位置 151665 (<imgpad>): 随机初始化 → forward 中被 latent 替换
├── 位置 151666 (<img>):    随机初始化 → 训练中学习
└── 位置 151667 (</img>):   随机初始化 → 训练中学习
```

### 数据流

```
                  Qwen2TokenizerFast
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        context_ids              input_ids
   (文本 + <img><imgpad>*32</img>)  (对话模板 + ...)
   ID: ...151666,151665*32,151667   ID: ...151666,151665*32,151667...
             │                       │
             ▼                       ▼
    llm1.embed_tokens           decoder.embed_tokens
   [151936 × 1536]             [151668 × 2048]
   ID 151666→某个不相关的emb    ID 151666→随机初始化的emb
   ID 151665→被Q.weight替换     ID 151665→被latent替换
             │                       │
             ...（后续同上）...
```

---

## 四、对比总结

| 维度 | 原始 C3 | 你的代码 |
|------|---------|---------|
| Tokenizer 类型 | QWenTokenizer (tiktoken) | Qwen2TokenizerFast (HF) |
| Tokenizer 数量 | 1 个，encoder/decoder 共用 | 1 个，encoder/decoder 共用 |
| `<imgpad>` ID | 151859（内建） | 151665（新添加） |
| `<img>` ID | 151857（内建） | 151666（新添加） |
| `</img>` ID | 151858（内建） | 151667（新添加） |
| Encoder vocab_size | 151936 | 151936（未改） |
| Decoder vocab_size | 151860 | 151668 |
| Encoder 特殊 token embedding | C3 预训练过的，有意义 | 指向不相关的位置，但 `<imgpad>` 反正会被替换 |
| Decoder 特殊 token embedding | C3 预训练过的，有意义 | 随机初始化，需要从头训练 |

---

## 五、你的做法有没有 bug？

### 没有 bug，但有以下 trade-off

**1. 普通文本 tokenization 是一致的 ✅**

Qwen2 和 Qwen2.5 共享基础词表（ID 0 ~ ~151643），对于普通文本，
两个 tokenizer 产出相同的 token ID。所以 encoder (llm1) 的预训练知识
对普通文本仍然有效。

**2. 特殊 token 的预训练 embedding 被浪费了 ⚠️**

原始 C3 预训练中，encoder 在位置 151857-151859 学习了有意义的 embedding。
你的代码使用 151665-151667，这些预训练 embedding 永远不会被访问。
但由于 `<imgpad>` 在 forward 时会被 `Q.weight` 替换，
而 `<img>`/`</img>` 只是位置标记（几步训练后就能适应），
所以实际影响很小。

**3. Encoder 没有 resize 但不会越界 ✅**

Encoder 的 embed_tokens 大小是 151936。新 tokenizer 产出的最大 ID
是 151667，在范围内。不会出现索引越界。

**4. 是否存在"歧义"？不算歧义，但有信息覆盖 ⚠️**

Qwen2.5 tokenizer 的 ID 151665-151667 在 encoder 的 embed_tokens 中
原本对应 Qwen2-1.5B 的某些特殊 token（可能是 tool_call 相关的）。
现在你把这些位置"借用"为 `<imgpad>`/`<img>`/`</img>`。
这不是"歧义"（因为训练数据里不会出现那些原始 token），
但确实是一种隐式覆盖。

---

## 六、如果想做到"最干净"的方案

```
方案 A：也给 encoder 添加 token 并 resize（推荐）
───────────────────────────────────────────────
1. encoder (llm1) 也用 Qwen2TokenizerFast
2. 也添加 3 个特殊 token
3. resize_token_embeddings(151668)
4. 新 token embedding 从零训练
→ encoder 和 decoder 的 vocab 完全对齐

方案 B：使用原始 QWenTokenizer + 原始 token ID
───────────────────────────────────────────────
1. 保持使用 QWenTokenizer（但需要 tiktoken 依赖）
2. token ID 自然是 151857-151859
3. encoder 的预训练 embedding 完全保留
→ 最忠实于原始设计，但 tokenizer 版本旧

方案 C：你现在的做法（实用主义）
───────────────────────────────────────────────
1. 用 Qwen2TokenizerFast，添加 3 个 token
2. Encoder 不 resize，decoder resize
3. 特殊 token ID 不同于预训练，但训练中能学会
→ 可以工作，但预训练的特殊 token embedding 被浪费
```
