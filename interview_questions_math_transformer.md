# 数学 + Transformer 面试题

## 题目说明
- 题量：10 题
- 题型：7 道单选 + 3 道多选
- 难度分布：简单 3 题 + 中等 5 题 + 困难 2 题
- 主题分布：数学类 3 题 + Transformer 架构 7 题
- 显示方式：移除 LaTeX，行内公式用反引号，推导过程用代码块，数学符号尽量使用 Unicode

---

## 一、单选题（7 题）

### 1. 数学类 | 简单 | 单选
设函数 `f(x) = ln(1 + eˣ)`，则 `f′(x)` 等于：

A. `eˣ`  
B. `eˣ / (1 + eˣ)`  
C. `1 / (1 + eˣ)`  
D. `ln(eˣ)`

答案：B

解析：  
对 `f(x) = ln(1 + eˣ)` 求导，使用链式法则：

```text
f′(x) = [1 / (1 + eˣ)] · eˣ
      = eˣ / (1 + eˣ)
```

因此正确答案是 `eˣ / (1 + eˣ)`。这个结果也正是 sigmoid 函数。

### 2. Transformer | 简单 | 单选
Transformer 中引入位置编码（Positional Encoding）的主要原因是：

A. 减少模型参数量  
B. 让模型感知序列中的顺序信息  
C. 替代注意力分数计算  
D. 只用于提升训练速度

答案：B

解析：  
自注意力本身对输入 token 的排列顺序不敏感。如果不显式加入位置信息，模型无法区分顺序不同但词相同的句子。位置编码的作用就是为模型补充顺序信息。

### 3. Transformer | 简单 | 单选
在自注意力（Self-Attention）中，某个 token 的 Query 通常会与谁计算相关性分数？

A. 只与自身的 Key  
B. 只与前一个 token 的 Key  
C. 与当前可见范围内所有 token 的 Key  
D. 与词表中所有词的 embedding

答案：C

解析：  
Self-Attention 的核心是让每个 token 与序列中其他可见 token 建立关系。对某个位置来说，它的 Query 会与所有可见位置的 Key 做点积，再经过 softmax 得到注意力权重。

### 4. 数学类 | 中等 | 单选
对于函数 `f(x) = x⁴ - 4x²`，点 `x = 0` 是：

A. 局部极大值点  
B. 局部极小值点  
C. 拐点  
D. 既不是极值点也不是拐点

答案：A

解析：  
先看一阶导，再看二阶导：

```text
f′(x) = 4x³ - 8x
f′(0) = 0

f″(x) = 12x² - 8
f″(0) = -8 < 0
```

因为 `f″(0) < 0`，所以 `x = 0` 是局部极大值点。

### 5. Transformer | 中等 | 单选
Scaled Dot-Product Attention 中，将点积结果除以 `√d_k` 的主要原因是：

A. 降低模型参数量  
B. 避免点积值过大，使 softmax 过于尖锐  
C. 让 Query 和 Key 的维度不同  
D. 为注意力机制增加非线性

答案：B

解析：  
当 `d_k` 较大时，Query 和 Key 的点积可能变得很大，导致 softmax 输出过于尖锐，训练时梯度变小、不稳定。除以 `√d_k` 是为了控制数值尺度，让训练更加稳定。

### 6. Transformer | 中等 | 单选
在 Transformer Decoder 中使用 masked self-attention 的主要作用是：

A. 让当前位置只能看到未来 token  
B. 防止当前位置看到未来 token  
C. 减少多头注意力中的 head 数量  
D. 用于替代前馈网络

答案：B

解析：  
在自回归生成任务中，当前位置只能依赖已经生成的历史内容，不能看到未来 token。mask 的作用就是遮住未来位置，保证训练和推理时的信息流一致。

### 7. Transformer | 中等 | 单选
相较于单头注意力（Single-Head Attention），多头注意力（Multi-Head Attention）的核心优势是：

A. 每个 head 一定学习完全相同的关系  
B. 可以在不同表示子空间中关注不同类型的依赖关系  
C. 将注意力复杂度从 `O(n²)` 降为 `O(n)`  
D. 不再需要残差连接

答案：B

解析：  
多头注意力通过不同的线性投影，把输入映射到多个子空间中分别建模，因此不同 head 有机会学习不同模式，例如局部依赖、长程依赖、语法关系和语义关系。

---

## 二、多选题（3 题）

### 8. 数学类 | 中等 | 多选
关于交叉熵（Cross Entropy）和 KL 散度，下列说法正确的是：

A. `KL(p ∥ q) ≥ 0`  
B. `KL(p ∥ q) = KL(q ∥ p)`  
C. 当 `p = q` 时，`KL(p ∥ q) = 0`  
D. `H(p, q) = H(p) + KL(p ∥ q)`

答案：A、C、D

解析：  
几个关键性质如下：

```text
1. KL(p ∥ q) ≥ 0
2. KL 散度通常不对称，即 KL(p ∥ q) ≠ KL(q ∥ p)
3. 当 p = q 时，KL(p ∥ q) = 0
4. H(p, q) = H(p) + KL(p ∥ q)
```

因此 A、C、D 正确，B 错误。

### 9. Transformer | 困难 | 多选
关于 Transformer 中的残差连接（Residual Connection）与 LayerNorm，下列说法正确的是：

A. 残差连接有助于缓解深层网络中的梯度传播问题  
B. 在较深的 Transformer 中，Pre-Norm 通常比 Post-Norm 更稳定  
C. LayerNorm 的统计量通常在 batch 维度上计算  
D. 残差相加要求子层输出与输入维度兼容，必要时可通过投影对齐

答案：A、B、D

解析：  
- 残差连接为梯度提供了更直接的传播路径，所以 A 正确。
- 在较深的 Transformer 中，Pre-Norm 往往比 Post-Norm 更稳定，所以 B 正确。
- LayerNorm 是对单个样本的特征维做归一化，不依赖 batch 统计量，所以 C 错误。
- 残差相加前要求张量维度兼容，不兼容时通常需要额外投影，所以 D 正确。

### 10. Transformer | 困难 | 多选
关于 Transformer 的训练与推理，下列说法正确的是：

A. 在自回归训练中，Decoder 常使用 teacher forcing  
B. 自回归推理时可以缓存历史 token 的 Key 和 Value 以减少重复计算  
C. 在编码器-解码器注意力中，Query 通常来自 Decoder，Key 和 Value 通常来自 Encoder 输出  
D. 增加 attention head 数量一定会严格降低推理延迟

答案：A、B、C

解析：  
- Teacher forcing 是自回归训练中的常见做法，即训练时使用真实历史 token 作为条件，所以 A 正确。
- 推理时使用 KV Cache，可以避免对历史 token 重复计算投影，所以 B 正确。
- 在 Encoder-Decoder Attention 中，通常是 Decoder 提供 Query，Encoder 输出提供 Key 和 Value，所以 C 正确。
- head 数量增加并不一定降低延迟，反而可能增加计算和访存开销，所以 D 错误。

---

## 三、答案速览

| 题号 | 题型 | 主题 | 难度 | 答案 |
|---|---|---|---|---|
| 1 | 单选 | 数学类 | 简单 | B |
| 2 | 单选 | Transformer | 简单 | B |
| 3 | 单选 | Transformer | 简单 | C |
| 4 | 单选 | 数学类 | 中等 | A |
| 5 | 单选 | Transformer | 中等 | B |
| 6 | 单选 | Transformer | 中等 | B |
| 7 | 单选 | Transformer | 中等 | B |
| 8 | 多选 | 数学类 | 中等 | A、C、D |
| 9 | 多选 | Transformer | 困难 | A、B、D |
| 10 | 多选 | Transformer | 困难 | A、B、C |
