# 面试笔试题 — 数学基础 & Transformer 架构

> **题型分布**：单选 7 题 + 多选 3 题 = 共 10 题
> **难度分布**：简单 30%（3 题）· 中等 50%（5 题）· 困难 20%（2 题）
> **知识分布**：数学（非线性代数）3 题 · Transformer 架构 7 题

---

## 第一题（单选 | 数学 · Softmax 性质 | ⭐ 简单）

设带温度参数的 softmax 函数为 `softmax(zᵢ / T)`，当温度 T → 0⁺ 和 T → +∞ 时，输出分布的行为分别是：

- **A.** 趋向均匀分布；趋向 one-hot 分布
- **B.** 趋向 one-hot 分布；趋向均匀分布
- **C.** 均趋向均匀分布
- **D.** 均趋向 one-hot 分布

> **答案：B**
>
> 温度 T 控制分布的"锐度"：
>
> - T → 0⁺ 时，`zᵢ / T` 中最大值项被极端放大，softmax 输出趋向 **one-hot**（概率集中在最大 logit 上）。
> - T → +∞ 时，所有 `zᵢ / T → 0`，各项差异被抹平，softmax 输出趋向 **均匀分布**。
>
> 这一性质在推理采样（temperature sampling）和知识蒸馏（soft label）中被广泛使用。

---

## 第二题（单选 | Transformer · RoPE | ⭐ 简单）

关于旋转位置编码（RoPE），以下哪项描述最为准确？

- **A.** RoPE 将绝对位置信息以加法方式叠加到 token embedding 上，与原始 Transformer 的正弦位置编码机制相同
- **B.** RoPE 通过对 Q 和 K 向量施加与位置相关的旋转变换，使 Q 和 K 的内积自然包含相对位置信息
- **C.** RoPE 仅对 Value 向量施加位置编码，不影响注意力分数的计算
- **D.** RoPE 在每个注意力头中使用相同的旋转角度，不区分维度

> **答案：B**
>
> RoPE（Su et al., 2021）的核心思想：对位置 m 的 Q 向量和位置 n 的 K 向量分别施加旋转矩阵 R(m) 和 R(n)，则内积 `(R(m)·q)ᵀ(R(n)·k) = qᵀR(n-m)k`，自然只依赖相对位置差 (n - m)。
>
> - A 错：RoPE 是乘性旋转，不是加法。
> - C 错：RoPE 作用于 Q 和 K，不作用于 V。
> - D 错：不同维度对应不同的旋转频率（类似正弦编码的多频率设计）。

---

## 第三题（单选 | Transformer · RMSNorm | ⭐ 简单）

相较于 LayerNorm，现代大语言模型（如 LLaMA）普遍采用的 RMSNorm 的主要区别在于：

- **A.** RMSNorm 在 batch 维度上归一化，而 LayerNorm 在特征维度上归一化
- **B.** RMSNorm 省略了减均值（去中心化）步骤，仅用均方根（RMS）进行缩放
- **C.** RMSNorm 不包含任何可学习参数
- **D.** RMSNorm 的计算复杂度高于 LayerNorm

> **答案：B**
>
> 两者对比：
>
> ```
> LayerNorm(x) = (x - mean(x)) / std(x) · γ + β    (去中心化 + 缩放 + 平移)
> RMSNorm(x)   = x / RMS(x) · γ                     (仅 RMS 缩放)
>
> 其中 RMS(x) = √(mean(x²))
> ```
>
> - A 错：两者都在特征维度（hidden dimension）上操作。
> - B 对：RMSNorm 去掉了减均值和加偏置步骤，实验表明去中心化对 Transformer 并非必要，且能减少计算开销。
> - C 错：RMSNorm 保留了可学习的缩放参数 γ。
> - D 错：RMSNorm 计算量更低（省去了均值计算和偏置）。

---

## 第四题（单选 | 数学 · 梯度推导 | ⭐⭐ 中等）

设模型输出 logits 为 z，经 softmax 得到概率 `p = softmax(z)`，真实标签为 one-hot 向量 y，交叉熵损失 `L = -Σᵢ yᵢ log(pᵢ)`。则损失对 logits 的梯度 `∂L/∂zᵢ` 等于：

- **A.** `yᵢ - pᵢ`
- **B.** `pᵢ - yᵢ`
- **C.** `-yᵢ / pᵢ`
- **D.** `pᵢ(1 - pᵢ)`

> **答案：B**
>
> 这是深度学习中最经典的梯度推导之一。分两步：
>
> ```
> ∂L/∂zᵢ = Σⱼ (∂L/∂pⱼ) · (∂pⱼ/∂zᵢ)
> ```
>
> 其中 `∂L/∂pⱼ = -yⱼ/pⱼ`，而 softmax 的 Jacobian 为 `∂pⱼ/∂zᵢ = pⱼ(δᵢⱼ - pᵢ)`。
>
> 合并化简后得到极其简洁的结果：
>
> ```
> ∂L/∂zᵢ = pᵢ - yᵢ
> ```
>
> 即 **softmax 输出减去 one-hot 标签**。这一简洁性是 softmax + cross-entropy 组合如此流行的重要原因。
> - C 是 `∂L/∂pᵢ`（对概率的梯度），不是对 logits 的梯度。
> - D 是 sigmoid 导数，与此处无关。

---

## 第五题（单选 | Transformer · Causal Mask | ⭐⭐ 中等）

在自回归语言模型的 Self-Attention 中，关于 Causal Mask 的实现，以下描述正确的是：

- **A.** 将注意力分数矩阵的**下三角**部分设为 -∞，保留上三角
- **B.** 将注意力分数矩阵的**严格上三角**部分（不含对角线）设为 -∞，在 softmax **之前**施加
- **C.** Causal Mask 仅在推理阶段使用，训练时使用完整的双向注意力
- **D.** Causal Mask 在 softmax **之后**将未来位置的权重置零

> **答案：B**
>
> Causal Mask 的正确实现要点：
>
> ```
> Attention Score Matrix (n × n):
>
>   位置  1   2   3   4
>   1  [ s   -∞  -∞  -∞ ]    位置 1 只能看自己
>   2  [ s   s   -∞  -∞ ]    位置 2 能看 1、2
>   3  [ s   s   s   -∞ ]    位置 3 能看 1、2、3
>   4  [ s   s   s   s  ]    位置 4 能看全部
>
> s = 正常的 attention score，-∞ 经 softmax 后变为 0
> ```
>
> - A 错：应该遮住的是上三角（未来位置），不是下三角。
> - B 对：严格上三角设为 -∞，对角线保留（每个位置可以 attend 到自己），且必须在 softmax 之前施加。
> - C 错：训练和推理都使用 Causal Mask，保证自回归的因果性。
> - D 错：在 softmax 之后置零会导致注意力权重不再归一化为 1，语义不正确。

---

## 第六题（单选 | Transformer · SwiGLU FFN | ⭐⭐ 中等）

现代大语言模型（如 LLaMA）中，前馈网络（FFN）普遍从标准 ReLU FFN 升级为 SwiGLU 结构。以下描述正确的是：

- **A.** SwiGLU 引入门控机制，FFN 从两个权重矩阵变为三个，表达能力更强
- **B.** SwiGLU 完全去掉了非线性激活函数，仅用线性变换
- **C.** SwiGLU 的参数量和计算量均低于标准 ReLU FFN
- **D.** SwiGLU 中的 Swish 激活函数等价于 ReLU

> **答案：A**
>
> 标准 FFN 与 SwiGLU 的对比：
>
> ```
> 标准 FFN:   FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂         → 2 个权重矩阵
>
> SwiGLU:     FFN(x) = [Swish(xW₁) ⊙ (xW₃)] W₂        → 3 个权重矩阵
>                       ~~~~~~~~~~~   ~~~~~
>                        激活分支     门控分支    ⊙ = 逐元素乘法
> ```
>
> - A 对：门控分支 xW₃ 控制信息的通过量，新增第三个矩阵 W₃，实验证明质量显著提升。
> - B 错：SwiGLU 使用 Swish（即 SiLU）激活函数，`Swish(x) = x · σ(x)`。
> - C 错：多了一个矩阵，参数量和计算量略有增加（通常通过缩小中间维度来补偿）。
> - D 错：Swish 是平滑的非单调函数，与 ReLU 不同。

---

## 第七题（多选 | Transformer · GQA/MQA | ⭐⭐ 中等）

关于 Multi-Head Attention（MHA）、Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA），以下说法**正确**的有：

- **A.** MQA 让所有 Query 头共享同一组 K、V，KV Cache 缩减为 MHA 的 1/h（h 为头数）
- **B.** GQA 将 Query 头分成若干组，每组共享一组 K、V，是 MHA 和 MQA 之间的折中方案
- **C.** 当 GQA 的 KV 组数等于 Query 头数 h 时，GQA 退化为标准 MHA；当组数为 1 时退化为 MQA
- **D.** MQA 在模型质量上与 MHA 完全一致，没有任何精度损失

> **答案：A、B、C**
>
> ```
> MHA:  h 个 Q 头，h 个 KV 头     → KV Cache = h 份
> GQA:  h 个 Q 头，g 个 KV 头     → KV Cache = g 份  (1 < g < h)
> MQA:  h 个 Q 头，1 个 KV 头     → KV Cache = 1 份
>
> GQA(g=h) = MHA,  GQA(g=1) = MQA
> ```
>
> - A ✓：MQA 所有头共享一组 KV，Cache 仅需 1/h。
> - B ✓：GQA（Ainslie et al., 2023）提出分组策略，平衡推理效率与模型质量。
> - C ✓：GQA 是 MHA 和 MQA 的统一框架，两个极端分别对应 MHA 和 MQA。
> - D ✗：MQA 压缩了 KV 的表示能力，实践中有一定质量损失，GQA 正是为了缓解这一问题而提出。

---

## 第八题（多选 | Transformer · KV Cache | ⭐⭐ 中等）

关于大语言模型推理中的 KV Cache 技术，以下说法**正确**的有：

- **A.** KV Cache 缓存已计算的 Key 和 Value 张量，避免自回归解码时对历史 token 的重复计算
- **B.** KV Cache 的显存占用与序列长度成线性关系，与层数和隐藏维度成正比
- **C.** Prefill 阶段处理完整 prompt，可并行计算所有 token 的 KV 并写入 Cache；Decode 阶段逐 token 生成，每步追加新的 KV
- **D.** KV Cache 同时缓存了 Query、Key 和 Value 三者

> **答案：A、B、C**
>
> - A ✓：在自回归解码中，已生成 token 的 K、V 不会改变，缓存后每步只需计算新 token 的 KV。
> - B ✓：KV Cache 总显存 ≈ `2 × 层数 × 序列长度 × KV维度 × 精度字节数`，与序列长度 n 成线性关系。
> - C ✓：这是 Prefill-Decode 两阶段架构的标准流程。Prefill 是 compute-bound，Decode 是 memory-bound。
> - D ✗：只缓存 K 和 V。Query 在每个解码步仅需计算当前新 token 的即可，无需缓存历史 Q。

---

## 第九题（单选 | 数学 · 交叉熵与 KL 散度 | ⭐⭐⭐ 困难）

训练语言模型时，我们最小化模型分布 Q_θ 与数据分布 P_data 之间的交叉熵 `H(P_data, Q_θ)`。以下推导和结论中，**正确**的是：

- **A.** 最小化 H(P_data, Q_θ) 等价于最小化 D_KL(P_data ‖ Q_θ)，因为数据分布的熵 H(P_data) 是关于 θ 的常数
- **B.** 最小化交叉熵等价于最大化模型分布 Q_θ 自身的熵 H(Q_θ)
- **C.** D_KL(P_data ‖ Q_θ) = H(P_data, Q_θ) - H(Q_θ)
- **D.** 最小化 D_KL(Q_θ ‖ P_data)（反向 KL）与最小化 D_KL(P_data ‖ Q_θ)（正向 KL）对 Q_θ 的优化结果完全相同

> **答案：A**
>
> KL 散度与交叉熵的关系：
>
> ```
> D_KL(P ‖ Q) = H(P, Q) - H(P)
>
> 即：H(P, Q) = H(P) + D_KL(P ‖ Q)
> ```
>
> - A ✓：因为 H(P_data) 不含参数 θ，对 θ 求梯度时是常数，所以 `min_θ H(P_data, Q_θ) ⟺ min_θ D_KL(P_data ‖ Q_θ)`。这也等价于最大化数据的对数似然。
> - B ✗：最小化交叉熵是在让 Q_θ 尽可能拟合 P_data，不是让 Q_θ 的熵最大化。
> - C ✗：正确关系应为 `D_KL(P ‖ Q) = H(P, Q) - H(P)`，不是减去 H(Q)。
> - D ✗：正向 KL（mode-covering）倾向让 Q_θ 覆盖 P_data 的所有模式；反向 KL（mode-seeking）倾向让 Q_θ 集中在 P_data 的主要模式上。两者优化行为截然不同，这在 VAE、RLHF 等场景中有重要影响。

---

## 第十题（多选 | Transformer · MLA | ⭐⭐⭐ 困难）

关于 DeepSeek-V2 提出的 Multi-head Latent Attention（MLA），以下说法**正确**的有：

- **A.** MLA 通过低秩联合压缩，将 KV 投影到低维潜在空间（latent space），推理时 KV Cache 只需存储低维的潜在向量
- **B.** MLA 的 KV Cache 显存开销可以显著低于 MQA/GQA，因为潜在向量的维度远小于原始多头 KV 的总维度
- **C.** MLA 在推理时从缓存的潜在向量恢复各头的 K、V，通过将恢复矩阵吸收进 Q 和 Output 的投影中来避免额外计算开销
- **D.** MLA 完全不需要 KV Cache，通过每步重新计算所有历史 token 的 KV 来节省显存

> **答案：A、B、C**
>
> MLA 的核心思想：
>
> ```
> 标准 MHA:  缓存 [K₁, K₂, ..., Kₕ, V₁, V₂, ..., Vₕ]  → 维度 = 2 × h × d_k
>
> MLA:       缓存 c = W_down · x                           → 维度 = d_c (≪ 2hd_k)
>            推理时: K, V = W_up_K · c,  W_up_V · c       (恢复各头 KV)
> ```
>
> - A ✓：MLA 对 KV 做低秩联合压缩，缓存压缩后的潜在向量 c，维度远小于完整 KV。
> - B ✓：MQA 缓存 1 组完整 KV（维度 2d_k），GQA 缓存 g 组。MLA 的潜在向量维度 d_c 可设计为极小值，Cache 开销更低。
> - C ✓：这是 MLA 的关键工程技巧——恢复矩阵 W_up 可以与 Q 投影和 Output 投影合并，推理时不需要显式恢复完整 KV，计算量不增加。
> - D ✗：MLA 仍然使用 KV Cache（缓存潜在向量），只是缓存的内容从完整 KV 变为压缩后的低维表示。

---

## 附加题（多选 | 数学 · Sigmoid | ⭐⭐ 中等）

关于 Sigmoid 函数 `σ(x) = 1 / (1 + e⁻ˣ)` 及其在现代深度学习中的关联，以下说法**正确**的有：

- **A.** σ(x) 满足 `σ(-x) = 1 - σ(x)`，即关于点 (0, 0.5) 中心对称
- **B.** `σ'(x) = σ(x) · [1 - σ(x)]`，其导数最大值为 **0.5**，出现在 x = 0 处
- **C.** Softplus 函数 `ln(1 + eˣ)` 可视为 ReLU 的平滑近似，其导数恰好等于 σ(x)
- **D.** SwiGLU 中使用的 Swish 激活 `Swish(x) = x · σ(x)` 是一个非单调函数，在 x < 0 区域可取负值

> **答案：A、C、D**
>
> 逐项分析：
>
> - A ✓：验证如下：
>
> ```
> σ(-x) = 1 / (1 + eˣ)
> σ(x) + σ(-x) = 1/(1+e⁻ˣ) + 1/(1+eˣ) = eˣ/(eˣ+1) + 1/(eˣ+1) = 1
> ```
>
> - B ✗：**最大值是 0.25 而非 0.5**。在 x = 0 处 σ(0) = 0.5，所以：
>
> ```
> σ'(0) = 0.5 × (1 - 0.5) = 0.5 × 0.5 = 0.25
> ```
>
> 令 u = σ(x) ∈ (0,1)，则 σ'= u(1-u) 在 u = 0.5 时取最大值 0.25。正是这个"最大才 0.25"的特性导致 Sigmoid 在深层网络中容易梯度消失。
>
> - C ✓：对 Softplus 求导：
>
> ```
> d/dx ln(1 + eˣ) = eˣ / (1 + eˣ) = 1 / (1 + e⁻ˣ) = σ(x)
> ```
>
> 即 Softplus 是 Sigmoid 的原函数（不定积分），Sigmoid 是 Softplus 的导数。
>
> - D ✓：Swish(x) = x · σ(x) 的行为：
>
> ```
> x = -1:  Swish(-1) = -1 × σ(-1) = -1/(1+e) ≈ -0.269  （取负值）
> x → -∞: Swish(x) → 0⁻                                  （从下方趋近 0）
> x → +∞: Swish(x) → x                                    （近似线性）
> ```
>
> Swish 先下降到负值再上升，是非单调的，这使它比 ReLU 更具表达力。

---

## 题目总览

| 题号 | 题型 | 知识领域 | 难度 |
|:---:|:---:|:---:|:---:|
| 1 | 单选 | 数学 · Softmax 温度 | ⭐ 简单 |
| 2 | 单选 | Transformer · RoPE 旋转位置编码 | ⭐ 简单 |
| 3 | 单选 | Transformer · RMSNorm | ⭐ 简单 |
| 4 | 单选 | 数学 · Softmax + CE 梯度推导 | ⭐⭐ 中等 |
| 5 | 单选 | Transformer · Causal Mask | ⭐⭐ 中等 |
| 6 | 单选 | Transformer · SwiGLU FFN | ⭐⭐ 中等 |
| 7 | 多选 | Transformer · GQA / MQA / MHA | ⭐⭐ 中等 |
| 8 | 多选 | Transformer · KV Cache | ⭐⭐ 中等 |
| 9 | 单选 | 数学 · 交叉熵与 KL 散度 | ⭐⭐⭐ 困难 |
| 10 | 多选 | Transformer · MLA（DeepSeek-V2） | ⭐⭐⭐ 困难 |
| 附加 | 多选 | 数学 · Sigmoid 性质 | ⭐⭐ 中等 |
