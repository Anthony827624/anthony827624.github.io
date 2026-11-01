---
title: "Agents的后训练流程"
tldr: "介绍智能体后训练的三个阶段（监督微调、在线策略蒸馏、强化学习）"
date: "2026-10-30"
---


后训练（post-training）指的是预训练（pre-training）之后的训练阶段，涵盖监督微调（supervised fine-tuning, SFT）、在线策略蒸馏（on-policy distillation）和强化学习（reinforcement learning, RL）。这个阶段能够将预训练模型转变为遵循指令、使用工具或解决特定任务的模型。当模型作为智能体（agent）时，它会多轮工作，调用工具，读取结果，决定下一步该做什么。智能体后训练因此必须考虑整个回合序列，而非单一答案。

## 1. 监督微调

模型学习复制代理轨迹，即代理每次调用工具完成任务以及任务结果的记录，这些通常由更强的模型生成。这种方法的主要优势是便宜、稳定。模型只看到正确的轨迹，并且不自己进行尝试。因此当它在完成任务时犯错，它仍没有学会如何纠错恢复。所以在线策略蒸馏和强化学习基于模型自身的输出进行训练，从而解决该问题。

### 1.1 设置

我们下面的代码目的是了解SFT的具体机制，而不是去训练一个强大的模型。因此代码很短，整个框架只有不到100行，没有训练代码。我们用`transformers`来加载模型和分词器（tokenizer），然后手动写损失（loss）和优化器（optimizer）。

只有少数需要安装的第三方库：

```bash
conda create -n post_train python=3.12 -y
conda activate post_train
pip install uv
uv pip install torch transformers
```

以`bfloat16`京都加载模型并转到GPU上（Mac是mps后端，大差不差）：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device(
    'cuda' if torch.cuda.is_available() 
    else 'mps' if torch.backends.mps.is_available() 
    else 'cpu'
)

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
```

### 1.2 监督就是next-token prediction

一个token是一个文本块，通常是单词或者词段，分词器将其映射为整数。模型读取一系列tokens，然后在每个位置计算出下一个token的官率分布。训练使用这些预测作为监督信号。在每个位置，模型都有一个预测分布和一个目标，也就是接下来实际的token，损失是他们之间的交叉熵。当模型自信且正确时，损失接近零。当模型自信但错误时，损失更大。

这是监督微调中所有的学习信号。在监督微调中没有关于好答案或者坏答案的概念，只有演示来表示接下来应该正确的token。因为信号是按照token计算的，因此模型在微调过程中的每个token都会得到一个梯度信号进行回传。

### 1.3 数据格式：提示词、输出、掩码

一个监督微调示例分为两部分。提示词是输入，比如用户的问题。模型产生的输出就会成为代理轨迹的一部分。我们希望模型能够学会生成输出而不是生成提示词，所以所有的损失都应该针对输出token进行反向传播。这就是掩码存在的原因。我们把输入和输出串联成一个序列并输入给模型，但是我让损失函数忽略输入的位置。在掩码序列中，每个输入token的label都是`-100`，而所有输出token的label都是真实的token id。这里的label也就是模型预测的目标token。

以下代码可以把聊天模版应用在一段简单对话，并且从助理掩码（assistant mask）中构建label：

```python
messages = [
    {"role": "user", "content": "Summarize in one word: always keep answers concise."}, 
    {"role": "assistant", "content": "Concise."}
]

enc = tokenizer.apply_chat_template(
    messages, 
    return_dict = True, 
    return_assistant_tokens_mask = True, 
    return_tensors="pt", 
).to(device)

input_ids = enc["input_ids"]
assistant_masks = torch.tensor(enc["assistant_masks"], device=device)

labels = input_ids.clone()
labels[assistant_masks == 0] = -100
```

其中，`messages`这个list保存对话内容，每一轮对话是一个`dict`。`apply_chat_template`将其渲染成token序列，然后`return_assistant_tokens_mask`能够标记哪些token属于`assistant`。除了`assistant`以外的其他掩码都是`0`，因此这里把其他掩码的label设为`-100`，如此就可以仅保留`assistant`中的内容作为损失计算的对象。

### 1.4 Pytorch训练循环

该循环包含3个步骤：

1. Forward pass：把序列输入模型得到logits
2. Shift & Align：平移logis和标签，使得每个位置都和其预测的token对齐
3. Loss & Backpropagate & Optimizer Step：计算交叉熵损失、梯度回传、优化器步进

其中logits是模型对每个输入位置得到的输出分布。例如位置`t`预测了位置`t+1`的token，所以我们需要去掉最后一个logit和第一个标签来对齐它们：

```python
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

for step in range(30):
    logits = model(input_ids).logits           # (1, T, vocab_size)
    shift_logits = logits[:, :-1, :]            # drop last position
    shift_labels = labels[:, 1:]                # drop first label
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,                      # skip masked tokens
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

- 因为最后一个logits的计算过程中，模型已经看到了完整的句子，当前序列里没有下一个token
- 因为第一个label是第“零”个token的预测目标值，但是根本没有第“零”个token，所以第1个label没有意义，要从第2个label开始

这就是一个完整的SFT step。至于`loss.backward()`和`optimizer.step()`中究竟发生了什么，不是这个blog的重点。

### 1.5 Take away

SFT本质上就是模仿。数据向模型展示该做什么，并且让模型去copy。其机制就是带有掩码的next-token prediction。所以数据就是上限，因为由SFT训练的模型并没有办法去产生其数据中没有过的行为。

因此SFT是稳定并且采样高效的，并且SFT的成本完全落在了如何编写和验证良好的output上。如果数据的提供者能够写出优质的output，那么SFT就是最简单的学习工具。但是如果我们只能对输出进行评分，而不能去量化地评估“哪个最好”，那么强化学习就是最好的工具。

## 2. 蒸馏

教师模型（teacher model）通过生成数据，或者对学生token的概率分布进行监督，从而使得学生模型（student）能够从教师模型中学习到“知识”的环节称为蒸馏。教师模型通常是一个更大的模型，或者多个专家模型，他们被提炼成一个学生模型（或者在自蒸馏（self-distillation）情况下是同一个模型）。监督信号非常稠密，对每个token都有一个对应的value，因此其训练所需要的采样比强化学习少（强化学习的奖励信号一般是稀疏信号）。

### 2.1 “大”老师和“小”学生

最基本的方法就是用一个“大”的、贵的老师，来训练一个“小”的、便宜的学生。

