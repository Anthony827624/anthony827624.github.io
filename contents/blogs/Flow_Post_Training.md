---
title: "Flow Model的强化学习后训练：从反向过程到前向过程"
tldr: "梳理Flow-GRPO, DanceGRPO, MixGRPO, TempFlow-GRPO, FPO, AWM, DiffusionNFT, DiffusionART八篇工作，围绕在流模型无法拿到log-likelihood，如何拿到policy gradient里的似然比。"
date: "2026-10-30"
---

## Introduction: 为什么Flow Model的RL是个问题

### 0.1 从LLM说起

大语言模型（LLM）的强化学习后训练可以概括成三步：

1. 让模型自己生成回答
2. 用奖励（reward）给回答打分
3. 提高高分回答的概率、降低低分回答的概率

第三部依赖**策略（policy）给这个回答的对数概率$\log\pi_\theta(y\mid c)$**。对自回归（Autoregressive）模型来说，它几乎是免费的，就是每个token的log-softmax加起来，仅仅需要一次前向传播。

但是图像和视频生成的主流模型，比如SD3.5、FLUX、HunyuanVideo，已经从扩散模型（Diffusion Model）转向了流匹配（Flow Matching/Rectified Flow/FlowMap）。同样的三步流程如果直接套用，会在以下几个地方卡住：

1. 没有Exploration：Flow Model常用确定性常微分方程（ODE）采样，模型自身没有随机性可供探索。
2. 没有likelihood：生成一张图要沿ODE积分几十步，这张图的精确对数概率就需要沿整条轨迹去积分速度场的散度，代价非常高，无法放进训练循环。
3. 没有Efficiency：每个样本都要几十次前向传播，在线RL又需要大量rollout，采样成本极高无比。

所以就有一些针对Flow Model的后训练方法出来了！

### 0.2 全文的Main Problem

PPO、GRPO这类policy gradient方法的核心是一个**似然比（likelihood ratio）**：
$$
r(\theta)=\frac{\pi_\theta(x\mid c)}{\pi_{\theta_{\text{old}}}(x\mid c)}
$$
本文讨论的8篇工作，其实都可以看成在回答同一个问题：

> Flow Model拿不到$\log\pi_\theta$，那么这个比值从哪里来？

目前有两条主要路线，外加一个试图融合两者的方法：

- 路线A：反向过程RL。把去噪链拆成几十个小步，每一步人为注入高斯噪声，也就是把ODE改造成等价的SDE。这样子每一步都是一个高斯分布，比值就有了闭式解。代表工作为Flow-GRPO, DanceGRPO, MixGRPO, TempFlow-GRPO。
- 路线B：前向过程RL。不碰采样过程，而是用训练时就在用的Flow Matching损失作为对数似然的ELBO。采样器可以是任意黑盒。代表工作为FPO, AWM, Diffusion NFT。
- 融合：DiffusionART同时诊断了两条路线的弱点，A 的信用分配差，B 的训练不稳定，并尝试取两者之长。

| 论文          | 路线    | TLDR                                                         |
| ------------- | ------- | ------------------------------------------------------------ |
| Flow-GRPO     | Reverse | 开山之作：ODE2SDE，让每一步都有高斯似然；训练时使用更少去噪步 |
| DanceGRPO     | Reverse | 统一Diffusion和Rectified Flow，扩展到视频，关注规模化稳定性  |
| MixGRPO       | Reverse | 只在一个滑动窗口内用SDE并优化，窗口外用ODE，大幅降低开销     |
| TempFlow-GRPO | Reverse | 轨迹分叉给中间步做credit assignment，按噪声水平给每一步加权  |
| FPO           | Forward | 用Flow Matching损失之差近似对数似然比，将采样器视为黑盒      |
| AWM           | Forward | 证明Reverse Process RL等价于“带噪声目标”的Flow Matching，因而方差更大 |
| DiffusionNFT  | Forward | 正负样本对比，直接在前向过程上做策略改进，跳出Policy Gradient框架 |
| DiffusionART  | Mixture | 先聚合Reward再施加到Reverse Process上，配合信赖域式的恢复更新 |

本文先在第1节把所需的背景知识讲清楚：Flow Matching、Policy Gradient，以及"似然比从哪来"这个问题的两种标准答案。之后的章节逐条路线展开。

## 1. 预备知识

### 1.1 Flow Matching：学一个速度场

本文统一采用Rectified Flow的约定（与Flow-GRPO一致）：

- $x_0\sim p_{\text{data}}$是真实数据，$x_1\sim\mathcal{N}(0,I)$是纯噪声
- $t\in[0,1]$，其中$t=0$是数据端，$t=1$是噪声端。

Rectified Flow在数据和噪声之间拉一条直线：
$$
x_t=(1-t)x_0+tx_1\label{interp}
$$
沿这条直线的速度是$u=\frac{dx_t}{dt}=x_1-x_0$，也就是由噪声端指向数据端。网络$v_\theta(x_t,t)$被训练去回归这个速度，这就是条件流匹配（Conditional Flow Matching, CFM）：
$$
\mathcal{L}_{\text{CFM}}(\theta)=\mathbb{E}_{t,x_0,x_1}\left[ \| v_\theta(x_t,t)-(x_1-x_0) \|^2 \right] \label{cfm}
$$
训练好之后，从噪声出发，沿着ODE一路从$t=1$积分到$t=0$，即$dx_t=v_\theta(x_t,t)$，就得到一个样本。最简单的Euler离散化是：
$$
x_{t-\Delta t}=x_t-v_{\theta}(x_t,t)\Delta t \label{euler}
$$
一个点$x_t$可能来自很多不同的（数据，噪声）配对，每个配对给出的“目标速度”$x_1-x_0$都不一样。网路学到的是这些目标速度的**平均值**，也就是边缘速度场$v_t(x)=\mathbb{E}[x_1-x_0\mid x_t=x]$。用单个配对的速度当监督信号看起来很粗糙，但下面的证明说明，它和直接回归边缘速度场的梯度完全一样，代价只是多了一项与参数无关的“目标噪声”。

<details>

<summary>

展开证明：Conditional Flow Matching与Marginal Flow Matching的梯度相同。

</summary>

记$u=x_1-x_0$，边缘速度场为$v_t(x) = \mathbb E[u \mid x_t = x]$。把条件损失按$v_t$拆开：
$$
\begin{aligned}
\mathbb E\big\|v_\theta(x_t,t) - u\big\|^2 = {}& \mathbb E\big\|v_\theta(x_t,t)-v_t(x_t)\big\|^2 + \mathbb E\big\|v_t(x_t) - u\big\|^2 \\
& + 2\,\mathbb E\big[\langle v_\theta(x_t,t) - v_t(x_t),\; v_t(x_t) - u\rangle\big]
\end{aligned}\notag
$$
交叉项是$\mathbb{E}\big[ \langle a,b \rangle \big]$，其中：

- $a=v_\theta(x_t,t) - v_t(x_t)$：网络输出减去边缘速度场。两者都是$x_t$和$t$的函数，所以$a$只依赖$x_t$
- $b=v_t(x_t) - u$：其中$u=x_1-x_0$，因此$b$就依赖于$x_t$具体是由哪一对$(x_0,x_1)$所构成的

这里的关键是，同一个$x_t$可以由很多不同的$(x_0,x_1)$配对插值得到。所以固定$x_t$后，$a$就定下来了，而$u$（因此$b$）仍然是随机的。

**分两层求期望**：我们可以把总期望拆成“外层对$x_t$、内层对给定$x_t$时的$(x_0,x_1)$”：
$$
\mathbb{E}\big[ \langle a,b \rangle \big]=\mathbb{E}_{x_t}\big[\mathbb{E}[\langle a,b \rangle]\mid x_t\big]
$$
对于内层$\mathbb{E}[\cdot\mid x_t]$：

- $a$是常数：因此可以提到期望外面，即$\mathbb{E}[\langle a,b \rangle\mid x_t]=\langle a,\mathbb{E}[b\mid x_t]\rangle$。
- $\mathbb{E}[b\mid x_t]$为零：由于$v_t(x_t)$在给定$x_t$时是常数，所以$\mathbb{E}[b\mid x_t]=v_t(x_t)-\mathbb{E}[u\mid x_t]=v_t(x_t)-v_t(x_t)=0$。

所以内层是$\langle a,0 \rangle=0$，外层再取期望还是0，交叉项消失！
$$
  \mathcal L_{\text{CFM}}(\theta) = \underbrace{\mathbb E\big\|v_\theta(x_t,t) - v_t(x_t)\big\|^2}_{\text{边缘 FM 损失}} + \underbrace{\mathbb E\big\|v_t(x_t) - u\big\|^2}_{\text{与 }\theta\text{ 无关的目标噪声}} \notag
$$

第二项不依赖 $\theta$，所以两种损失有相同的最优解和相同的梯度。但用单个样本估计梯度时，第二项代表的目标波动会进入随机梯度，使梯度估计的方差更大。

</details>
