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
<summary>展开证明：Conditional Flow Matching与Marginal Flow Matching的梯度相同。</summary>

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
\mathbb{E}\big[ \langle a,b \rangle \big]=\mathbb{E}_{x_t}\big[\mathbb{E}[\langle a,b \rangle]\mid x_t\big] \notag
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

### 1.2 为什么拿不到似然

对确定性ODE采样得到的样本$x_0$，其精确对数概率满足（瞬时变量替换公式）：
$$
\log p_0(x_0) = \log p_1(x_0) + \int_{0}^{1}\nabla\cdot v_t(x_t)dt \label{cov}
$$
其中积分沿着生成这个样本的整条ODE轨迹进行。

速度场在某处“发散”（散度为正），概率质量就被摊薄；“汇聚”（散度为负）则被压缩。如果要知道终点的概率密度，就需要把沿途每一处的摊薄或者压缩全都累加起来。这有两个麻烦：

1. 散度$\nabla\cdot v$是雅可比矩阵的trace，维度等于latent的维度（图像latent动辄数万维），只能用Hutchinson之类的随机估计，方差很大。
2. 要对$\theta$求梯度，就得穿过整条轨迹反向传播。在Flow-GRPO中也明确指出，确定性动力学下计算$p(x_{t-1}\mid x_t,c)$需要做散度估计，代价很高。

此外还有一个更根本的问题：确定性采样除了初始噪声没有任何随机性，RL赖以学习的**探索**无从谈起。

<details>

<summary>展开证明：瞬时变量替换公式</summary>

在ODE，即$\frac{dx_t}{dt}=v_t(x_t)$下，边缘密度满足连续性方程：
$$
\partial_t p_t(x) + \nabla\cdot(p_t(x)v_t(x))=0 \notag
$$
沿一条轨迹$x_t$对$\log p_t(x_t)$求全导数：
$$
\frac{\mathrm d}{\mathrm dt}\log p_t(x_t) = \frac{1}{p_t}\Big(\partial_t p_t + \nabla p_t \cdot v_t\Big) = \frac{1}{p_t}\Big(-\nabla\cdot(p_t v_t) + \nabla p_t\cdot v_t\Big) = -\nabla\cdot v_t(x_t) \notag
$$

最后一步用了 $\nabla\cdot(p_t v_t) = \nabla p_t\cdot v_t + p_t\,\nabla\cdot v_t$。从 $t=1$ 积分到 $t=0$：
$$
\log p_0(x_0) - \log p_1(x_1) = \int_1^0 -\nabla\cdot v_t(x_t)\,\mathrm dt = \int_0^1 \nabla\cdot v_t(x_t)\,\mathrm dt \notag
$$
即得结论。

</details>

### 1.3 Policy Gradient、PPO和GRPO

给定提示词$c$，策略$\pi_\theta(\cdot\mid c)$生成样本$x$，奖励为$r(x,c)$。目标是最大化期望奖励$J(\theta) = \mathbb E_{c}\,\mathbb E_{x\sim\pi_\theta(\cdot\mid c)}[r(x,c)]$，其梯度为：
$$
\nabla_\theta J(\theta) = \mathbb E_{c,\,x\sim\pi_\theta}\Big[\big(r(x,c) - b(c)\big)\,\nabla_\theta \log \pi_\theta(x\mid c)\Big] \label{pg}
$$
其中$b(c)$是任意只依赖$c$的基线（Baseline），不改变梯度的期望，但能降低方差。$A = r - b$称为**优势**（advantage）。这个公式的意思很简单：把高于平均分的样本的对数概率往上推，低于平均分的往下压。

**PPO** 在此基础上做了两件事：用旧策略$\pi_{\theta_{\text{old}}}$采集的样本做多轮更新（因此需要重要性比值 $r(\theta)$ 修正），并把比值截断在$[1-\varepsilon, 1+\varepsilon]$内，防止一步走得太远：
$$
\mathcal J_{\text{PPO}}(\theta) = \mathbb E\Big[\min\big(r(\theta)\,A,\; \operatorname{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon)\,A\big)\Big],\qquad r(\theta) = \frac{\pi_\theta(x\mid c)}{\pi_{\theta_{\text{old}}}(x\mid c)} \label{ppo}
$$
**GRPO** 去掉了 PPO 的价值网络。对同一个 prompt 采样一组 $G$ 个样本，用组内统计量归一化奖励作为优势：
$$
A_i = \frac{r(x^i, c) - \operatorname{mean}\big(\{r(x^j, c)\}_{j=1}^G\big)}{\operatorname{std}\big(\{r(x^j, c)\}_{j=1}^G\big)} \label{grpo}
$$
很多实现还会加一项 $-\beta\,D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$，让模型不要离预训练模型太远。图像生成天然适合GRPO：对同一个prompt生成一组图，“比同组平均好多少”就是最自然的优势估计，不需要额外训练一个价值网络。

注意：式 $\eqref{pg}$ 和式 $\eqref{ppo}$ 里都出现了$\pi_\theta(x\mid c)$，这正是上文说拿不到的东西。

<details>
<summary>展开证明：对数导数技巧与基线不变性</summary>

**对数导数技巧**：对固定$c$：
$$
\begin{aligned}
\nabla_\theta \mathbb E_{x\sim\pi_\theta}[r(x)] &= \int r(x)\,\nabla_\theta \pi_\theta(x)\,\mathrm dx \\
&= \int r(x)\,\pi_\theta(x)\,\nabla_\theta\log\pi_\theta(x)\,\mathrm dx \\
&= \mathbb E_{x\sim\pi_\theta}\big[r(x)\,\nabla_\theta\log\pi_\theta(x)\big]
\end{aligned}\notag
$$
这里用了$\nabla_\theta\pi_\theta = \pi_\theta\nabla_\theta\log\pi_\theta$。注意奖励$r$只以数值的形式出现，不需要对它求导，所以奖励可以是检测器、OCR、VLM 打分等任意黑盒。

**基线不变性**：对任意不依赖$x$的$b$：
$$
\mathbb E_{x\sim\pi_\theta}\big[b\,\nabla_\theta\log\pi_\theta(x)\big] = b\int\nabla_\theta\pi_\theta(x)\,\mathrm dx = b\,\nabla_\theta\!\int\pi_\theta(x)\,\mathrm dx = b\,\nabla_\theta 1 = 0 \notag
$$
所以减去基线不改变梯度期望。GRPO 的组内均值就是对$b(c)$的一个蒙特卡洛估计；除以组内标准差则是一种额外的尺度归一化，严格来说会改变梯度的尺度，但实践中能稳定训练。

</details>

### 1.4 似然比从哪里来？

#### 1.4.1 答案一：把去噪链看成多步MDP

把采样过程离散到时间网格$1 = t_K > t_{K-1} > \dots > t_0 = 0$上，简记$x_k:=x_{t_k}$。DDPO 提出可以把去噪过程看成一个马尔可夫决策过程（MDP）：

- 状态$s_k = (c, t_k, x_k)$，动作$a_k=x_{k-1}$
- 策略$\pi_\theta(a_k\mid s_k) = p_\theta(x_{k-1}\mid x_k, c)$
- 只有最后一步有奖励：$r(x_0, c)$

整条轨迹$\tau = (x_K, \dots, x_0)$的概率可以分解为$p_\theta(\tau\mid c) = p(x_K)\prod_{k=1}^{K} p_\theta(x_{k-1}\mid x_k, c)$，于是策略梯度就变成了：
$$
\nabla_\theta J(\theta) = \mathbb E_{\tau\sim p_\theta}\Big[A(x_0, c)\sum_{k=1}^{K}\nabla_\theta\log p_\theta(x_{k-1}\mid x_k, c)\Big] \label{mdp-pg}
$$
这一步的精妙之处在于"换掉了要算的东西"：终点的边缘似然$\log p_\theta(x_0)$很难算，但轨迹似然可以拆成每一步的转移似然之和，而每一步只要是个简单分布（比如高斯）就很好算。PPO 的比值也相应地变成逐步的比值，即：
$$
r_k(\theta) = \frac{p_\theta(x_{k-1}\mid x_k, c)}{p_{\theta_{\text{old}}}(x_{k-1}\mid x_k, c)}
$$
但对于Flow Model，问题马上就来了：在确定性ODE下，式$\eqref{euler}$意味着$p_\theta(x_{k-1}\mid x_k)$是一个狄拉克$\delta$函数，没有可用的密度，也没有随机性。因此针对Reverse Process的后训练工作，就是**把每一步变成一个有随机性、有闭式密度的高斯转移**，同时尽量不改变模型原本的生成分布。

这个表述还埋下了两个后续工作要处理的问题：

1. **信用分配（Credit Assignment）**：式$\eqref{mdp-pg}$中所有步共享同一个终点优势$A(x_0, c)$，哪一步真正贡献了好结果是分不清的。
2. **采样器绑定**：训练时必须使用这个特定的随机采样器，否则就没有逐步的似然。

<details>

<summary>展开证明：轨迹形式的策略梯度</summary>

奖励只依赖终点$x_0$，而$x_0$的边缘分布由轨迹分布边缘化得到，所以
$$
J(\theta) = \mathbb E_{x_0\sim p_\theta}[r(x_0)] = \mathbb E_{\tau\sim p_\theta}[r(x_0)] \notag
$$
对轨迹分布使用对数导数技巧（见1.3节证明详情）：
$$
\nabla_\theta J = \mathbb E_{\tau\sim p_\theta}\big[r(x_0)\,\nabla_\theta\log p_\theta(\tau)\big] \notag
$$
轨迹似然的对数为
$$
\log p_\theta(\tau) = \log p(x_K) + \sum_{k=1}^{K}\log p_\theta(x_{k-1}\mid x_k, c) \notag
$$
初始噪声分布$p(x_K) = \mathcal N(0, I)$与$\theta$无关，求梯度后消失，于是
$$
\nabla_\theta J = \mathbb E_{\tau}\Big[r(x_0)\sum_{k=1}^{K}\nabla_\theta\log p_\theta(x_{k-1}\mid x_k, c)\Big] \notag
$$
再按1.3节减去基线，$r$换成$A$，即得式$\eqref{mdp-pg}$。

</details>

#### 1.4.2 答案二：用ELBO代替似然

Diffusion Model和Flow Model的训练损失本身就和似然有关：在合适的加权下，单个干净样本$x_0$上的去噪损失等于负的证据下界（ELBO）加一个与参数无关的常数：
$$
\mathcal L_\theta(x_0) = -\operatorname{ELBO}_\theta(x_0) + \text{const},\qquad \operatorname{ELBO}_\theta(x_0) \le \log \pi_\theta(x_0) \label{elbo}
$$
于是可以用ELBO之比代替似然之比：
$$
\hat r(\theta) = \frac{\exp\big(\operatorname{ELBO}_\theta(x_0)\big)}{\exp\big(\operatorname{ELBO}_{\theta_{\text{old}}}(x_0)\big)} = \exp\big(\mathcal L_{\theta_{\text{old}}}(x_0) - \mathcal L_\theta(x_0)\big) \label{fpo-ratio}
$$
实践中，$\mathcal L_\theta(x_0)$用若干组随机抽取的$(t, x_1)$做蒙特卡洛估计，新旧两个模型共用同一组随机数。

Flow Matching的损失小，说明模型“很擅长把噪声推向$x_0$”，也就意味着$x_0$在这个模型下更可能被生成。所以“让高优势样本的Flow Matching损失变小”就近似于“提高高优势样本的似然”。这条路线有几个吸引人的地方：

- **完全不碰采样器**：样本可以由任意ODE/SDE求解器、任意步数生成，训练时只需要终点$x_0$。
- **没有长链信用分配**：整张图被当作一个“动作”，MDP的时间跨度只有1步。
- **和预训练目标一致**：RL阶段优化的仍是Flow Matching损失，只是按优势重新加权。这和LLM里“预训练与RL都在优化对数似然”的对称性是一致的。

代价也很清楚：ELBO只是似然的下界，式$\eqref{fpo-ratio}$里混进了“ELBO与真实似然的差距”这一项，并且只用少量$(t,x_1)$估计时，比值本身也有偏差。

### 1.6 小结

|              | 路线 A：反向过程（MDP）     | 路线 B：前向过程（ELBO）       |
| ------------ | --------------------------- | ------------------------------ |
| “动作”是什么 | 每一步去噪$x_k \to x_{k-1}$ | 整个终点样本$x_0$              |
| 似然从哪来   | 每步高斯转移的闭式密度      | Flow Matching损失（ELBO 代理） |
| 是否精确     | 对所用的SDE采样器是精确的   | 近似（下界+蒙特卡洛估计）      |
| 采样器       | 必须是特定的随机采样器      | 任意黑盒                       |
| 信用分配     | $K$ 步共享一个终点优势      | 单步，无此问题                 |
| 训练目标     | 与预训练目标不同            | 与预训练目标一致               |
