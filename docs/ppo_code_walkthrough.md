# 核心代码讲解 —— 逐类、逐函数吃透 PPO（含与 DQN 的全面对照）

---

## 0. 先建立整体认知：数据是怎么流动的

```
        ┌──────────────── 一次策略更新（update）────────────────┐
        │                                                        │
环境 ──帧──> utils.preprocess ──> 状态 s (4,80,80)                │
                                    │                            │
                                    ▼                            │
                        ActorCritic(s)                           │
                          ├─ actor  ──> π(a|s) ──采样──> a        │  重复 N_STEPS=1024 次
                          └─ critic ──> V(s)                     │  （阶段一：只采集，不训练）
                                    │                            │
                    env.frame_step(a) ──> (r, done, 下一帧)       │
                                    │                            │
                                    ▼                            │
                    RolloutBuffer 存 (s,a,r,done,logπ,V)          │
        │                                                        │
        │  阶段二：GAE(λ) 倒推 → 优势 A、回报 G                    │
        │  阶段三：切小批量 ×4 轮，用【裁剪替代目标】更新            │
        │  阶段四：这 1024 条数据全部作废（on-policy 铁律）          │
        └────────────────────────────────────────────────────────┘
```

**和 DQN 那张图并排看，区别只有三处**：
1. DQN 每步都训练，PPO 攒够 1024 步才训练；
2. DQN 往 5 万容量的池子里存、随机抽，PPO 存在 1024 的临时数组里、用完就扔；
3. DQN 输出 Q 值，PPO 输出「动作概率 + 局面价值」两样东西。

---

## 1. PPO 的六个核心概念（先懂原理，再看代码）

### 1.1 策略梯度：直接学"怎么选动作"

DQN 走的是**间接路线**：先学出每个动作的评分 Q(s,a)，再"选评分最高的"。
PPO 走的是**直接路线**：直接学一个策略 π(a|s)——输入状态，输出每个动作的概率。

目标函数非常朴素：**让"带来高回报的动作"的概率变大**。

```
        J(θ) = E[ log π_θ(a|s) · A(s,a) ]
                └──────┬──────┘   └──┬──┘
                  这个动作的对数概率   这个动作有多好
```

- A > 0（这个动作比平均好）→ 提高它的概率；
- A < 0（这个动作比平均差）→ 降低它的概率。

PyTorch 里"最小化损失"，所以代码里写的是 `-log π · A`（加负号）。

### 1.2 为什么必须有 V(s)？—— 基线的意义

上面的公式里有个 A(s,a)（优势函数），它的定义是：

```
    A(s,a) = Q(s,a) − V(s)
             └─┬──┘   └─┬──┘
          选 a 能得多少  这个局面平均能得多少
```

**为什么不直接用 Q(s,a) 当权重？**
因为"这个局面的平均水平"本身就有高低。假设某个局面怎么走都能拿 100 分，
那里面所有的动作 A 都是正的，全都被"鼓励"——但真正拉开差距的是"比平均好多少"。
减去基线 V(s) 之后，只有**真正优于平均**的动作才会被强化，**梯度的方差大幅降低**，训练稳定得多。

PPO 没有 Q 网络，所以它需要单独训一个 critic 来估 V(s)——这就是 **Actor-Critic** 结构：
- **actor**（演员）：学策略 π，负责选动作；
- **critic**（评论家）：学价值 V，负责当基线、评价演员。

> 一句话总结：critic 存在的唯一意义，就是给 actor 提供一个"参照系"。

### 1.3 GAE(λ)：优势 A 到底怎么算

理论上 A(s,a) = Q − V，但 Q 我们不知道。常用做法是**用实际采样到的回报去估计**。

先定义单步 TD 误差（"这一步的现实和预期的差距"）：

```
    δ_t = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
          └────────── 实际拿到的 ──────────┘   └─ 之前以为能拿的 ─┘
```

GAE（Generalized Advantage Estimation）把未来多步的 δ 按 λ 加权累加：

```
    A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```

代码里写成**从后往前的递推**（只需扫一遍，O(n)）：

```python
gae = delta + gamma * lam * (1 - done) * gae     # gae 此时是 t+1 时刻的 A
```

两个极端帮你理解 λ：

| λ | 含义 | 特点 |
|---|---|---|
| λ = 0 | A_t = δ_t，只用一步信息 | **偏差大**（太依赖 V 的准确度）、方差小 |
| λ = 1 | A_t = 完整蒙特卡洛回报 | 偏差小、**方差大**（一条轨迹的运气影响太大） |

λ = 0.95 是论文推荐值，实践中几乎不用调。

最后，critic 的回归目标（回报 G）就是：

```
    G_t = A_t + V(s_t)      # 优势 + 基线 = 真实回报
```

### 1.4 PPO 的灵魂：裁剪替代目标

**问题**：策略梯度有个致命弱点——**一次更新步子太大会直接把策略推垮，而且不可逆**。
（DQN 也有类似问题，靠"目标网络 + 回放池"缓解；PPO 用的是完全不同的思路。）

**PPO 的解法**：用"新旧策略的概率比"限制更新幅度。

```
    r_t(θ) = π_θ(a|s) / π_θ_old(a|s)          ← 概率比，衡量策略变了多少

    L^CLIP(θ) = E[ min( r_t(θ)·A_t , clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]
```

**为什么是 min？** 分两种情况看（我们要**最大化**这个目标，所以 min 是"保守的下界"）：

| 情况 | 不被裁剪时 | 裁剪的效果 |
|---|---|---|
| A > 0（好动作） | r 越大目标越大 → 想把概率无限拉高 | r 被封顶在 **1+ε**，不许一次拉爆 |
| A < 0（坏动作） | r 越小目标越大 → 想把概率压到 0 | r 被封底在 **1−ε**，不许一次压死 |

ε = 0.2 意味着**每次更新，单个动作的概率最多变化 20%**。
这就是 "Proximal"（近端）的含义：**新策略不许离旧策略太远**。

**还有一个对称的裁剪：价值函数裁剪（`CLIP_VLOSS`）**

策略那边限制"概率别跳太远"，critic 这边也该限制"V(s) 别跳太远"——
否则 4 个 epoch 反复更新下来，V 会被一路拽偏，而 V 一偏，优势 A 就全错了。

```python
v_loss_unclipped = (values - returns) ** 2
v_clipped = old_values + torch.clamp(values - old_values, -ε, ε)   # 新V相对旧V的偏移也裁到 ±ε
v_loss = torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
```

两个关键点：
1. `old_values` 必须是**采集时冻结**的 V，不能是每轮重算的（否则"锚点"自己会动，裁剪就没意义了）；
2. 取 `max` 而不是 `min`：只在"裁剪后更糟"时才惩罚，从而阻止 V 往偏离更远的方向跑。

> ⚠️ 一个容易踩的坑：CleanRL 写 `v_loss = 0.5 * MSE` 再乘 `vf_coef=0.5`，实际权重是 **0.25**；
> 本项目写 `MSE * 0.5`，实际权重是 **0.5**。后者严格对应论文 `L^VF = c1·(V−V_targ)², c1=0.5`。
> 两种都能跑，但你要知道自己用的是哪一套。

代码对应：

```python
ratio = torch.exp(logprobs - old_logprobs)              # r(θ)，用对数相减再取指数防溢出
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - ε, 1 + ε) * advantages
policy_loss = -torch.min(surr1, surr2).mean()           # 加负号：PyTorch 只做最小化
```

### 1.5 on-policy：为什么数据用完就扔

概率比 r(θ) = π_新 / π_旧 这个式子，只有在**"数据确实是 π_旧 采出来的"**时才成立。
一旦策略更新过，手上的数据就是"旧旧策略"采的，用它算出来的比值在数学上就是错的。

所以 PPO 必须：采一批 → 更新 → **全部丢弃** → 用新策略重新采。

代价：**样本效率远低于 DQN**（DQN 一条经验能用几百次）。
收益：训练极其稳定，几乎不用调参。

一句话对比：
> DQN = 省数据，但训练不稳；PPO = 费数据，但训练稳如老狗。

### 1.6 熵奖励：防止"一根筋"

策略的熵：H(π) = −Σ π(a)·log π(a)。

- 熵大 = 策略还很随机，愿意尝试；
- 熵小 = 策略很确定，几乎只选一个动作。

本游戏只有 2 个动作，策略**极易**在训练早期就塌缩成"永远不拍翅"（然后小鸟直接坠地）。
更糟的是塌缩后：`log π ≡ 0` → 概率比恒为 1 → 优势再大梯度也是 0 → **永远救不回来**。

所以 PPO 在损失里加一项熵奖励，**奖励高熵**：

```python
entropy_loss = -entropy.mean()      # 最大化熵 ⇔ 最小化 -熵
loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss
```

`ENT_COEF=0.01` 不大，但足以在早期把策略"托住"。
实测：本项目训练到 8 万步时熵稳定在 **0.24** 左右不再下降，正是它的功劳。

---

## 2. `ppo/config.py` —— 哪些超参是 PPO 独有的

| 超参 | 值 | 含义 / 为什么这么取 |
|---|---|---|
| `N_STEPS` | 1024 | 每次更新前采集多少步。**PPO 独有** |
| `N_EPOCHS` | 4 | 这批数据重复训练几轮。**PPO 独有** |
| `GAE_LAMBDA` | 0.95 | GAE 的 λ，论文推荐，几乎不用调。**PPO 独有** |
| `CLIP_RANGE` | 0.2 | 裁剪系数 ε，PPO 的灵魂。**PPO 独有** |
| `ENT_COEF` | 0.01 | 熵奖励系数，防塌缩。**PPO 独有** |
| `VF_COEF` | 0.5 | 价值损失权重。**PPO 独有** |
| `MAX_GRAD_NORM` | 0.5 | 梯度裁剪。**PPO 独有** |
| `LR` | 2.5e-4 | 比 DQN 的 1e-4 大（有裁剪保护，不怕训崩） |
| `GAMMA` | 0.99 | 与 DQN 相同 |
| `MAX_STEPS` | 2,000,000 | 与 DQN 相同 → 曲线可以在同一横轴上比 |

**没有** `OBSERVE / EXPLORE / INIT_EPSILON / FINAL_EPSILON / TARGET_SYNC / REPLAY_SIZE`
—— 这些是 DQN 专属的，PPO 一项都不需要。这个"消失的清单"本身就是最好的对照材料。

---

## 3. `ppo/utils.py` —— 与 DQN 版逐行相同

刻意复制了一份而不是 import 上级目录：
① 你要求两个项目代码分离；
② 更重要的是两份一模一样才能保证"输入分布"这个变量被控制住，对比才可信。

---

## 4. `ppo/model.py` —— ActorCritic 类

### 4.1 一个主干、两个头（对照 DQN 的一个头）

```
                        ┌─ actor  (512 → 2)  → 动作 logits
卷积主干(与DQN相同) → 512维特征 ─┤
                        └─ critic (512 → 1)  → V(s) 标量
```

卷积主干（`features` + `shared`）与 DQN 的 `QNet` **完全一致**——刻意如此，
这样"网络容量"这个变量就被控制住了，差异只来自算法本身。

### 4.2 正交初始化与 `gain=0.01`（面试常问）

```python
if m is self.actor:   gain = 0.01        # 策略头
elif m is self.critic: gain = 1.0        # 价值头
else:                 gain = sqrt(2)     # 卷积/共享层（跟随 ReLU）
nn.init.orthogonal_(m.weight, gain=gain)
```

- **正交初始化**：保持信号方差不随层数爆炸/消失，是 PPO 论文和 SB3 的默认做法。
- **actor 的 gain=0.01**：让初始 logits 极小 → softmax 后概率**接近均匀 0.5/0.5**。
  实测本项目初始概率正是 `[0.500, 0.500]`（标准差 0.0005），探索最充分。
  若初始就"很自信"，会过早收敛到次优策略。
- **critic 的 gain=1.0**：价值需要正常尺度的输出，不需要压小。

### 4.3 `forward()` 里的 `x = x / 255.0`

这一个除法是**必需的**，去掉它训练会在第 2 次更新直接崩掉。完整推导见 `ppo/model.py` 注释，
这里给结论：

| | 不除以 255 | 除以 255 |
|---|---|---|
| 初始 V(s) | ~168（真实回报只有 1~40） | ~0.1 |
| 价值 loss | 10983 | 0.003 ~ 0.2 |
| 熵（第2次更新） | **0.000（永久塌缩）** | 0.68 |
| 解释方差 | −2.4 | 0.99 |

**为什么 DQN 不需要？** DQN 的靶子 `r + γ·max Q'` 里 Q 和 Q' 出自同一网络，量级自动抵消；
而 PPO 的 V(s) 是**基线**，基线一旦和回报不同量级，优势 A 就彻底失真。
→ 这正体现了 **PPO 比 DQN 对尺度和初始化更敏感**。

### 4.4 两个方法，分饰两角

| 方法 | 用在哪 | 干什么 |
|---|---|---|
| `get_action_and_value(x)` | **采集阶段** | 按概率**采样**一个动作，返回 `(a, logπ, H, V)` |
| `evaluate_actions(x, actions)` | **更新阶段** | 用**当前**策略重算旧动作的 `logπ`，返回 `(logπ, H, V)` |

为什么要两个？因为 PPO 要把一批数据训练 `N_EPOCHS=4` 轮，每轮更新后策略都变了，
同一批 (s,a) 的概率也变了，必须**每轮重新算** `log π_新`，才能和采集时存的 `log π_旧` 相比。

```python
dist = Categorical(logits=logits)     # logits → 可采样的类别分布
action = dist.sample()                # 采样（这就是 PPO 的探索机制）
log_prob = dist.log_prob(action)      # 存下来，后面算概率比要用
```

---

## 5. `ppo/rollout.py` —— RolloutBuffer（对照 DQN 的 dqn/replay.py）

### 5.1 两个文件放一起看，差异即本质

| | DQN `ReplayBuffer` | PPO `RolloutBuffer` |
|---|---|---|
| 容量 | 50000 | 1024（= N_STEPS） |
| 寿命 | 长期保存 | 更新一次后全部作废 |
| 抽样 | 随机乱序 | 按时间存，切小批量时才打乱 |
| 存的量 | `(s, a, r, s', done)` | `(s, a, r, done, logπ_旧, V_旧)` + 算出的 `(A, G)` |
| 策略类型 | off-policy | **on-policy** |

### 5.2 为什么必须多存 `logπ_旧` 和 `V_旧`

- `logπ_旧(a|s)`：算概率比 `r(θ) = exp(logπ_新 − logπ_旧)` 的分母。**DQN 完全不需要这个**。
- `V_旧(s)`：算 GAE 需要**采集当时**的价值估计。

⚠️ 注意是"采集当时"的值：如果更新了几轮之后再回头用新网络的 V 去算 GAE，
那 A 和 G 就会随着更新而变，等于"边训练边改目标"，训练必崩。
所以**必须在更新开始前就把 A 和 G 全部算完并冻结**——这就是主循环里
"先 `compute_returns_and_advantages`，再 `update_policy`"顺序不能颠倒的原因。

### 5.3 GAE 的递推实现

```python
gae = 0.0
for t in reversed(range(self.n_steps)):
    if t == self.n_steps - 1:
        next_non_terminal = 1.0 - self.dones[t]
        next_value = last_value          # ← rollout 之外，用 last_value 兜底
    else:
        next_non_terminal = 1.0 - self.dones[t]
        next_value = self.values[t + 1]
    delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
    gae = delta + gamma * lam * next_non_terminal * gae
    self.advantages[t] = gae
self.returns = self.advantages + self.values
```

三个易错点：

1. **`last_value` 必须传**：最后一步没有"下一步"可算，只能单独跑一次网络
   求 `V(s_最后)` 来 bootstrap。漏了它，最后几步的优势会被系统性低估。
2. **`(1 − done)` 屏蔽跨局**：这一步如果撞死了，"下一状态"其实是新一局的第一帧，
   它的价值与本局无关，必须乘 0 屏蔽掉。
3. **必须倒着扫**：`gae` 依赖 `t+1` 时刻的值。

验证用例（`gamma=1, lam=1`，每步 r=1，第 3 步死亡，V 全 0）：
得到 `A = [3, 2, 1, 2, 1]`。
解读：死亡那步 A=1（只剩自己的即时奖励），它之前的 A₀=3（累积了 3 步），
死亡之后的 A₃=2、A₄=1 属于**新的一局**重新开始累积。链路被正确切断。

### 5.4 优势标准化

```python
self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
```

优势的尺度会随训练剧烈变化（不同 rollout 之间能差几十倍），
不标准化等于学习率在乱跳。这是 PPO 的默认操作（`NORMALIZE_ADV=True`）。

**在哪一步标准化，有两种做法**（`config.NORM_ADV_PER_MINIBATCH` 控制）：

| 做法 | 位置 | 谁在用 |
|---|---|---|
| `False`：整批统一 | 算完 GAE 后，对 1024 条一次性标准化 | 早期 baselines |
| `True`：每个小批量内 | 每个 minibatch 取出后各自标准化 | **CleanRL / stable-baselines3（本项目默认）** |

两者都能跑。差别在于：`True` 是"局部自适应"——每个 minibatch 都被拉回均值 0、标准差 1，
即使某个 minibatch 恰好全是高优势样本也不会失控。工业界默认用 `True`。

---

## 6. `ppo/train.py` —— 四阶段主循环

### 6.1 四个阶段（顺序绝对不能乱）

```
阶段一 collect_rollout()         用当前策略采 1024 步（不训练）
阶段二 compute_returns_and_advantages()   算 A 和 G（此时策略还没更新）
阶段三 update_policy()           切小批量，用裁剪目标更新 4 轮
阶段四 （隐式）下一轮 buffer.reset()，旧数据全部作废
```

### 6.2 `collect_rollout()` —— 与 DQN 的一处重要差别

```python
nxt = init_state(frame) if terminal else next_state(state, frame)
```

撞死后环境会自动重开一局，此时必须把帧栈**重置成"4 张新局首帧"**，
不能让上一局的残影顺着滑动窗口滑进新局。

- **DQN 无所谓**：它只把 s' 存进池子，下一局照样从 s' 开始，最多是一两条"跨局"的脏数据，
  5 万条里占极小比例。
- **PPO 必须处理**：它要靠 V(s) 当 GAE 的基线，跨局的脏状态会污染价值估计，
  而价值一歪，优势就全歪。

### 6.3 `update_policy()` —— 三行公式

```python
ratio = torch.exp(logprobs - old_logprobs)                        # ① 概率比

# ② 裁剪替代目标（可选：先在小批量内把 A 标准化到均值0、标准差1）
pg_loss = -torch.min(ratio * adv,
                     torch.clamp(ratio, 1-ε, 1+ε) * adv).mean()

# ③ 价值回归（可选：加价值裁剪，限制 V 别跳超过 ε）
v_clipped = old_values + torch.clamp(values - old_values, -ε, ε)
v_loss = torch.max((values - returns)**2, (v_clipped - returns)**2).mean()

entropy_loss = -entropy.mean()                                    # ④ 熵奖励（最大化熵）
loss = pg_loss + VF_COEF * v_loss + ENT_COEF * entropy_loss
```

- `torch.exp(logπ_新 − logπ_旧)`：用对数相减再取指数，避免直接做除法时上溢/下溢。
- `nn.utils.clip_grad_norm_(net.parameters(), 0.5)`：**梯度裁剪**，
  与"目标裁剪"是两回事——它限制的是梯度向量的模长，防止单次更新步子太大。
- **KL 早停**（`TARGET_KL`，默认关闭）：每轮 epoch 后检查 `approx_kl`，
  超过阈值就提前收手。这是 TRPO"信任域"思想的残留——PPO 有了 clip 兜底后一般不需要，
  但调大学习率或数据分布突变时它是个保险丝。

### 6.4 学习率线性衰减

```python
lr = linear_schedule(cfg.LR, 0.0, t / max_steps)
```

PPO 标配。后期学习率趋近 0，策略能稳定收敛而不是在最优解附近来回抖。
衰减只依赖"当前步数"，所以**续训时自动接上**，不用额外存状态。

### 6.5 诊断指标怎么读（这是 PPO 相比 DQN 最大的工程优势）

| 指标 | 健康范围 | 异常说明 |
|---|---|---|
| `train/explained_variance` | 从 0 → **0.9+** | 一直 ≤0 说明 critic 没学会，策略也别想好 |
| `train/entropy` | 缓慢下降，**别到 0** | 骤降到 0 = 策略塌缩，基本不可逆 |
| `train/approx_kl` | 0.001 ~ 0.05 | >0.1 说明更新过猛，考虑降低 LR |
| `train/clip_fraction` | 0 ~ 0.2 | 长期 >0.5 说明数据在"喊"要大改但被 ε 按住 |
| `loss/value` | 缓慢下降 | 突然飙升 = 价值崩了（常见于输入尺度问题） |
| `rollout/flap_ratio` | 0.1 ~ 0.3 | →0 说明"永远不拍翅"，→1 说明"疯狂拍翅"，都是塌缩 |

实测（本项目 RTX 3050，8 万步）：
解释方差 0.99、熵从 0.69 稳定降到 0.24、KL ~0.001、拍翅率 0.15 —— 全部健康。

---

## 7. `ppo/play.py` —— 与 DQN 版只差一行

```python
# DQN：
action = int(net(s).argmax(dim=1).item())        # Q 值最大的动作
# PPO：
logits, _value = net(s)
action = int(logits.argmax(dim=1).item())        # 概率最大的动作
```

为什么评估时不采样？训练时采样是为了探索；评估时我们要的是**确定性最优表现**
（deterministic evaluation）。命令行参数与 DQN 版完全一致：
`--episodes N`、`--no-render`。

---

## 8. DQN vs PPO 全面对照（本文重点）

### 8.1 算法层面

| 维度 | DQN | PPO |
|---|---|---|
| 学什么 | Q(s,a)：动作评分 | π(a\|s) + V(s)：策略 + 价值 |
| 策略从哪来 | 由 Q 值**间接**导出（argmax） | **直接**参数化一个策略网络 |
| 动作空间 | 天然只支持**离散** | 离散、连续都支持（本任务是离散） |
| 探索方式 | ε-贪心（外部强加的随机） | 按概率**采样**（内建于策略） |
| 更新目标 | 回归到 `r + γ·max Q'` | 最大化**裁剪替代目标** |
| 稳定手段 | 目标网络 + 经验回放 | 裁剪 ε + 熵奖励 + 梯度裁剪 |
| 数据策略 | **off-policy**（老数据反复用） | **on-policy**（用完就扔） |
| 网络数量 | 2（策略网 + 目标网） | 1（actor/critic 共享主干） |
| 样本效率 | **高** | 低 |
| 调参难度 | **高**（易 Q 爆炸、易塌缩） | 低（开箱即用） |
| 典型失效模式 | Q 值发散、策略塌缩 | 熵过早归零、critic 尺度失控 |

### 8.2 代码层面（逐文件）

| 文件 | DQN | PPO |
|---|---|---|
| `ppo/config.py` | `OBSERVE/EXPLORE/ε/TARGET_SYNC/REPLAY_SIZE` | `N_STEPS/N_EPOCHS/GAE_LAMBDA/CLIP_RANGE/ENT_COEF/VF_COEF` |
| `ppo/model.py` | `QNet`：1 个头输出 Q | `ActorCritic`：2 个头输出 logits + V |
| `dqn/replay.py` / `ppo/rollout.py` | `deque(maxlen=50000)` + `random.sample` | 预分配 1024 数组 + 顺序写 + 小批量打乱 |
| 训练触发条件 | `t > OBSERVE and len(memory) >= 32` | 攒满 `N_STEPS` 就更新 |
| 一次更新用多少数据 | `BATCH_SIZE=32` | `1024 × 4 轮` |
| 需要存 log π 吗 | 不需要 | **必须**（算概率比） |
| 死亡后状态处理 | 直接 `state = nxt` | 必须 `init_state(frame)` 重置帧栈 |
| 输入归一化 | 不需要 | **必须 `/255`** |
| 裁剪对象 | 无（靠目标网络） | **策略裁剪** + **价值裁剪**（双保险） |
| 优势标准化 | 无此概念 | 每个小批量内标准化（CleanRL/SB3 默认） |
| 早停机制 | 无 | `target_kl` 触发时提前结束剩余 epoch（可选） |

### 8.3 关键结论

> **DQN 用"空间"换"稳定"**：存 5 万条老数据，用随机抽样打破相关性、用目标网络稳住靶子。
> 代价是内存和 off-policy 带来的训练不稳定。
>
> **PPO 用"约束"换"稳定"**：不存老数据，但用数学手段（裁剪）把每次更新的幅度锁死。
> 代价是样本效率，收益是训练极其稳定、超参鲁棒。

---

## 9. 调参与排错

| 现象 | 可能原因 / 处理 |
|---|---|
| **熵突然归零、拍翅率变 0 或 1** | 策略塌缩。加大 `ENT_COEF`（0.01 → 0.05）；检查是否忘了 `/255` |
| 价值 loss 上万 | 几乎一定是**输入没归一化**，看 `ppo/model.py` 4.3 节 |
| 解释方差长期 ≤ 0 | critic 没学会：加大 `VF_COEF`、或降低 `LR` 让训练更稳 |
| 训练很稳但就是不涨分 | `N_STEPS` 太小 → 改成 2048；或 `N_EPOCHS` 太大导致过拟合 → 改成 2~3 |
| approx_kl 经常 > 0.1 | 更新过猛：降低 `LR`、或减小 `N_EPOCHS` |
| 显存不够 | 调小 `N_STEPS` 或 `BATCH_SIZE` |
| 想更快出结果 | 把 `MAX_STEPS` 调小（学习率衰减会自动跟着变快） |
