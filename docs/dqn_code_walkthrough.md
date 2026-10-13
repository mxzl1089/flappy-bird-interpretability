# 核心代码讲解 —— 逐类、逐函数吃透 DQN

> 配套代码在同级目录。建议**先看本文建立原理框架，再对照代码**，最后合上代码能自己复述。

---

## 0. 先建立整体认知：数据是怎么流动的

```
  ┌──────────────────────────────────────────────────────────────┐
  │  环境 game/  (pygame 跑 Flappy Bird)                          │
  │  frame_step(动作) ──► 画面 288x512x3 、奖励 r 、是否死亡 done  │
  └───────────────────────────┬──────────────────────────────────┘
                              │
                    dqn/utils.py  预处理
                   缩到 80x80 → 转灰度 → 二值化 (0/255)
                              │
                    dqn/utils.py  帧堆叠
                   最近 4 帧叠起来 → 状态 s，形状 (4, 80, 80)
                              │
                              ▼
                    dqn/model.py   QNet 卷积网络
                    s ──► [Q(s,不拍), Q(s,拍)]   两个动作的"打分"
                              │
                    dqn/train.py   ε-贪心：ε 概率乱选，否则选分高的
                              │
                              ▼  得到动作 a
                   环境执行 a，拿到 (新画面, r, done)
                              │
                    dqn/replay.py  把 (s, a, r, s', done) 存进回放池
                              │
                    dqn/train.py   train_step()：
                    随机抽 32 条，让 Q(s,a) 逼近  r + γ·max Q_target(s',·)
                              │
                   每 10000 步：把目标网络的权重同步成策略网络
```

**一句话概括每步**：看画面 → 打分 → 选动作 → 拿奖励 → 存经验 → 回头修正打分。

---

## 1. DQN 的五个核心概念（先懂原理，再看代码）

### 1.1 Q 值与贝尔曼方程（最重要）

**Q(s, a) 是什么？**
> "在状态 s 下做动作 a，之后一直按最优策略走，预计能拿到的**累计折扣奖励**"。

有了 Q，决策就变得极其简单：**选 Q 值最大的那个动作**。
所以强化学习的难题从"学策略"变成了"学一个打分函数 Q"。

**那 Q 该怎么学？用贝尔曼方程（递归定义）：**

$$Q(s,a) \;=\; r + \gamma \cdot \max_{a'} Q(s', a')$$

> "在 s 做 a 的价值" = "立刻拿到的奖励 r" + "打折后的下一步最好能拿多少"。

这就是**监督学习的标签**！我们把神经网络估的 `Q(s,a)` 往 `r + γ·max Q(s',·)` 上回归（MSE），
反复迭代，Q 就会收敛到真实值。这就是 `train_step()` 干的事。

| 符号 | 代码里对应 | 含义 |
|---|---|---|
| `Q(s,a)` | `policy_net(states).gather(1, actions)` | 网络当前对"做过的那个动作"的打分 |
| `r` | `rewards` | 这一步环境给的奖励 |
| `γ` | `cfg.GAMMA = 0.99` | 折扣因子，越大越看长远 |
| `max Q(s',·)` | `target_net(next_states).max(dim=1)[0]` | 用**目标网络**算下一步的最好分数 |
| `(1 - dones)` | 同左 | 死了就没有"下一步"，未来项清零 |

### 1.2 ε-贪心（探索 vs 利用）

- **以概率 ε 随机选动作** → 探索，去试没试过的操作；
- **否则选 Q 最大的动作** → 利用，用当前学到的最优策略。

为什么不一直选最优？因为一开始网络是随机的，它的"最优"是瞎猜的，
不探索就永远发现不了"原来拍翅能过管"。

**ε 要衰减**：初期 ε 大（多探索），后期 ε 小（多利用）。
本项目：`0.1 → 0.01`，在 100 万步内**线性衰减**（`dqn/train.py` 主循环里那行）。

> 为什么初始值用 0.1 而不是 1.0？
> ε=1.0 是纯随机，一开始会灌进大量"贴地暴毙"的垃圾样本，反而拖慢收敛。0.1 更稳。

### 1.3 经验回放（为什么必须有）

两个理由，**面试两个都要讲**：

1. **打破样本相关性**：游戏里连续两帧几乎一样，样本强相关。
   拿强相关样本按顺序做梯度下降，网络会"记住最近这一段"，训练剧烈震荡甚至发散。
   回放池存起来**随机抽样**，把相关性打散。
2. **重复利用数据**：一条"成功穿管"的珍贵经验可以被反复抽到，样本利用率远高于用一次就扔。

实现上就是个**定长循环队列**（`deque(maxlen=...)`），满了自动丢最旧的。

### 1.4 目标网络（原版没有，本版新增）

**问题**：如果目标 `r + γ·max Q(s',·)` 里的 Q 用的就是**正在更新的那个网络**，
那么目标会"追着自己跑"——你刚把它抬高，下一轮它又成了新目标，形成正反馈 → **Q 值爆炸、训练发散**。

**解法**：拷贝一份网络当"靶子"（目标网络），**冻结住不训练**，
每隔 N 步才把策略网络的权重硬同步过去。这样目标在 N 步内是**固定的**，训练就稳了。

> ⚠️ **原 TF 版没有目标网络**！它用同一个网络算目标，只能靠 `lr=1e-6`（极小学习率）
> 硬扛住不发散——代价就是收敛极慢。本版加了目标网络后，学习率可以用 `1e-4`，快很多。
> 这是本移植版最重要的改进，也是很好的面试谈资。

### 1.5 帧堆叠（为什么要叠 4 帧）

单张静态图**看不出小鸟在上升还是下落**。把连续 4 帧叠成一个状态（4 个通道），
网络才能推断出**运动方向和速度**——这对"什么时候该拍翅"是决定性信息。
这是所有像素类 DQN（Atari、Flappy Bird）的标准操作。

---

## 2. `dqn/config.py` —— Config 类

**为什么单独抽出来？** 训练和评估共用一套配置，不会出现"训练 4 帧、评估 1 帧"的不一致；调参只改一处。

| 超参 | 值 | 一句话解释 |
|---|---|---|
| `NUM_ACTIONS` | 2 | 0=不拍翅，1=拍翅 |
| `FRAME_SIZE` | 80 | 预处理后每帧边长 |
| `STACK_LEN` | 4 | 帧堆叠长度（见 1.5） |
| `GAMMA` | 0.99 | 折扣因子 |
| `LR` | 1e-4 | 学习率（原版 1e-6，本版有目标网络所以能调大） |
| `BATCH_SIZE` | 32 | 每次更新抽多少条 |
| `REPLAY_SIZE` | 50000 | 回放池容量 |
| `OBSERVE` | 100000 | **观察期**步数：只采集不训练 |
| `EXPLORE` | 1000000 | **探索期**步数：ε 在这段内线性衰减 |
| `INIT/FINAL_EPSILON` | 0.1 / 0.01 | ε 起止值 |
| `TARGET_SYNC` | 10000 | 目标网络同步间隔 |
| `MAX_STEPS` | 2000000 | 总步数上限（原版是死循环） |
| `SAVE_EVERY` | 10000 | 每多少步存一个**步数编号**模型（不覆盖） |
| `LOG_EVERY` | 1000 | 每多少步：打印控制台 + 写一次 TensorBoard |
| `USE_TB` | True | 是否写 TensorBoard |

**三个训练阶段**（控制台会打印，务必知道自己在哪）：

| 阶段 | 步数范围 | 行为 |
|---|---|---|
| 观察期 | `t ≤ OBSERVE` | **只采集经验、不训练**，先把池子填满 |
| 探索期 | `OBSERVE < t ≤ OBSERVE+EXPLORE` | 训练 + ε 从 0.1 线性降到 0.01 |
| 训练期 | 之后 | ε 固定 0.01，纯微调 |

> 为什么要观察期？一开始网络是随机权重，池子里只有几十条时训练 = **对着噪声过拟合**。
> 先攒满 10 万条再开始学，稳得多。

`device()` 是一个 `@staticmethod`，自动选 CUDA/CPU——你机器上会打印 `cuda`。

---

## 3. `dqn/utils.py` —— 3 个预处理函数

| 函数 | 输入 → 输出 | 作用 |
|---|---|---|
| `preprocess_frame(frame)` | 288×512×3 → 80×80 | 缩放 + 灰度 + 二值化 |
| `init_state(frame)` | 一帧 → (4,80,80) | 开局时复制 4 份凑够通道 |
| `next_state(state, frame)` | (4,80,80)+新帧 → (4,80,80) | 滑动窗口：新帧进、旧帧出 |

**为什么预处理要这么做？**
- 尺寸太大算不动；颜色对这个任务没用（判断管子位置只需要**轮廓**）；
- 二值化（>1 变 255，其余变 0）能去掉抗锯齿的灰色边缘和背景噪点，只留黑白分明的轮廓，**网络更好学**。

**滑动窗口的可视化**（数字代表帧的新旧，0=最新）：

```
旧状态 [0, 1, 2, 3]  ──►  新状态 [新, 0, 1, 2]     （3 被挤掉）
```

> 原版用 `cv2.resize` + `cv2.cvtColor`；本版改用 **PIL**（`Image.resize` + `convert("L")`），
> 效果等价，但你环境里没装 cv2，这样就不用额外装包了。

---

## 4. `dqn/model.py` —— QNet 类（卷积神经网络）

### 4.1 类的两个必需方法

```python
class QNet(nn.Module):          # 必须继承 nn.Module
    def __init__(self, ...):
        super().__init__()      # 必须调用！它会登记子模块，否则优化器找不到参数
        # 在这里定义层
    def forward(self, x):
        # 在这里写前向传播；反向传播由 autograd 自动生成，不用你写
```

### 4.2 结构一览（输入 4×80×80）

| 层 | 参数 | 输出尺寸 | 说明 |
|---|---|---|---|
| Conv2d | 4→32, k=8, s=4, **p=2** | 32 @ 20×20 | 大卷积核大步长，快速降采样 |
| ReLU + MaxPool | 2×2, s=2 | 32 @ 10×10 | |
| Conv2d | 32→64, k=4, s=2, **p=1** | 64 @ 5×5 | |
| Conv2d | 64→64, k=3, s=1, **p=1** | 64 @ 5×5 | 保持尺寸，加深非线性 |
| Flatten | — | 1600 | 64×5×5 |
| Linear | 1600→512 + ReLU | 512 | |
| Linear | 512→2 | **2** | 两个动作的 Q 值 |

**输出层为什么不加 softmax？**
因为 Q 值是**回归目标**（可能等于 12.7 这种任意实数），不是概率。加 softmax 会把它压成 0~1，完全错误。

### 4.3 padding 是怎么算出来的（重点，面试会问）

TensorFlow 的 `padding='SAME'` 会**自动补零**保证 `输出 = ceil(输入/步长)`；
**PyTorch 没有 'SAME'，必须自己算出 padding 填进去**：

```
输出尺寸公式：  out = (in + 2*p - k) / s + 1

conv1: (80 + 2p - 8)/4 + 1 = 20  →  (72+2p)/4 = 19  →  2p = 4   →  p = 2
conv2: (10 + 2p - 4)/2 + 1 = 5   →  ( 6+2p)/2 =  4  →  2p = 2   →  p = 1
conv3: ( 5 + 2p - 3)/1 + 1 = 5   →   2+2p     =  4  →  2p = 2   →  p = 1
```

> ✅ **前向传播能跑通本身就证明了 padding 算对了**——
> 因为 `nn.Linear(1600, 512)` 要求展平后**正好是 1600**，
> 只要有一个 padding 算错，尺寸就对不上，PyTorch 会直接报错。

---

## 5. `dqn/replay.py` —— ReplayBuffer 类

| 方法 | 作用 |
|---|---|
| `__init__(capacity)` | `deque(maxlen=capacity)`，一行实现 FIFO 自动淘汰 |
| `push(s,a,r,s',done)` | 存一条五元组经验 |
| `sample(batch_size, device)` | 随机抽一批，**转成可直接训练的 tensor** |
| `__len__()` | 让 `len(memory)` 可用，训练门槛就靠它判断 |

**`sample()` 返回的 5 个 tensor（形状为什么要这样设计）：**

| 返回 | 形状 | 为什么 |
|---|---|---|
| `states` | (B,4,80,80) | 直接进卷积 |
| `actions` | (B,**1**) int64 | `gather` 要求索引是 **long 型**且**二维** |
| `rewards` | (B,) | |
| `next_states` | (B,4,80,80) | |
| `dones` | (B,) 1.0/0.0 | 用 `(1-dones)` 相乘就能把"死了的未来奖励"清零 |

> ⚠️ **性能坑（已在代码里修掉）**：
> 不能把"一堆小 numpy 数组"直接扔给 `torch.as_tensor`——PyTorch 会警告 `extremely slow`（只能逐个拷贝）。
> 因为**每一步训练都要采样一次**，这里慢会拖垮整体速度。
> 正确做法：先 `np.asarray(...)` 拼成一整块连续内存，再一次性转 tensor。

---

## 6. `dqn/train.py` —— 训练主循环（核心中的核心）

### 6.1 四个函数各司其职（这就是"非扁平结构"的好处）

| 函数 | 职责 |
|---|---|
| `make_env(no_render)` | 创建游戏环境（含一个延迟 import 的坑，见下） |
| `select_action(...)` | ε-贪心选动作 |
| `train_step(...)` | **DQN 的心脏**：抽一批做一步梯度下降 |
| `save_ckpt(...)` | 存档（权重 + 优化器状态 + 步数） |
| `main()` | 主循环：把上面串起来 |

### 6.2 `train_step()` —— 三行对应贝尔曼方程

```python
# ① 网络对"实际执行过的动作"的打分  Q(s,a)
q_values = policy_net(states).gather(1, actions).squeeze(1)

# ② 目标  r + γ·max_a' Q_target(s',a')   —— 用目标网络算，且不回传梯度
with torch.no_grad():
    next_q = target_net(next_states).max(dim=1)[0]
    targets = rewards + (1.0 - dones) * cfg.GAMMA * next_q

# ③ 回归
loss = F.mse_loss(q_values, targets)
optimizer.zero_grad()   # 必须手动清零！PyTorch 梯度默认【累加】
loss.backward()
optimizer.step()
```

三个容易讲不清的点，**务必记牢**：

1. **`gather(1, actions)` 在干什么？**
   `policy_net(states)` 输出 (B, 2)，是**每个动作**的 Q 值。
   但我们只关心**实际执行过的那一个动作**的 Q 值，`gather` 就是按 `actions` 把那一列取出来。

2. **为什么要用 `torch.no_grad()` 包住目标？**
   目标是"标签"，应该是**固定值**。如果不 `no_grad`，反向传播会顺着目标网络一路传回去，
   目标就"自己追自己"了（正是 1.4 讲的问题）。

3. **为什么 `optimizer.zero_grad()` 必须写？**
   PyTorch 的梯度是**累加**的（为了支持多卡/大 batch 累积）。
   不清零，第 2 步的梯度会叠在第 1 步上，越积越大直接训崩。这是 TF 转 PyTorch 最常见的 bug。

### 6.3 `make_env()` 里的延迟 import（一个真实的坑）

```python
def make_env(no_render):
    if no_render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # 必须在 import 前设置！
    sys.path.append(Config.GAME_DIR)
    import wrapped_flappy_bird as game_env                  # 这行会触发 pygame.init()
    return game_env.GameState()
```

**为什么不能在文件顶部 import？**
因为 `wrapped_flappy_bird` 模块在 **import 的瞬间**就会执行 `pygame.init()`，
而 `SDL_VIDEODRIVER` 这个环境变量必须在 `pygame.init()` **之前**设置才生效。
所以必须等参数解析完、设好环境变量之后再 import。

### 6.4 主循环的执行顺序

```
① ε-贪心选动作 a
② env.frame_step(a) → (画面, 奖励 r, 死亡 done)
③ 预处理堆叠成 s'；把 (s,a,r,s',done) 存进回放池
④ 过了观察期 且 池子够一个 batch → train_step() 做一步梯度下降
⑤ 每 TARGET_SYNC 步 → 目标网络硬同步
⑥ ε 按步数反推（`get_epsilon(t)`，续训自动接上，见下方说明）
⑦ 定期打印日志 / 存档
```

> **续训时会发生什么？（面试也常问）**
>
> - **步数**从 `latest.json` 恢复，所以阶段（观察/探索/训练）直接按步数接着走，**不会重来**；
> - **ε 由步数反推**（`get_epsilon`），自动接上。**不能**把 ε 重置成 `INIT_EPSILON`——
>   否则会"明明快训完了，却又开始 10% 随机乱飞"，把已学策略冲掉。这是续训最经典的坑；
> - **回放池不保存**（50000 条 × 4×80×80 浮点 ≈ 5 GB，太贵）→ 续训时池子是空的。
>   所以只要 t 已过观察期，**凑满 32 条（约 32 步）就立刻开训**，**不会**重走 10 万步观察期；
>   代价是最初若干批来自很小的池子，更新噪声偏大，几万步内会自然回填到满，影响可忽略。

**存档为什么要存优化器状态？**
Adam 有自己的动量缓存，丢了它续训会"卡顿"一阵；步数不存则 ε 衰减和观察期判断会从头再来。

### 6.5 存档机制：步数编号 + 指针

```python
def save_ckpt(step, policy_net, target_net, optimizer, cfg):
    path = os.path.join(cfg.CKPT_DIR, f"ckpt-{step}.pt")   # 步数编号，永不覆盖
    torch.save({"policy": ..., "target": ..., "optim": ..., "step": step}, path)
    with open(cfg.LATEST_JSON, "w", encoding="utf-8") as f:  # 更新指针
        json.dump({"model_file": path, "step": step}, f)
```

**为什么不直接覆盖同一个 `latest.pt`？**
- 覆盖式存档一旦训崩**没法回滚**；
- 步数编号（`ckpt-10000.pt` / `ckpt-20000.pt` ...）可以随时挑历史版本继续或评估；
- `latest.json` 只记"最新是谁 + 步数"，续训时读它即可（和你另一个 Flappybird 项目的机制一致）。

### 6.6 TensorBoard：训练过程的仪表盘

```python
writer = SummaryWriter(cfg.TB_DIR)                    # 日志写到 runs/tb
writer.add_scalar("step/avg_loss", avg_loss, t)
writer.add_scalar("episode/score", episode_score, t)  # 每局结束写一次
...
writer.close()                                        # 结束时关掉，确保刷盘
```

启动看板（另开终端）：

```bash
D:/envs/pandagym/python.exe -m tensorboard.main --logdir runs/tb --host 127.0.0.1 --port 6006
# 浏览器打开 http://127.0.0.1:6006
```

**怎么靠曲线判断训练好坏？（面试也能讲）**

| 现象 | 判断 |
|---|---|
| `episode/score` 一直是 0 | 还没学会；前 10 万步是观察期**本来就不训练**，别急 |
| `step/q_max` 单调暴涨 | **Q 值爆炸** → 检查目标网络是否启用、学习率是否过大 |
| `step/avg_loss` 不降 | 学习率问题，或 `zero_grad()` 漏了 |
| `step/epsilon` 不变 | 还在观察期（ε 衰减要等 `t > OBSERVE`） |

---

## 7. `dqn/play.py` —— 评估模式

与训练唯一的区别：**ε=0 全程纯贪心，不训练、不存经验、不更新网络**。

两个规范写法值得注意：

- `@torch.no_grad()` 装饰器 = 把整个函数体包进 `with torch.no_grad()`，推理省显存提速；
- `net.eval()` 切评估模式（本项目没有 dropout/batchnorm，但**必须养成这个习惯**）。

得分统计用 `if reward == 1.0: score += 1` 来数过管数——因为环境死亡时会重置计分，直接读环境的 score 会拿到 0。

### 7.1 `--no-render`：无头快速评估

```bash
python dqn/play.py --episodes 10              # 开窗口，人眼看，30FPS 限速
python dqn/play.py --episodes 50 --no-render  # 不开窗口，约 1 局/秒
```

实现只有三处，讲的都是通用套路：

**① argparse 的 `store_true`**

```python
parser.add_argument("--no-render", action="store_true", help="关闭窗口渲染")
...
play(n_episodes=args.episodes, render=not args.no_render)
```

`action="store_true"` 的意思是：**命令行出现这个开关就置 True，不出现就是 False，后面不用跟值**。
所以 `--no-render` 是个"负向开关"，真正传给函数的是 `render = not args.no_render`。
（为什么不直接写 `--render`？因为评估默认就想要快的，但你偶尔又想看画面——把"快"设成显式开关反而容易忘加。）

**② `SDL_VIDEODRIVER=dummy`（关键：必须在 import pygame 之前）**

```python
if not render:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.chdir(cfg.ROOT)
sys.path.append(cfg.GAME_DIR)
import wrapped_flappy_bird as game_env      # ← 这一行内部才执行 pygame.init()
```

`dummy` 是 SDL 提供的一个"假显示器驱动"：所有绘图函数照常工作（画面画在内存里），只是不往真实屏幕刷。
和 `dqn/train.py` 的 `make_env()` 是同一个套路——**pygame/SDL 的驱动在 `pygame.init()` 那一刻就定死了，之后改环境变量没用**，
所以必须靠"延迟 import"把 pygame 的导入推迟到设完环境变量之后。这是本项目两个入口共用的一个坑。

**③ 把 `render` 透传给 `frame_step`**

```python
frame, reward, terminal = env.frame_step(a_vec, render=render)
```

环境侧（`game/wrapped_flappy_bird.py`）拿到 `render=False` 时只跳过两句：

```python
if render:
    pygame.display.update()   # 刷到真实窗口（人看）
    FPSCLOCK.tick(FPS)        # 限速 30FPS —— 最大瓶颈
```

注意 `SCREEN.blit(...)` 那些**照常执行**，并且 `image_data` 照样从内存 surface 里取出来返回——
也就是说**网络的输入完全没变，评估是逐帧等价的**，纯粹砍掉了"给人看"和"等时间"两件事。
这正是无头模式能放心用来统计真实水平的原因。

> ⚠️ **得分方差极大**：同一个模型跑 50 局，1 根到 20 多根都会出现。
> 别用 3~5 局的均值判断模型好坏，至少 50 局；对比两个模型时最好跑同样的局数。

---

## 8. 从 TensorFlow 移植到 PyTorch 的 6 个坑

| # | 坑 | TF 版 | PyTorch 版 |
|---|---|---|---|
| 1 | **数据布局** | NHWC `(80,80,4)` | NCHW `(4,80,80)`，状态数组要做对 |
| 2 | **padding** | `padding='SAME'` 自动补零 | 必须手算填数字（见 4.3） |
| 3 | **计算图** | `tf.placeholder` + `Session.run` 静态图 | 动态图，直接调 `net(x)`；用 `no_grad()` 控制是否求导 |
| 4 | **梯度清零** | 优化器内部处理 | **必须手动** `optimizer.zero_grad()` |
| 5 | **保存/加载** | `tf.train.Saver()` 存到 checkpoint 目录 | `torch.save(net.state_dict())` / `load_state_dict()` |
| 6 | **设备** | 自动占 GPU | 必须显式 `.to(device)`，且**数据和模型要在同一个设备** |

---

## 9. 我相对原版改了什么、为什么

| 改动 | 原版 | 本版 | 理由 |
|---|---|---|---|
| **目标网络** | ❌ 没有（用同一网络算目标） | ✅ 有，每 10000 步同步 | 原版只能靠 `lr=1e-6` 硬扛不发散；加了目标网络后可用 `1e-4`，收敛快得多 |
| **渲染开关** | 每帧强制 30 FPS | `frame_step(..., render=False)` | 30 FPS 是训练最大瓶颈，关掉提速数十倍 |
| **无头支持** | 需要窗口 | `SDL_VIDEODRIVER=dummy` | 可以后台训练 |
| **步数上限** | 死循环 | `MAX_STEPS` | 便于控制实验时长 |
| **预处理库** | `cv2` | `PIL` | 你环境没装 cv2，PIL 效果等价 |
| **代码组织** | 单文件平铺 | 按职责分模块 | 便于逐个击破地学习 |

**完全没改的**（保证你学到的是那个经典实现）：
网络结构、输入尺寸、二值化阈值、奖励函数、ε 起止值与衰减步数、回放池容量、batch size、γ。

---

## 10. 调参与排错

| 现象 | 可能原因 / 处理 |
|---|---|
| 训练很久还是不会过管 | 观察期 10 万步内**本来就不训练**，属于正常；先看是否过了观察期 |
| loss 不降或变成 NaN | 学习率太大 → 调小 `LR`；确认 `zero_grad()` 没漏 |
| Q 值越来越大、策略塌缩 | 目标网络没同步 / 没启用 → 检查 `USE_TARGET_NET` 与 `TARGET_SYNC` |
| 一直"贴地暴毙" | ε 太大 → 降低 `INIT_EPSILON` |
| 想看训练过程 | 去掉 `--no-render`（会很慢，仅演示） |
| 评估太慢 / 想统计真实水平 | `dqn/play.py --episodes 50 --no-render`（约 1 局/秒） |
| 无头模式报 SDL / 窗口错误 | 确认 `SDL_VIDEODRIVER` 在 `import wrapped_flappy_bird` **之前**设置 |
| 显存不够 | 调小 `BATCH_SIZE` |

**想快速看到效果怎么办？**
把 `dqn/config.py` 里的 `OBSERVE` 临时改小（比如 `2000`）、`EXPLORE` 改小（比如 `50000`），
几分钟就能看到它在学（但学不好，只是验证流程）。正式训练再改回来。
