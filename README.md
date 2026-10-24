# Flappy Bird：强化学习训练与策略解释

**DQN / PPO · 跨难度评测 · 输入归因与位置干预 · 线性叶代理树**

基于 [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird) 的游戏环境与素材，以 PyTorch 实现 DQN 和 PPO，围绕“策略学得怎样、决策依赖什么、能否用可读模型解释”开展三项工作。以下为已完成实验的结果与适用范围；不包含模型权重，也不将单次训练结果解释为算法的普遍优劣。

<img src="assets/figures/fig00_demo.gif" width="240" alt="Flappy Bird 智能体游戏示意动图">

## 核心工作与结果

| 工作 | 实现与分析 | 本次实验的主要观察 |
|---|---|---|
| 一：训练与评测 | DQN、PPO；检查点评测；跨难度测试 | 扫描检查点的最高平均得分分别为 118.8、2158.9；冻结 DQN 的跨难度得分非单调 |
| 二：策略行为解释 | 五种输入归因、区域遮挡、相位分析、位置干预 | 小鸟区域的重要性随相位变化；所测边界帧上，竖直位移呈现比水平位移更明确的方向性响应 |
| 三：代理模型解释 | 双动作线性叶树、特征影响、路径高亮、遮挡对照 | 历史拍翅树约 71% 分裂位于最新帧；特征影响计算的行索引问题已在当前实现中修正，历史空间排序待重新计算 |

## 工作一：算法实现与评测

### 训练设置

输入为连续四帧组成的状态，每帧处理为 80×80 二值图；动作为拍翅或不拍翅。存活、过管、死亡奖励分别为 0.1、1、−1。DQN 使用经验回放与目标网络，PPO 使用 GAE、策略概率比裁剪与优势标准化。实现支持无窗口训练、TensorBoard 日志及模型和优化器状态恢复。

两个算法分别训练约 200 万环境步。DQN 使用原像素尺度，PPO 将像素归一化至 [0,1]，其他训练设置也不完全相同，因此本实验不单独识别算法本身的因果效应。训练期得分含探索行为，推理期使用确定性动作，两条曲线的统计口径不同。

<table><tr><td width="50%"><img src="assets/figures/fig03_training_curve.png" alt="训练期最近100局平均得分"></td><td width="50%"><img src="assets/figures/evaluation.png" alt="推理期完整检查点学习曲线"></td></tr><tr><td>训练期：最近 100 局平均得分。</td><td>推理期：固定检查点各评测 10 局，区间为按回合 bootstrap 得到的 95% 区间。</td></tr></table>

### 检查点表现与跨难度评测

共评测 40 个 DQN 和 38 个 PPO 检查点，每个 10 局，共 780 局。

| 算法 | 最高均分对应环境步 | 平均得分 | 该检查点单局最高分 |
|---|---:|---:|---:|
| DQN | 1,890,000 | 118.8 | 495 |
| PPO | 1,860,000 | 2158.9 | 4291 |

以上为扫描后的最佳检查点，并非独立重复训练的平均结果。DQN 在 75 万至 80 万步的检查点评测中，均分由 86.4 降至 5.0，说明本次训练存在明显波动；仅凭该现象不能确证“灾难性遗忘”。后续解释分析统一冻结第 189 万步 DQN，不再训练。

<img src="assets/figures/gap_distribution.png" width="720" alt="冻结DQN的跨缝隙宽度得分分布">

*固定模型，在管道缝隙 100、120、150 像素下各评测 30 局，平均得分分别为 187.5、357.8、135.5。120 像素设置中部分回合达到 1000 分停止上限。该评测与上表的 10 局检查点评测是两个独立批次。*

更宽缝隙同时改变容错空间和视觉输入分布，可能共同影响结果；当前实验没有将两者分离，因此仅得出“所测设置下得分非单调”的结论。

## 工作二：输入归因与位置干预

### 分析对象与采样口径

“帧”指被采样的时间点；网络在该时间点接收由当前帧和前三帧组成的状态。不同实验独立采样，样本数量不应互相替代。

| 分析 | 样本及设置 |
|---|---|
| 相位与位置干预 | 45 局、89,423 帧；17 局发生死亡，28 局达到每局 2500 帧上限 |
| 决策边界帧 | 在上述轨迹中筛选 `abs(Q(s,1) − Q(s,0)) < 0.5`，得到 41,867 帧 |
| 区域遮挡 | 另一批 45 局、99,030 帧；每 16 帧抽取 1 帧得到 6190 帧，补充 435 帧死亡前窗口样本，按回合和帧号去重后共 6625 帧 |
| 代理解释对照 | 从独立测试回合的采样帧中取 120 帧，进行分块遮挡 |

边界筛选将分析集中于两个动作价值接近、容易改变动作的情形，适合研究局部决策敏感性；所得翻转比例不代表所有游戏帧上的比例。帧之间存在时间相关性，按帧重采样的区间不等于按独立回合估计的不确定性。

### 输入归因与删除检验

对冻结模型的动作价值计算 Saliency、SmoothGrad、Integrated Gradients、Occlusion 和 KernelSHAP。归因回答“输入变化与当前输出有何关系”，不等于模型内部的注意力权重。

<img src="assets/figures/local_attribution.png" width="850" alt="同一输入的五种局部归因">

*五种归因方法揭示不同的局部敏感区域；图像方向沿用原始输入约定，显示时转换为屏幕坐标。*

<img src="assets/figures/deletion.png" width="720" alt="按归因排序删除像素的动作价值保留曲线">

*按归因强度依次遮挡像素，记录动作价值保留曲线。下降更快说明排序在该遮挡规则下更能定位敏感输入；该检验依赖遮挡基线，不能证明解释唯一或完全忠实。*

区域归因密度定义为：

$$D_r=\frac{\text{区域 }r\text{ 的归因份额}}{\text{区域 }r\text{ 的面积份额}}.$$

密度 1 对应全图平均水平。以过管时刻对齐并划分 36 个相位区间，小鸟区域的平均密度在周期内约从 0.92 变化至 9.29；17 次死亡事件对齐后，死亡前 30 帧窗口中小鸟区域密度峰值为 14.81。这些是条件统计，不能据此认定该区域变化导致死亡。

<table><tr><td><img src="assets/figures/phase.png" alt="按过管相位对齐的区域归因"></td><td><img src="assets/figures/death.png" alt="死亡事件对齐的区域归因"></td></tr><tr><td>过管相位对齐揭示周期内的归因变化。</td><td>死亡事件对齐揭示失败前的局部敏感性变化。</td></tr></table>

### 区域遮挡与位置干预

区域遮挡在四帧输入中将对应区域置零，分别统计动作价值相对变化、单位面积变化和动作翻转率。这三个量的分母不同，不能互换。地面遮挡的原始价值变化较大，小鸟区域单位面积影响突出，下管道区域的动作翻转更明显。

<img src="assets/figures/region_ablation.png" width="850" alt="五区域遮挡的输出变化">

*区域遮挡反映对当前输入的敏感性；大面积遮挡也可能引入偏离正常游戏画面的输入。*

位置干预只改变小鸟在**最新帧**中的位置，使用游戏素材重新渲染，其余三帧历史保持不变。以

$$m(s)=Q(s,1)-Q(s,0),\qquad \Delta m=m(s')-m(s)$$

衡量拍翅倾向变化。再按小鸟是否与管道在水平方向重叠划分“管道内”和“管道间”，比较竖直位移与水平位移。屏幕坐标向下为正；历史图使用向上为正的展示约定，解读方向时须留意坐标。

<img src="assets/figures/intervention_geometry.png" width="720" alt="重新渲染小鸟位置的干预示意">

*干预保留管道、背景及历史帧，改变最新帧小鸟位置。历史零位移核验在 139 帧上与原渲染逐像素一致。*

<img src="assets/figures/intervention_response.png" width="850" alt="竖直位置干预与动作价值差响应">

*所测边界帧中，小鸟下移通常提高拍翅倾向，上移则相反；管道内对部分竖直位移更敏感。*

<img src="assets/figures/intervention_control.png" width="850" alt="水平位移对照与边界阈值敏感性">

*水平位移未呈现同样明确的方向性趋势；收紧边界阈值至 0.2、0.1 后主要方向仍可观察。该结果不等于水平位置完全无影响。*

这里只测量即时动作价值和决策变化，没有执行干预后的完整轨迹；单帧改动可能破坏时间连续性。因此不将其解释为真实飞行动作的因果收益。

## 工作三：线性叶代理树与解释核验

### 代理拟合

借鉴 Linear Model U-Tree（LMUT）的线性叶结构，离线拟合冻结 DQN 的两个动作价值，每个动作一棵树。本实现不是完整的在线 LMUT 算法。训练与最终测试使用不同回合；训练数据内部另留一部分帧用于开发。

四帧共 25,600 个二值像素特征，历史数据筛除近常量特征后保留 19,012 个。树按方差减少选择分裂，叶节点进行岭回归：

$$\widehat Q_a(s)=w_{\ell,a}^{\mathsf T}x+b_{\ell,a},\qquad
\min_{w,b}\sum_{i\in\ell}(Q_a(s_i)-w^Tx_i-b)^2+\lambda\|w\|_2^2.$$

<img src="assets/figures/lmut_structure.png" width="650" alt="线性叶代理模型结构示意">

*结构示意沿用报告中的 LMUT 方法说明图；项目实现为离线双动作代理树。方法来源见文末参考文献。*

### 特征影响与路径高亮

设节点样本集合为 $N$，教师价值在节点内的方差为 $v_N$，左右子节点为 $L,R$：

$$\Delta v_N=v_N-\frac{|L|}{|N|}v_L-\frac{|R|}{|N|}v_R.$$

对在节点 $N$ 用于分裂的特征 $f$，定义：

$$I_{N,f}=\left(1+\frac{w_{N,f}^{2}}{\sum_jw_{N,j}^{2}}\right)[\Delta v_N]_+,\qquad
I_f=\sum_{N:\,\text{split}(N)=f}I_{N,f}.$$

节点回归使用筛选后的 1024 个特征；分裂特征不在该集合中时权重修正项取 0，权重平方和为 0 时也取 0。以全部保留特征（包括零影响特征）的平均影响作为高亮阈值。在具体样本从根到叶的路径上，只高亮满足阈值的分裂像素。

历史拍翅树含 339 个叶节点、338 次分裂，其中 241 次位于最新帧（约 71%）。这描述代理树的结构，不意味着 DQN 只依赖最新帧。

<table><tr><td><img src="assets/figures/feature_influence.png" alt="历史代理树特征影响图"></td><td><img src="assets/figures/path_highlighting.png" alt="历史代理树局部路径高亮图"></td></tr><tr><td>历史特征影响可视化，尚未按修复后的索引重新生成。</td><td>历史路径高亮示例，依赖特征影响阈值。</td></tr></table>

**计算修正。** 整理时确认旧脚本在子集上建树后，曾将节点的局部行号用于完整数组，导致节点回归和特征影响的样本错位。当前代码将建树、节点回归、方差计算全部绑定于同一训练子集，并设置针对性测试。上面两幅原始图作为历史记录保留，不将其像素排序或高亮阈值作为修复后的定量结论；本次整理未重新运行完整代理拟合。

### 代理影响与教师遮挡敏感度

“留出”指这些帧来自未参与代理训练的测试回合，而非从训练结果中挑选出的成功帧。历史核验从该集合抽取 120 帧，将四个输入通道分别划分为 8×8 像素区块，共 400 个块。代理块影响为 $T_b=\sum_{f\in b}I_f$；教师敏感度为

$$D_b=\frac1n\sum_{i=1}^{n}|Q_1(s_i)-Q_1(M_b(s_i))|,$$

其中 $M_b$ 仅遮挡一个通道中的一个区块。二者量纲不同，用空间分布对照检查代理是否突出教师敏感区域，不据此认定代理完整解释教师。当前 `surrogate.py` 可保存两张分块图；历史代理影响受上述索引问题影响，需重新拟合后再做定量判断。

## 安装与运行

建议 Python 3.10+；从仓库根目录执行。先按硬件安装 PyTorch，再安装依赖。CPU 可运行测试和小规模推理，完整训练及归因建议使用 GPU。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python dqn/train.py --no-render --fresh --max-steps 2000000
python ppo/train.py --no-render --fresh --max-steps 2000000
python dqn/eval_curve.py --episodes 10
python ppo/eval_curve_ppo.py --episodes 10
python generalization/gap_eval.py --help
```

权重保存在各训练脚本的 `runs/` 目录，默认解释检查点为 `runs/ckpt-1890000.pt`。仓库不提供预训练权重；新训练不会保证逐点复现历史曲线。指定自己的检查点：

```bash
python interpretability/posthoc/explain_dqn.py --ckpt runs/ckpt-1890000.pt --episodes 45
python interpretability/posthoc/cf_regions.py --ckpt runs/ckpt-1890000.pt --episodes 45
python interpretability/posthoc/position_intervention.py --ckpt runs/ckpt-1890000.pt --episodes 45
python interpretability/intrinsic/surrogate.py --ckpt runs/ckpt-1890000.pt
```

`cf_regions.py` 保留原区域遮挡及探索性干预实现；报告的“最新帧重新渲染位置干预”对应新增的 `position_intervention.py`。后者是对报告方法的独立整理实现，不能声称与历史采样逐帧一致。代理脚本支持 `--data` 读取已采集的数组，避免重复运行游戏。

## 目录与复现边界

```text
dqn/                       DQN 训练、评测与游戏运行
ppo/                       PPO 训练、评测与游戏运行
env/                       游戏环境
generalization/            跨难度评测
interpretability/posthoc/  归因、区域遮挡、位置干预
interpretability/intrinsic/ 线性叶树与代理解释
results/                   历史逐回合结果与轻量日志
assets/                    游戏素材、原始示意动图和报告图片
tests/                     索引、数值计算、渲染一致性检查
```

图表和分数来自已有实验，不是本次整理重新训练所得。版本历史保留旧实现；重复图片、探索性扫描和历史辅助脚本已从当前目录移除。原实验文件和 LaTeX 报告未修改。模型、缓存、完整轨迹及 TensorBoard 事件文件不进入 Git。

## 来源与许可

- 游戏环境与素材：[yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)，保留上游 MIT 声明，见 [LICENSE](LICENSE)。
- DQN：Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015。
- PPO：Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017。
- 代理树方法：Liu et al., *Toward Interpretable Deep Reinforcement Learning with Linear Model U-Trees*。方法示意图用于学术说明，来源与自有实现区分标注。
