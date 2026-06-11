# AIQuant 后续工作交接(MA_TRADES 多资产扩散)

> 给接手的 Claude Code:这份是自包含交接。先读「0. 背景与已确立事实」(别重做已经做完的),再按三类任务推进。操作命令在文末「附录」。详细历史见 `EXPERIMENT_PLAN_ZH.md` §3.4f/g/h 和 memory `project-cluster38-run`。
> 日期基准:本文件写于 2026-06-11。

---

## 0. 背景与已确立事实(**不要重做**)

**方法是什么**:`MA_TRADES` = 把原 **TRADES** 主干原封不动包一层(`models/diffusers/multi_asset/shared_score_net.py` 里 `backbone = TRADES(...)`,资产轴拍进 batch 轴 → 同一主干跨资产共享权重 = **P0**)。在 P0 之上加**图耦合 = P1**(`graph/`:coupler→rolling_stats→edge_weight_net→message_passing→aggregator→noise_fusion,残差门控 `eps_fused = eps_local + gamma·MLP([eps_local, agg])`,gamma 初始 0)。相对 TRADES 的全部增量 = (1) 跨资产权重共享、(2) 图耦合。

**已确立结论(2026-06-11,多 seed 双指标确认)**:
- ❌ **图耦合(P1)无收益,是 dead weight**。放开 gamma(`GRAPH_LR=3e-4`)+ 3 seed(1234/42/7)配对:val_ema Δ(g−ng)=−0.001±0.0026(噪声);lob_bench MEAN Δ=+0.008(略差,符号摇摆);inter_arrival 3/3 全差。gamma 自发收敛到 ±0.01、符号 seed 间翻转 → 优化器主动放弃耦合。**不要再去"修图发散"——那是历史误诊(真因是 LR 过高,已解决)。**
- ✅ **训练默认 `LR=3e-4`**(1e-3 种子不稳定,会卡 val 5–6)。
- ✅ **基础方法的订单放置(depth/levels)接近采样地板**,远好于打乱天花板 —— 真实正面信号。
- ⚠️ **时序(inter_arrival ~0.6)是一致复现的短板**。
- ⚠️ **当前 lob_bench 评测不可与 TRADES 对标**:teacher-forced 单步 + 共享真实 book → book 指标(spread/imbalance/volume/ofi)恒等 0(假);单边生成 + order_id 全 0 → 3 个指标退化成 1.0;只 800 笔、单窗、只评了 BTC。**只测到订单边缘分布,没测到市场动态。**
- ⚠️ **共享权重(P0)的收益从未被验证**(没做过"共享 MA vs 每资产独立 TRADES"对照)。这是多资产方法真正该回答的问题,目前空着。

**关键 ckpt(`data/checkpoints/MA_TRADES/`)**:稳定 LR 基线 `val_ema=0.963_..._ng_s1234_lr3e4_e5.ckpt`(no-graph)、`val_ema=0.943_..._g_s1234_lr3e4_e5.ckpt`(graph);多 seed `*_{ng_s,g_s}{1234,42,7}_..._ms.ckpt`。lob_bench 输出在 `data/lobbench/`(单跑)和 `data/lobbench/ms/<tag>/`(多 seed)。

---

## 1. 数据(评测真实化 + 覆盖面) — ✅ 已完成(2026-06-11,见 `EXPERIMENT_PLAN_ZH.md` §3.4i)

> 动机:这是**性价比最高、解锁一切的前置**。不修这块,既无法与 TRADES 对标,也无法可信地评判任何改进。
>
> **✅ 完成情况**:新建 `DeepMarket/lob_engine.py`(price-time 撮合引擎)+ `build_lobbench_eval_ar.py`(自回归 rollout 评测构建器)。
> - **1.1** 有状态 rollout 替换 teacher-forced:gen 盘口=引擎重建,book 指标(spread/imbalance/volume/ofi)不再恒 0,真正是 gen 消息的函数。
> - **1.2** order_id:引擎分配 id + 撤单/成交引用真实挂单价;3 个退化指标(log_time_to_cancel / ask_cancellation_depth / ask_cancellation_levels 等)全部从 1.0 变成有意义 <1.0;双边生成。
> - **1.3** ETH+BTC、8 窗口×500 笔、跨窗口 bootstrap CI。
> - **结果**:21/21 指标全部 <1.0;MEAN BTC 0.456 / ETH 0.482。**新短板浮现=spread~0.98(rollout 盘口枯竭,cancel/exec≫limit)**,详见 §3.4i。
> - ⚠️ 关键数据事实:crypto adapter 的 depth 列恒 0 → 用绝对价解码而非 ABIDES depth-relative;DDIM-10 让 sequential rollout 跑得动。
> - 下面 1.1/1.2/1.3 原始任务描述保留供参考。

### 1.1 自回归生成(让 book 指标变真)
- **目标**:用有状态 rollout 替换 teacher-forced 单步,使 spread/imbalance/volume/ofi 不再恒等 0。
- **怎么做**:优先用仓库自带 **ABIDES**(`DeepMarket/ABIDES/abides.py`,README 有 `world_agent_sim` 调用例;解码逻辑参考 `WorldAgent._postprocess_generated_TRADES`)。若 ABIDES 接 MA 太重,退而用 **JaxLob** 从生成的 message 序列重建 book(`lob_bench` 侧已有依赖)。
- **改哪**:`DeepMarket/build_lobbench_eval.py`(现在是 teacher-forced;加一条 rollout 生成路径,env 切换)。
- **验收**:book 指标出现非 0、非退化的 L1 值;gen book 与 real book 的 spread/imbalance 分布可比。

### 1.2 消除退化指标(order_id + 双边)
- **目标**:让 `log_time_to_cancel` / `limit_ask_order_depth` / `ask_cancellation_depth`(现恒为 1.0)变得有意义。
- **根因**:生成的 message `order_id` 全 0 → 撤单无法对应到挂单;生成偏单边(bid)。
- **怎么做**:在解码/生成里建模或回填 order_id(撤单引用真实存在的挂单),并检查双边生成(方向分布 gen 0.475 vs real 0.47 其实接近,所以主因是 order_id,不是方向)。
- **改哪**:`build_lobbench_eval.py` 的 `decode_order` / `write_msgbook`。
- **验收**:三个退化指标给出 <1.0 的真实分数。

### 1.3 评测覆盖面 + 误差棒
- **目标**:从"单窗 800 笔单资产"扩到可统计的规模。
- **怎么做**:评 **ETH**(不只 BTC);多个 test 窗口;`N_EVENTS` 调大(env 已支持);每个配置跑多 seed 给均值±CI(score_lobbench 已输出 bootstrap CI)。
- **验收**:每个指标有跨窗口/seed 的均值和置信区间。

---

## 2. 短板改进 & 卖点强调

> **✅ 完成(2026-06-11,见 EXPERIMENT_PLAN_ZH.md §3.4k)**:tiling 法做"ETH 数据贫瘠 / BTC 满量"的数据效率头对头(独立 TRADES vs 共享 MA no-graph,full/p20/p10×2seed)。**预测度量(eval_p0)上 P0 卖点成立**:独立随数据缩小退化 ~26%(几乎全在 price,l1_price 0.027→0.15),共享主干钉在满量水平(cont_l1~0.19,l1_price~0.03);lob_bench bid 侧 depth 指标方向一致,但单 seed lob_bench 聚合被 spread/timing 噪声淹没看不出(需多 seed AR 才能在聚合上确证)。
> **✅ 完成(2026-06-11,§3.4l)**:试了 option (a) 时间通道 MSE 加权 + option (b) log-time 参数化,**两者都没降 lob_bench inter_arrival**(稳在 ~0.9);(a) 改善 teacher-forced 点时间误差但恶化 AR 分布,(b) 模型有效但 inter_arrival 不变。目标未达成,剩 option (c) point-process 头未试(列后续)。
> **✅ 完成(2026-06-11,§3.4j)**:TRADES 论文(arXiv 2502.07071)数字已抓取归档,并标注不可与我们的 crypto/AR lob_bench 直接对标。

### 2.1 【卖点】验证共享权重(P0)的价值 —— 最该先做
- **目标**:回答"多资产共享主干到底有没有用",这才是论文的核心多资产实验(图已废)。
- **怎么做**:head-to-head —— **(A) 每资产独立训练的 TRADES**(`CHOSEN_MODEL=TRADES`,BTC 一个、ETH 一个)vs **(B) 共享主干 MA_TRADES no-graph**(`DISABLE_GRAPH=1`)。重点考察**数据效率 / 小样本资产**:把某个资产的训练数据截断到 10–30%,看共享主干是否靠另一资产"借力"而独立 TRADES 退化。多 seed。
- **改哪**:用现成 `run_ma_c38.py`(MA 侧);TRADES 侧用仓库原训练入口(README §"Training a TRADES Model",`configuration.py` 设 `CHOSEN_MODEL=cst.Models.TRADES`)或仿照 `run_ma_c38.py` 写一个 `run_trades_c38.py`。两侧都用 `LR=3e-4`、同协议评 lob_bench。
- **验收**:得到"共享 vs 独立"在(全量 / 小样本)下的 lob_bench + val 对比;若共享在小样本资产上显著更好 → 多资产方法的真实卖点成立。

### 2.2 【短板】攻时序:inter_arrival
- **目标**:把一致偏高的 inter_arrival(~0.6,地板 0.064)降下来 —— 唯一稳定的指标缺口,改了直接涨分。
- **怎么做(择一/组合)**:(a) 给 time 维单独加大损失权重;(b) log-time 参数化;(c) 换**点过程(Hawkes / point-process)头**专门建模事件间隔。
- **改哪**:损失 `ma_diffusion_engine.py` 的 `_mse_loss_per_asset`(目前对所有列等权 L2);或在 `SharedScoreNet`/`TRADES` 输出加 time 头。
- **验收**:inter_arrival L1 明显下降且不损害其他指标,多 seed 稳定复现(不像 §3.4h 那样反号)。

### 2.3 【对标靶子】拉 TRADES 原论文数字
- **目标**:给改进定一个量化基准。TRADES 论文 = arXiv **2502.07071**(`berti2025trades`)。
- **怎么做**:WebFetch/WebSearch 取论文报告的 lob_bench / predictive-score 数字(LOBSTER INTC/TSLA)。**注意协议差异**(他们 ABIDES rollout、股票;我们若还没做 1.1 就不可直接比)。**不要凭记忆编数字。**

---

## 3. 图的探索 & 训练稳定性

> 前提:§3.4h 已证当前设置下图无用。本节是"如果还想救图"的研究方向 + 顺手的稳定性改进。优先级低于 §1、§2。

### 3.1 【图】换图能起作用的战场
- **判断**:当前 BTC+ETH 同步性不足以让跨资产耦合产生增益;残差小门控太弱;图只从去噪损失拿梯度,信号不够。
- **怎么做**:
  - **数据**:换**强 lead-lag 标的对** —— 指数/期货 vs 成分股、ADR、同一资产多交易所(此项也属 §1 数据采集;`data/_adapter_raw/` + `build_real_datasets.py` 是现成的归一化/切分管线)。
  - **结构**:用比 gamma 残差门控更强的耦合;或加**跨资产辅助目标**(如让图分支预测另一资产下一步),给它去噪损失之外的梯度。
  - **训练**:更长 epoch;别用 `GRAPH_LR=1e-5` 钉死 gamma。
- **验收**:在新标的/新结构上,graph−nograph 的 Δ 在多 seed 上**符号一致且超出方差**(对照 §3.4h 的判定标准)。

### 3.2 【稳定性】降种子敏感
- **目标**:消除 0.94–0.96 间的 seed 抖动 + 历史上的 LR 敏感。
- **怎么做**:(a) **LR warmup + cosine 调度**(现在是固定 LR + on_validation 里手动减半);(b) 把 `_mse_loss_per_asset` 的 **L2-norm 改成 mean / 归一化**(现在用 `torch.norm(p=2)` 不是 mean,loss 尺度 ~√(K·F),早期梯度偏热,疑似种子敏感来源);(c) 检查 `LossSecondMomentResampler` 冷启动。
- **改哪**:`run_ma_c38.py`(调度)、`ma_diffusion_engine.py`(损失)、`TRADES/Sampler.py`(resampler)。
- **验收**:固定超参下多 seed 的 val 方差下降;能用回 LR≥1e-3 而不发散。

---

## 附录:操作速查(已在本环境验证)

**连 cluster38**(唯一通路,密码在脚本内):
```bash
bash ~/.aiquant_c38.sh '<远程命令>'
```
**激活环境**(远程默认无 conda/python in PATH):
```bash
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh
conda activate deep_market   # DeepMarket 训练/生成；评分用 conda activate lob
```
**仓库**:NAS `/DATA/DATANAS1/wpj24/AIQuant`(`DeepMarket/` 子目录)。2 张 A100(GPU0/1)。

**训练 launcher** `DeepMarket/run_ma_c38.py`,env 钮:
`SMOKE=0 NDEV=1 SEED=<int> LR=3e-4 GRAPH_LR=<图分支LR> EPOCHS=5 ASSETS=BTCUSDT,ETHUSDT DISABLE_GRAPH={0,1} DISABLE_ARB_GUIDANCE=1 CKPT_SUFFIX=<tag> CUDA_VISIBLE_DEVICES=<0|1>`
> 注:`DISABLE_ARB_GUIDANCE=1` 必须带(P3 guidance 有 NaN bug,见 memory)。
```bash
CUDA_VISIBLE_DEVICES=0 SMOKE=0 NDEV=1 SEED=1234 LR=3e-4 EPOCHS=5 ASSETS=BTCUSDT,ETHUSDT \
  DISABLE_GRAPH=1 CKPT_SUFFIX=mytag setsid python run_ma_c38.py > /tmp/mytag.log 2>&1 &
```

**评测管线**(两步,各自 env):
```bash
# 1) 生成(deep_market, GPU)：CKPT/OUT/N_EVENTS 可 env 覆盖
CKPT=<ckpt路径> OUT=<输出目录>/BTCUSDT/2024-01-01 python DeepMarket/build_lobbench_eval.py
# 2) 评分(lob)：OUT 同上
OUT=<同上> python lob_bench/score_lobbench.py     # 打印 L1/Wass + MEAN，写 lobbench_scores.json
```

**多 seed / 多 ckpt 驱动**:本 session 用过 `/tmp/multiseed_driver.sh`(训练队列)、`/tmp/eval_ms_driver.sh`(评测队列)。`/tmp` 会被清,**接手第一步建议把这两个模式固化进仓库**(如 `DeepMarket/_scripts/`,untracked)。结构很简单:外层 `for SEED/ for tag` 循环,GPU0/GPU1 各起一个 `setsid python ... &` 后 `wait`。

**等远程长任务**:`Bash run_in_background` + `until grep <DONE标记> <log>; do sleep 60; done`,别前台 sleep 轮询。

**别碰**:不提交 `data/`/`*.npy`/`*.ckpt`;不引入 git-lfs(cluster 无 lfs)。
