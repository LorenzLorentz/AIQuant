# cluster38 多资产实验：模型、数据集与训练计划

> 最后更新：2026-06-11。环境：cluster38（2×A100-40GB）。
> 仓库部署在 NAS `/DATA/DATANAS1/wpj24/AIQuant`（多资产图扩散主线，区别于本地 Mac 的数据适配器线）。
> 相关记忆/文档：`DATA_SOURCES_ZH.md`、`DATA_STRATEGY_ZH.md`（均在本地 Mac 仓库）。

---

## 0. 运行环境（cluster38）

| 项 | 情况 |
|---|---|
| 连接 | 直连 SSH 不行；`cluster0`(密钥) → `cluster38`(密码 `sshpass`)。本地封装 `~/.aiquant_c38.sh` |
| GPU | 2×A100-SXM4-40GB；驱动 **CUDA 12.2** |
| 关键修复 | 共享 `deep_market` env 原本 `torch 2.12+cu130`（要 CUDA13）→ GPU 不可见；已换 `torch 2.5.1+cu121` |
| 磁盘 | `/home` **100% 满**；所有缓存/临时/输出重定向到 NAS（`MPLCONFIGDIR`、`PIP_CACHE_DIR`、`TMPDIR=/tmp/wpj24_tmp`）|
| conda | `source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh && conda activate deep_market` |
| 日志 | `/DATA/DATANAS1/wpj24/AIQuant_logs/` |

---

## 1. 模型清单（`DeepMarket/models/`）

| 模型 (`cst.Models`) | 类型 | 位置 | 说明 |
|---|---|---|---|
| **TRADES** | 条件扩散（单资产） | `diffusers/TRADES` | 原始主模型（LOB 订单生成扩散） |
| **MA_TRADES** | **多资产共享骨干扩散 + 图耦合** | `diffusers/multi_asset` | 当前主线 WIP。共享 TRADES 骨干 + 资产 embedding；图模块做跨资产 message-passing |
| **CGAN** | 条件 GAN | `gan` | 基线对照，已有预训练权重 |
| **CDT** | 条件扩散 transformer 骨干 | `diffusers` | 枚举中存在 |
| feature augmenters | MLP 特征增强 | `feature_augmenters` | 给扩散输入做增强 |

**MA_TRADES 内部分阶段（P0–P4，来自 `DATA_STRATEGY_ZH.md` 与代码开关）：**

- **P0** 共享骨干多资产扩散（`shared_score_net` + `ma_gaussian_diffusion`）✅ smoke 通过
- **P1** 图耦合 message-passing（`graph/`：aggregator/edge_weight_net/message_passing/relation_embedding/noise_fusion/rolling_stats），`gamma` 控制耦合强度 ✅ smoke 通过
- **P2** spread-aware conditioning
- **P3** quasi-no-arbitrage energy guidance（默认关；`DISABLE_ARB_GUIDANCE`）
- **P4** posterior consistency / cross-asset 指标

**消融开关**（`configuration.py`）：`DISABLE_GRAPH`、`FREEZE_EDGE_WEIGHTS`、`DISABLE_ARB_GUIDANCE`、`GRAPH_RELATION_DIM`、`GRAPH_HIDDEN_DIM`。

---

## 2. 数据集清单

所有训练数据统一为 **每资产一个目录、46 列 `.npy`**：
`[time(z), event_type{0,1,2}, size(z), price(z), direction(±1), depth(z), 40列LOB(z)]`
（从零训练，故用「训练集统计量做 per-列 z-score、第 1/4 列分类列不动」即可，无需复用论文常数。）

### 2.1 cluster38 上现有、可直接训练的数据（`DeepMarket/data/`）

| 资产目录 | 真/合成 | 来源 | train 行数 | 深度 | 备注 |
|---|---|---|---|---|---|
| **INTC** | **真实** | LOBSTER 样本 2012-06-21（仓库自带 zip）→ 仓库 `preprocess_data()` | 520,584 | 真 10 档 L3 | 真·逐笔，唯一真实 LOBSTER |
| **BTCUSDT** | **真实** | Tardis binance-futures book_snapshot_25，2024-01-01 | 255,000 | 真 10 档（25→10） | order 列为快照差分**合成 proxy** |
| **ETHUSDT** | **真实** | 同上 | 255,000 | 真 10 档 | 与 BTC 天然耦合 → P1 图耦合有真实信号 |
| **AAPL_A / AAPL_B** | **真实** | Databento AAPL mbp-10（30 分钟，约 $0.067）切两半 | 206,071 ×2 | 真 10 档 | 真·多档股票；proxy order |
| SYNTH | 合成 | 我此前生成的耦合假资产 | 520,584 | — | **已被真实数据取代，建议弃用** |

> **关键 caveat（论文须如实写）**：除 INTC 外，crypto/股票快照源的 order 流是**从盘口差分合成的 proxy**，不是真实成交/撤单序列；不能宣称 L3 message replay realism。多档 book（`cond_lob`）与跨资产耦合是真实的。

### 2.2 本地已验证、可按需补充的源（在 Mac `~/aiquant_data_verify/`，未传服务器）

| 源 | 状态 | 量/限制 |
|---|---|---|
| Tardis 原始日（BTC 144万 / ETH 152万行） | 已下完整 1 天，npy 仅截 30 万 | 免费仅「每月 1 号」单日；更多需 $650/月或自录 |
| Databento AAPL | 仅 30 分钟样本 | **$125 额度**可在服务器上拉更多（约 3 月 QQQ+成分） |
| FI-2010 | 1 个 test 文件 | sanity-check |
| Binance bookTicker | 1 整天 1850 万行 | **仅 L1(4/40)**，深度退化，主线勿用 |
| Binance 自录 depth20 | 仅 18s 验证 | 免费无限、**向前实时采集**（需服务器挂 tmux 跑数小时） |

### 2.3 数据规模判断

- **第一波实验**：现有真实数据（INTC 52万 / BTC+ETH 各 25.5万 / AAPL 20.6万×2）**足够**，无需再下。
- **扩规模时**：股票走 **Databento $125**（服务器有公网，key 在本地 `.env`）；加密走 **服务器自录**（零成本）或付费 Tardis。

---

## 3. 训练计划

### 3.1 多 GPU 策略（重要约束）

- **真正的单模型 DDP 当前跑不通**：WIP 模型把 `cst.DEVICE` 写死为 `cuda:0`（type_embedder 权重、若干索引张量），DDP rank1（cuda:1）设备不匹配；且图参数每步未必参与 loss（需 `find_unused_parameters`）。
- **当前方案**：用两张卡跑**两个并行单卡实验**（天然就是消融），互不干扰。
- **待办（如需数据并行训大模型）**：把多资产模型里写死的 `cst.DEVICE` 改成「跟随输入张量的 device」，再开 `ddp_find_unused_parameters_true`。

### 3.2 实验矩阵

| # | 实验 | 数据 | 模型/开关 | 卡 | 目的 |
|---|---|---|---|---|---|
| **E1** | 图耦合 ON vs OFF | INTC+SYNTH（旧）| MA_TRADES，`DISABLE_GRAPH` 0/1 | 0/1 | 已在跑（用合成数据，将被 E2 取代）|
| **E2** | **真实 crypto 篮子** | **BTCUSDT+ETHUSDT** | MA_TRADES graph ON | 0 | 主线：真实耦合资产 |
| **E3** | 图消融 | BTCUSDT+ETHUSDT | MA_TRADES graph OFF | 1 | E2 对照，验证 P1 增益 |
| **E4** | 真实股票多资产 | INTC+AAPL_A（或 AAPL_A+B）| MA_TRADES graph ON | — | 真实股票线 |
| **E5** | 单资产基线 | 各单资产 | TRADES / CGAN | — | baseline 对照 |
| **E6** | P2/P3 消融 | BTC+ETH | spread / energy guidance 开关 | — | 方法贡献拆解 |
| **E7** | 评测闭环 | 生成 vs 真实 | `lob_bench` | — | 分布 / cross-corr / lead-lag 指标 |

### 3.3 启动方式

启动器 `DeepMarket/run_ma_c38.py`（不改 WIP `configuration.py`，env 控制）：

```bash
# 环境
source /DATA/DATANAS1/wpj24/miniconda3/etc/profile.d/conda.sh && conda activate deep_market
cd /DATA/DATANAS1/wpj24/AIQuant/DeepMarket
export MPLCONFIGDIR=/DATA/DATANAS1/wpj24/.mplcache TMPDIR=/tmp/wpj24_tmp

# E2: 真实 BTC+ETH，图 ON，GPU0（需先把 run_ma_c38.py 的资产改成 BTCUSDT/ETHUSDT）
SMOKE=0 NDEV=1 EPOCHS=5 DISABLE_GRAPH=0 CUDA_VISIBLE_DEVICES=0 python run_ma_c38.py
# E3: 同数据，图 OFF，GPU1
SMOKE=0 NDEV=1 EPOCHS=5 DISABLE_GRAPH=1 CUDA_VISIBLE_DEVICES=1 python run_ma_c38.py
```

> 注：`run_ma_c38.py` 目前资产硬编码为 `[INTC, SYNTH]`，需加一个 `ASSETS` 环境变量来切换到 `BTCUSDT,ETHUSDT`（下一步改）。

环境变量：`SMOKE`(1=极小验证) `NDEV`(GPU 数) `EPOCHS` `DISABLE_GRAPH` `CKPT_SUFFIX`。
默认超参（`HP_TRADES_FIXED` + `configuration.py`）：SEQ_SIZE 256、AUGMENT_DIM 64、CDT_DEPTH 8、BATCH 256、LR 1e-3、扩散步 100。单卡约 **11 分钟/epoch**（52 万行）。

### 3.4 实测发现（2026-06-10）

- **P3 energy guidance 在真实 crypto 数据上产生 NaN**：`MA_TRADES` graph ON + 默认 `DISABLE_ARB_GUIDANCE=False` 在 BTC+ETH 上训练时，loss 出 NaN，污染 TRADES 的重要性采样器（`Sampler.py` 的 `np.random.choice(..., p=p)` 报 `probabilities contain NaN`），第 2 个 batch 即崩。
  - **规避**：`DISABLE_ARB_GUIDANCE=1` 后 graph(P1) 路径正常训练。
  - **根因待修（WIP 代码）**：quasi-no-arbitrage energy guidance 的能量/梯度数值不稳定，需加 NaN/inf 保护或归一化；修好前 P3 在真实数据上不可用。
  - 启动器已加 `GRAD_CLIP`(默认 1.0)、`DISABLE_ARB_GUIDANCE`、`FREEZE_EDGE_WEIGHTS` 三个 env 开关。

### 3.4b 修复与验证（2026-06-10 第二轮）

- **Fix 1 — NaN 健壮性（已修+验证）**:`Sampler.py` `update_losses` 跳过非有限 loss、`sample()` 对权重 `nan_to_num`+clip+退化时回退均匀分布;启动器加 `gradient_clip_val`(默认1.0)。~~根因是图路径梯度爆炸~~(**已更正,见 §3.4f**:根因是 LR=1e-3 过高导致训练种子不稳定,与图无关)。
- **Fix 2 — DDP 设备可移植(已修+验证,真双卡跑通)**:把 diffusion schedule 张量(alphas/posterior 等)与 TRADES backbone 的 `t_embedder`/`pos_embed` 正弦嵌入注册为 **buffer**(原为普通属性,停在 cuda:0);forward 内新建张量改用输入张量 device;`Sampler.sample(device=)`;EMA 在 `on_fit_start` 对齐 device;DDP 用 `ddp_find_unused_parameters_true`。`NDEV=2` 真实 BTC+ETH 已 `TRAIN_DONE`。
  - 遗留(非阻塞):DDP 下 `self.log("val_ema_loss")` 建议加 `sync_dist=True`(EarlyStopping/checkpoint 跨卡一致性);当前用 rank0 值。

### 3.4c E2/E3 结果(真实 BTC+ETH,5 epoch)

| 实验 | best val_ema | 走势 | checkpoint |
|---|---|---|---|
| **E3 no-graph** | **0.918**(epoch4) | 单调下降 0.95→0.92,健康 | `val_ema=0.918_..._nograph.ckpt` |
| **E2b graph** | **~6.0**(epoch0) | 始终不降,从未超过 epoch0 | `val_ema=5.976_..._graph_noarb.ckpt` |

→ ~~**结论**:图耦合路径反而把训练带坏~~ **⚠️ 此结论已被 §3.4f 推翻**:E2b/E3 的对比**没有控制随机种子**,差异是 LR×种子混杂,**不是图造成的**。详见 §3.4f。

### 3.4d E4/E5/E7 可运行性确认

- **E4(INTC+AAPL_A 多资产 MA_TRADES)**:✅ smoke 通过(train `(2,206071,6)`,0 报错)。
- **E5(单资产 TRADES baseline,INTC)**:✅ smoke 通过(走单资产 LOBDataset+DiffusionEngine,train `(520584,46)`,loss 2.77)。
- **E7(lob_bench)**:✅ **已跑通并出分**。
  - 环境修复:`pip install -U statsmodels>=0.14.4 scikit-learn>=1.5`(numpy2 兼容);推本地修好导入的 `lob_bench/*.py` 到 NAS。
  - 适配器 `DeepMarket/build_lobbench_eval.py`(deep_market env,GPU):加载 no-graph BTC+ETH checkpoint,**batched teacher-forced** 在真实 test 窗上逐步生成下一笔订单(BTC+ETH 双资产条件,取 BTC),按本数据集自己的 per-列 train 统计量反归一化、用 `type_embedder` 权重解码 event_type,写成 LOBSTER message+book CSV(real/gen/cond 三目录 + Simple_Loader 命名)。生成 800 笔 ≈ 28s。
  - 打分 `lob_bench/score_lobbench.py`(lob env):`Simple_Loader` + `scoring.run_benchmark`(DEFAULT 全 21 指标,L1+Wasserstein,bootstrap CI)。结果存 `data/lobbench/.../lobbench_scores.json`。
  - **结果(gen vs real BTCUSDT,L1)**:mean **0.209**;inter_arrival 0.598、limit_bid_depth **0.019**、bid_cancel_depth 0.022、limit_levels 0.035/0.111、vol_per_min 0.48;**degenerate**:limit_ask_depth / ask_cancel_depth / log_time_to_cancel = 1.0(单边生成 + order_id 全 0)。
  - **诚实 caveat**:gen 与 real **共用真实 book**(teacher-forced),故 book-only 指标(spread/imbalance/volume/ofi)恒等 → L1=0,**非模型功劳**;要让这些有意义需用 JaxLob 从 gen messages 重建 gen book。time_to_cancel 因 order_id 全 0 退化。真正有信号的是订单放置类指标(depth/levels/inter-arrival)。ask 侧退化提示生成方向严重偏 bid——又一个建模 bug。

### 3.4e lob_bench 基线对比(2026-06-10,gen vs real BTCUSDT,800 笔)

统一协议:同一条真实续写为 real;每个候选都是「续写订单 + 其所在 book 上下文」。floor=另一段独立真实订单(各用自己的 book)=采样地板;naive=列内打乱真实订单=破坏联合结构的天花板;book-only 指标(spread/imbalance/volume/ofi)因协议恒等已排除。L1 越小越像真实。

| 指标 | floor(地板) | naive(天花板) | **no-graph** | graph |
|---|---|---|---|---|
| log_inter_arrival_time(时序) | 0.064 | 0.001* | **0.660** | 0.992 |
| limit_bid_order_depth | 0.000 | 0.780 | **0.017** | 0.911 |
| bid_cancellation_depth | 0.000 | 0.697 | **0.026** | 0.848 |
| limit_bid_order_levels | 0.138 | 0.369 | **0.034** | 1.000 |
| limit_ask_order_levels | 0.068 | 0.544 | 0.139 | 1.000 |
| bid_cancellation_levels | 0.138 | 0.383 | 0.057 | 1.000 |
| ask_cancellation_levels | 0.090 | 0.458 | 0.117 | 1.000 |
| **MEAN(有效,<1.0)** | **0.071** | **0.462** | **0.150** | **0.917** |

\* naive 打乱保留了 inter-arrival 边缘分布 → 该指标 ≈0,对时序不是有意义的天花板。

**ask DEPTH 指标(limit_ask_order_depth / ask_cancellation_depth)= 1.0 对所有候选(含 floor)** → 是评测/价格重建 artifact,非模型问题,已排除。(早先怀疑的"生成偏 ask/bid 单边"被否定:方向分布 gen 0.475 vs real 0.47,几乎一致。)

**结论**:
- **no-graph 在订单放置(depth/levels)上接近采样地板、远好于 naive 天花板**(MEAN 0.150 vs floor 0.071 vs naive 0.462)——这是真实的正面结果,模型学到了"订单相对盘口怎么放"。
- **no-graph 时序差**:inter_arrival 0.660 ≫ floor 0.064 —— event timing 没学好,是主要短板。
- ~~**graph 全面崩**(MEAN 0.917)~~ **⚠️ 作废,见 §3.4g**:这个 graph checkpoint 来自一次种子不稳定(LR 过高)的发散训练。稳定 LR 新 ckpt 重评后 **graph MEAN 0.206 ≈ no-graph 0.215**。

### 3.4f 图发散归因 —— 重大更正(2026-06-11)

**结论:图耦合不是发散的原因;真正病根是 LR=1e-3 过高导致训练对随机种子极不稳定。"图破坏训练"全面撤回。**

逐步证据(诊断脚本 `diag_graph.py` / `diag_grad.py`,实验 launcher 加了 `SEED`/`LR`/`GRAPH_LR` env):

1. **gamma 不是病根**:图融合是残差门控 `ε_fused = ε_local + gamma·MLP(...)`,`gamma` 初始 0。先怀疑 gamma 冷启动失控(实测旧 run gamma 1000 步窜到 0.40),给 gamma+图模块单独 1e-5 LR 压住了 gamma(→0.0009),但 **loss 照样发散到 ~6**。
2. **图前向惰性**:在发散 checkpoint 上,`‖ε_fused−ε_local‖/‖ε_local‖ = 1.3%`;关掉 fuse(disable_graph)后纯 trunk 的 loss = 5.484,**与开图完全相同** → 坏的是 trunk 本身,不是融合。
3. **图反向惰性**:同一 batch 上 graph-ON vs OFF 的 **trunk 梯度逐位相同**(cosine=1.000000,范数比 1.0),图分支梯度 ~0.004,grad-clip 对两者缩放一致 → 排除跨资产梯度耦合 / 裁剪缩放。
4. **种子对照 A/B**(同 seed,唯一变量 disable_graph):batch-0 loss 逐位相同(5.40619),graph-ON 与 no-graph **全程贴合**(差 ~5% 恒定偏置)。**图非因果。**
5. **真病根 = LR**:同一个"卡住"的 seed,只把 LR 1e-3→3e-4:

   | seed | LR 1e-3 | LR 3e-4 |
   |---|---|---|
   | 1234 | 卡 val 4.9 | **降到 val 1.05**(3ep) |
   | 42 | 卡 val 6.1 | **降到 val 1.30**(ep0,继续降) |
   | 原随机种子(E3) | **0.918**(运气好) | — |

   LR=1e-3 下多数种子卡在 val 5–6,个别走运降到 0.918;LR=3e-4 把卡住的种子都救回。

6. **稳定 LR 下的公平对照**(seed 1234,LR 3e-4,3 epoch):**no-graph val_ema 1.049 vs graph-ON 1.037 —— 统计上相同,图略优。** checkpoint:`val_ema=1.037_..._g_s1234_lr3e4.ckpt` / `val_ema=1.049_..._ng_s1234_lr3e4.ckpt`。

**含义 / 行动项**:
- E2/E3 那张"graph 6.0 vs nograph 0.918"对比表(§3.4c)**没控种子,作废**;3.4e 里的 graph 列同样作废(那个 graph ckpt 来自发散训练)。
- **所有 MA_TRADES 训练默认改用 LR=3e-4**(或加 warmup);1e-3 不可靠。
- 图耦合在当前规模/训练长度下**既不帮也不害**(gamma~0 时本就近似恒等)。"图是否真有增益"是个需要更长训练 / 更强耦合的**研究问题,不是 bug**。
- no-graph 的 lob_bench 正面结论(订单放置接近地板,§3.4e)不受影响;graph 的 lob_bench 已用稳定 LR 的新 ckpt 重做,见 §3.4g。

### 3.4g 稳定 LR 收尾:5-epoch 公平对照 + 有效 lob_bench 重评(2026-06-11)

把 §3.4f 的 seed 1234 / LR 3e-4 对照延长到 **5 epoch**(双卡并行,唯一变量 disable_graph),训完用**新 ckpt** 重跑 lob_bench(替换 §3.4e 中作废的发散-ckpt graph 列)。

**(a) 训练 val_ema 收敛**(seed 1234,LR 3e-4,BTC+ETH):

| epoch | no-graph | graph |
|---|---|---|
| 1 | 1.148 | 1.125 |
| 2 | 1.049 | 1.037 |
| 3 | 0.990 | 1.006 |
| 4 | 0.963 | 0.977 |
| **5** | **0.963** | **0.943** |

→ LR=3e-4 在 5 epoch **稳定收敛到 ~0.95**,复现了当初"走运种子"的 0.918 水平(印证 §3.4f:0.918 是种子运气,不是 no-graph 的优势)。`graph gamma` 全程打印 0.000000(实际 ~1e-3),图融合近似恒等。ckpt:`val_ema=0.963_..._ng_s1234_lr3e4_e5.ckpt` / `val_ema=0.943_..._g_s1234_lr3e4_e5.ckpt`。

**(b) lob_bench 重评**(gen vs real BTCUSDT,800 笔,同 §3.4e 协议;`build_lobbench_eval.py` + `score_lobbench.py` 已加 `CKPT`/`OUT` env 钮):

| 指标(L1,越低越像真实) | no-graph e5 | graph e5 |
|---|---|---|
| **MEAN(全 21 项)** | 0.215 | **0.206** |
| **MEAN(有效,<1.0)** | 0.084 | **0.074** |
| log_inter_arrival_time(时序) | 0.666 | **0.592** |
| vol_per_min | 0.364 | **0.217** |
| limit_bid_order_levels | 0.091 | **0.078** |
| ask_cancellation_levels | **0.151** | 0.211 |
| 退化项=1.0(单边生成伪影) | log_ttc / limit_ask_depth / ask_cancel_depth | 同 |

**结论(替换 §3.4e graph 列)**:
- 稳定 LR 下 **graph(0.206)≈ no-graph(0.215)**,lob_bench 上统计相同 —— 与 val loss(0.943 vs 0.963)一致。**旧的"graph 全面崩 MEAN 0.917"彻底作废**(那是发散 ckpt 伪结果)。
- graph 在**订单时序**(inter_arrival 0.592 vs 0.666、vol_per_min 0.217 vs 0.364)上略好 —— 恰是 §3.4e 标记的 no-graph 弱点;但 ask_cancellation_levels 反而略差,且**仅单 seed**,只能算提示性。**⚠️ 该"提示"已被 §3.4h 否定:放开 gamma 多 seed 后 inter_arrival 优势反号,是噪声。**
- 退化项(log_ttc / limit_ask_depth / ask_cancel_depth = 1.0)仍是单边生成 + order_id 全 0 的评测伪影,两模型一致,与模型质量无关。

### 3.4h 图收益的决定性检验:放开 gamma + 多 seed 配对(2026-06-11)—— 结论:图无收益

§3.4g 的单 seed 迹象(graph 略好、inter_arrival 占优)有两个硬伤:n=1 无误差棒,且 gamma 被 `GRAPH_LR=1e-5` 钉在 ~0(图前向贡献仅 ~1%,等于没真正开图)。本实验**放开 gamma**(`GRAPH_LR=3e-4`,与 trunk 同速)并跑 **3 个 seed 配对**(1234/42/7,每个 seed graph vs no-graph,唯一变量 disable_graph,LR3e-4,5ep,BTC+ETH),直接检验"图是否有收益"。

**(a) gamma 放开后并不暴走、也长不大**:3e-4 下 gamma 温和增长(seed1234 半 ep 到 0.0028),5ep 末仅到 **±0.01 量级,且符号 seed 间翻转**(+0.0086 / −0.0143 / +0.0085)→ 优化器不把 gamma 推大,即**数据/任务不需要跨资产耦合**(不是冷启动暴走,也不是没训到)。

**(b) val_ema 配对**(Δ=graph−nograph,负=graph好):

| seed | no-graph | graph(放开) | Δ |
|---|---|---|---|
| 1234 | 0.963 | 0.965 | +0.002 |
| 42 | 0.949 | 0.946 | −0.003 |
| 7 | 0.948 | 0.946 | −0.002 |
| **均值** | | | **−0.001 ± 0.0026** |

**(c) lob_bench 配对**(同上,正=graph差):

| seed | MEAN(all) ng→g (Δ) | MEAN(非退化) (Δ) | inter_arrival ng→g (Δ) |
|---|---|---|---|
| 1234 | 0.200→0.206 (+0.006) | 0.067→0.074 (+0.007) | 0.601→0.651 (**+0.050**) |
| 42 | 0.235→0.276 (+0.041) | 0.108→0.155 (+0.047) | 0.620→0.622 (+0.002) |
| 7 | 0.256→0.232 (−0.024) | 0.132→0.104 (−0.028) | 0.610→0.640 (+0.030) |
| **均值 Δ** | **+0.008** | +0.009 | **+0.027(3/3 全差)** |

**结论(最终,推翻 §3.4g 的"提示性优势")**:
- **图耦合(P1)在当前规模/数据/训练长度下没有收益。** val loss 均值 Δ=−0.001(远小于 std 0.0026,符号不一致);lob_bench 总分均值 Δ=+0.008(略差,符号摇摆)。
- **§3.4g 的 inter_arrival 优势是噪声且已反号**:放开 gamma 后 3 个 seed 上 graph 在 inter_arrival **全部更差**(+0.050/+0.002/+0.030)。单 seed 的 0.592 vs 0.666 是偶然。
- gamma 自发收敛到 ~0(±0.01、符号不定)→ 模型主动放弃图耦合。**图这一层在此设置下是 dead weight。**
- 行动项:在当前 BTC+ETH / 5ep 规模下不必带 graph;若仍要论证图的价值,需要换**更强 cross-asset 信号的标的/任务**(强 lead-lag 对)、**更长训练**、或**结构性更强的耦合**(非残差小门控)——属研究方向,非本基线的提升点。ckpt:`*_{ng_s,g_s}{1234,42,7}{_,_rg_}ms.ckpt`;lob_bench 输出 `data/lobbench/ms/<tag>/`。

### 3.4i 评测真实化(任务桶1完成,2026-06-11):自回归 rollout + 撮合引擎

新建 `DeepMarket/lob_engine.py`(独立 price-time 撮合引擎)+ `DeepMarket/build_lobbench_eval_ar.py`(AR rollout 评测构建器),替换 teacher-forced 单步 `build_lobbench_eval.py`,完成 `NEXT_WORK_ZH.md` 任务桶1(§1.1/1.2/1.3)。

**方法**:
- **(1.1)自回归**:目标资产每生成一笔订单 → 解码 → 喂入撮合引擎 → 用引擎重建的盘口作为下一步条件;对手资产保持真实(graph 近恒等 gamma~0,影响可忽略)。**写给 lob_bench 的 gen 盘口 = 引擎重建**,故 spread/imbalance/volume/ofi 真正是 gen 消息的函数,不再恒等真实盘口。
- **(1.2)order_id**:引擎给挂单分配 id,撤单/成交引用被击中挂单的 id 与价格(LOBSTER 约定:delete/exec 消息带被影响订单的价)。real/gen 走**同一引擎、同初始真实盘口播种** → 盘口差异只反映消息差异,无 real/gen 流程偏置。
- **关键数据事实**:crypto adapter 的 **depth 列恒 0** → 不能用 ABIDES 的 depth-relative 定价;改用**绝对价解码 + tick 量化**(BTC 0.1,ETH 0.01,从相邻档位间隔自动估计 `10**round(log10(median_gap))`)。**DDIM 10 步**推理让 500 步 sequential rollout 在分钟级跑完(DDPM 100 步会慢 10×)。
- **(1.3)覆盖**:`ASSET` 切 BTC/ETH;**8 个独立窗口 × 500 笔** → lob_bench 把每个窗口当一个 period → 跨窗口 **bootstrap CI**(`L1_CI` 列)。

**结果**(gen vs real,8 窗口×500 笔,DDIM-10,ckpt `ng_s1234_lr3e4_e5`,`data/lobbench/ar/<asset>/`):

| 指标(L1,越低越像真实) | BTC | ETH |
|---|---|---|
| **MEAN(全 21 项,全部 <1.0)** | **0.456** | **0.482** |
| spread | 0.979 | 0.976 |
| orderbook_imbalance | 0.242 | 0.265 |
| ofi / up/stay/down | 0.43/0.29/0.43/0.39 | 0.47/0.47/0.44/0.47 |
| ask/bid_volume | 0.56/0.40 | 0.49/0.57 |
| vol_per_min | 0.341 | 0.390 |
| **log_time_to_cancel**(曾退化 1.0) | **0.559** | **0.520** |
| limit_ask/bid_order_depth | 0.25/0.84 | 0.20/0.86 |
| **ask/bid_cancellation_depth**(ask 曾退化) | 0.81/0.69 | 0.83/0.74 |
| limit_ask/bid_order_levels | 0.13/0.13 | 0.11/0.07 |
| **ask/bid_cancellation_levels**(曾退化 1.0) | **0.18/0.14** | **0.15/0.06** |
| log_inter_arrival_time | 0.80 | 0.90 |

**结论**:
- **三类退化全部消除**:此前 teacher-forced 下 book-only(spread/imbalance/volume/ofi)恒等 0(假)+ 3 个 order_id/单边退化指标恒 1.0。现在 **21/21 全部给出有意义 <1.0 分数**(MEAN(all)=MEAN(non-degen),两资产)。**1.1/1.2/1.3 验收全部达成。** 生成双边(buy-frac BTC 0.51 / ETH 0.52)。
- **新短板浮现 = spread(~0.98)+ bid 侧 depth(~0.85)**:AR rollout 下 gen 盘口随时间变宽——模型订单类型配比 limit 0.18 / cancel 0.48 / exec 0.33,**净抽流动性 → 盘口变稀 → spread 分布偏离 real**。这是"让 book 指标变真"后**暴露的新建模弱点**(类型配比 / 盘口补充),取代旧的"只有 inter_arrival 短"叙事。可在 §2 改进里加盘口守恒/类型先验一并治。
- inter_arrival 仍偏高(0.80/0.90),与历史一致;时序仍是短板。
- **⚠️ 协议变更,不可与 §3.4e/g/h 旧数字直接对标**:旧 = teacher-forced 单步 + 共享真实盘口 + 单窗 800;新 = AR rollout + 重建盘口 + 8 窗×500 + DDIM-10。旧 MEAN≈0.21 是"只测订单边缘 + book 恒 0 拉低均值"的产物;新 MEAN≈0.46-0.48 是**全 21 项都真正参与**的结果,数值更高但**信息量完全不同(更诚实)**。

**新增/改动脚本**(NAS + 本地 `_c38/`):`lob_engine.py`、`build_lobbench_eval_ar.py`(env:`CKPT/OUT/ASSET/N_EVENTS/WINDOWS/STRIDE/SAMPLING/DDIM_NSTEPS/SIZE_SCALE/TICK/SEED`)、`score_lobbench.py`(资产标签自动)。旧 `build_lobbench_eval.py` 保留。

### 3.4j 对标靶子:TRADES 原论文数字(任务桶2 §2.3 完成,2026-06-11)

来源 arXiv **2502.07071**(Berti/Prenkaj/Velardi,*TRADES: Generating Realistic Market Simulations with Diffusion Models*),经 ar5iv HTML 抽取(引用前建议再对 PDF Table 1 核对一遍)。

- **数据/协议**:LOBSTER NASDAQ **TSLA + INTC**,2015-01-02~30;train 17 天 / val 1 天 / test 2 天(01-29、01-30);共 ~24M 订单事件。评测用 **ABIDES world-agent**:前 15min 真实 market replay 暖启,之后扩散模型接管 10:00–12:00;**每 2h 平均生成 ~50,000 笔**;扩散 **100 步**;RTX3090 上每模拟 1h ≈ 6h 算力。
- **主指标 = "predictive score"(MAE,越低越好)**:在合成数据上训练一个价格预测模型,到真实数据上测的 MAE。**这不是 lob_bench 的 L1/Wasserstein**,是下游任务误差。Table 1(两天均值):

  | 方法 | TSLA | INTC |
  |---|---|---|
  | Market Replay(真实上限) | 0.923 | 0.149 |
  | IABS | 1.870 | 1.866 |
  | CGAN | 3.453 | 0.699 |
  | **TRADES** | **1.213** | **0.307** |

  → 文中称对 SoTA "×3.27 / ×3.48 改进"(TSLA/INTC)。
- **分布覆盖**(PCA 凸包交集占真实分布比):TRADES 67.04% > CGAN 57.49% > IABS 52.92%。
- 论文**没有** inter-arrival / 订单时序的单独指标表(只有 stylized facts:收益自相关、波动聚集)。
- ⚠️ **不可与我们的数字直接对标**:(1)指标不同——他们是 downstream-task MAE,我们是 lob_bench 分布 L1;(2)资产不同——他们 LOBSTER 股票,我们 Tardis crypto;(3)协议不同——他们 ABIDES 2h rollout ~50k 笔,我们引擎 AR 8 窗×500。论文数字只能当"扩散方法在该范式下能做到接近 Market Replay"的定性参照,**不是我们 MEAN≈0.46 的可比基线**。真正可比的同协议靶子需在我们自己的 lob_bench 上跑 CGAN/Market-Replay baseline(§2 可选)。

### 3.4k 共享权重(P0)价值:数据效率头对头(任务桶2 §2.1,2026-06-11)

**问题**:多资产共享主干到底有没有用?设计成"**共享主干能否救一个数据贫瘠的资产**"(论文真正的多资产卖点;图已废)。

**方法 / 关键数据技巧**:`MultiAssetLOBDataset` 把所有资产按行对齐并截到 min_len,所以不能简单"少给 ETH 几行"(会把 BTC 也截短)。改用 **tiling**:把 ETH 的前 p% **唯一**行平铺回满长 255k(`data/ETHUSDT_p20`=51k唯一×5、`ETHUSDT_p10`=25.5k唯一×10;val/test=满量真实 ETH)。于是 ETH 信息量贫瘠、BTC 仍满量且对齐。同一个 tiled 文件同时喂独立 TRADES 和共享 MA,两者 ETH 数据完全一致,唯一差别 = 共享主干是否也训练满量 BTC。矩阵:**独立 TRADES-ETH** {full,p20,p10}×seed{1234,42} vs **共享 MA(BTC满+ETH-p)** no-graph;B_full=复用 `ng_s{1234,42}_ms`。LR3e-4/5ep/DISABLE_GRAPH=1。

**评测 1(主,teacher-forced 下一笔预测误差,`eval_p0.py`,对两种引擎用同一度量;val_ema 跨引擎不可比所以另建)**——ETH 测试集 3000 位置、同一 ESEED、z 空间 L1:

| ETH 数据 | 独立 TRADES `cont_l1` | 共享 MA `cont_l1` | 独立 `l1_price` | 共享 `l1_price` |
|---|---|---|---|---|
| full | 0.188 | 0.196 | 0.027 | 0.027 |
| p20 | **0.237** | **0.197** | 0.143 | **0.029** |
| p10 | **0.236** | **0.191** | 0.153 | **0.030** |

→ **独立 TRADES 随 ETH 数据缩小退化 ~26%,且退化几乎全在 price**(l1_price 0.027→0.15,5–6× 恶化);**共享主干把数据贫瘠资产钉在满量水平**(cont_l1 全程 ~0.19,l1_price 全程 ~0.03)——靠 BTC 的"价格相对盘口"结构借力。满量时共享几乎不损失(0.196 vs 0.188)。**P0 卖点成立(预测度量,2 seed 一致)。**

**评测 2(lob_bench AR,canonical,单 seed 1234)**:MEAN(all) A/B = full 0.433/0.466、p20 0.450/0.522、p10 0.477/0.474——**聚合 MEAN 看不出共享优势**(被 spread~0.9 / inter_arrival~0.9 / volume 等 rollout 动态项主导,price 占比小;且 B_p20 的 limit_ask_order_depth=0.82 是单 seed 离群把 B_p20 拉高)。但**订单放置的 bid 侧 depth 指标支持 rescue**:`limit_bid_order_depth` 独立 0.48→0.93→0.96(full→p20→p10)vs 共享 0.62→0.82→0.70(p10 共享 0.70 ≪ 独立 0.96);`bid_cancellation_depth` p10 共享 0.73 vs 独立 0.95。

**评测 2b(2 seed lob_bench,2026-06-11 补)**:加跑 seed42 后,**聚合 MEAN(all) 仍不分**(A/B:full 0.463/0.464、p20 0.472/0.499、p10 0.485/0.489——共享在 p20 反而略差,2 seed 仍是噪声);但**价格放置指标 `limit_bid_order_depth`(2 seed 均值)确证 rescue**:full A/B=0.638/0.697(满量独立更好,无干扰收益),p20=0.966/**0.826**、p10=0.966/**0.751**(数据贫瘠时共享显著更好)。即"共享救数据贫瘠资产"在**放置维度**上 2 seed 稳健,在 21 项聚合上被 spread/timing 噪声淹没。

**结论(诚实,2 seed 定稿)**:**共享主干对数据贫瘠资产的救援是真实的,集中在"价格相对盘口的放置"维度** —— teacher-forced 预测度量(l1_price)干净一致 + lob_bench `limit_bid_order_depth` 2 seed 确证;但 **lob_bench 21 项聚合 MEAN 看不出**(price 占比太小、聚合被 spread/timing/volume 主导)。满量时共享几乎不损失甚至略让(无干扰)。脚本:`eval_p0.py`、`build_lobbench_eval_ar_p0.py`(MODEL 切换,支持单资产 TRADES)、`_scripts/{p0_train,p0_eval,p0_lobbench}.sh`(`p0_lobbench.sh` 带 `S` env 跑多 seed);launcher 加了 `VAL_CHECK_INTERVAL/LIMIT_VAL_BATCHES` 钮。

### 3.4l 攻时序 inter_arrival:时间通道加权(任务桶2 §2.2,2026-06-11)

**做法(option a)**:在 `ma_gaussian_diffusion._mse_loss_per_asset` 的 epsilon-MSE 里给**时间通道(idx 0)加权**(`TIME_LOSS_W` env,sqrt 进 L2 norm;默认 1.0=baseline 不变,所以现有 `ng_s*_ms` 即 baseline)。treatment `TIME_LOSS_W=5`,2 seed,MA no-graph BTC+ETH 满量。

**结果(ETH)**:

| 度量 | baseline(TW=1) | treatment(TW=5) |
|---|---|---|
| `l1_time_z`(eval_p0,teacher-forced,2 seed 均值) | 0.272 | **0.252**(↓~7%) |
| 其他字段(l1_price/l1_size/cont_l1) | 0.028/0.30/0.196 | 0.029/0.30/0.193(未损,略好) |
| **lob_bench `log_inter_arrival_time`(AR,seed1234)** | **0.886** [.885,.906] | **0.939** [.931,.946](**更差**,CI 不重叠) |
| lob_bench MEAN(all) | 0.466 | 0.517(更差) |

**做法(option b,后续)**:**log-time 参数化**——时间列(idx 0,原始 inter-arrival)先 `log1p` 再 z-score(`build_logtime_data.py` → `data/{BTCUSDT_lt,ETHUSDT_lt}`),重尾→近高斯。MA no-graph 训练(2 seed),AR 评测加 `LOGTIME=1`(col_stats 对时间列 log1p、解码 `expm1`)。逐笔解码后的 gen inter-arrival 与 real 贴合(0.052 vs 0.054)。

**两种做法的 lob_bench(ETH,AR,inter_arrival 越低越好)**:

| 模型 | log_inter_arrival_time | MEAN(all) |
|---|---|---|
| **baseline**(TW=1,ng_s1234) | **0.886** | 0.466 |
| option a:TIME_LOSS_W=5(s1234) | 0.939 | 0.517 |
| option b:log-time(s1234 / s42) | 0.941 / 0.931 | 0.490 / 0.463 |

**结论(诚实,两个 lever 都 negative)**:
- option (a) MSE 时间加权:teacher-forced 点时间误差 ↓7%(2 seed、不伤其他字段),但 AR 分布 inter_arrival **更差**(0.886→0.939,CI 不重叠)——MSE 重加权治不了分布/重尾。
- option (b) log-time 参数化:模型有效(MEAN 0.46–0.49 ≈ baseline),但 inter_arrival **0.93–0.94 ≈ baseline 0.886、并未改善**(略差)。
- **inter_arrival(~0.9 L1)对"损失加权"和"边缘分布对数重参"都鲁棒不变**。注意:AR rollout inter_arrival 0.9 ≫ teacher-forced 0.66(§3.4e)≫ floor 0.064 —— 缺口主要在 **AR 序贯时间生成**(逐笔看着对、整段分布偏),不是边缘分布形状,所以 (a)/(b) 都治不了。

**做法(option c,2026-06-11 补,✅ 唯一奏效)**:**条件神经点过程时间头**(`pp_model.py`:GRU over 最近 K=32 笔 (log1p dt, etype) → log-normal 混合,NLL 训练;每资产一个)。AR 评测加 `PP_CKPT` env:**时间字段由点过程采样、其余字段(etype/size/price/dir)仍由扩散生成**,逐笔把生成的 (dt,etype) 喂回点过程历史。不动扩散模型(2.1/桶1 结果不受影响)。

| inter_arrival(ETH,AR) | baseline | (a) TW=5 | (b) log-time | **(c) point-process** |
|---|---|---|---|---|
| **L1** | 0.886 | 0.939 | 0.93–0.94 | **0.840**(s1234 0.845 / s42 0.835) |
| **Wasserstein** | 1.206 | — | — | **~0.50**(0.545 / 0.452) |

→ **点过程头是唯一降 inter_arrival 的 lever,2 seed 一致**:L1 0.886→0.840(~5%),**Wasserstein 1.21→0.50(2.4×,分布距离大幅改善)**。代价:`vol_per_min`(时间耦合指标)变差(0.48→0.55),故 lob_bench MEAN ~持平(0.466→~0.477)。

**结论(诚实,定稿)**:
- option (a) MSE 时间加权:teacher-forced 点时间误差 ↓7% 但 AR 分布 inter_arrival **更差**(0.886→0.939)。
- option (b) log-time 参数化:模型有效但 inter_arrival **不变**(~0.93)。
- **option (c) 点过程头:✅ inter_arrival L1 0.886→0.840、Wasserstein 1.21→0.50(2 seed 稳)** —— 证实"问题在序贯时间生成机制,需用点过程而非损失/参数化去治"。代价是 vol_per_min 等时间耦合项,需联调(如点过程与扩散联合训练 / 共享上下文)。**任务桶2 §2.2 用 option (c) 取得正向进展(分布层面显著);后续可把点过程头并入主干联合训练以避免 vol_per_min 退化。** 全部补丁/数据可逆(`TIME_LOSS_W` 默认 1.0、`PP_CKPT` 不设即原行为、`*_lt` 与 `pp/` 为新增,不影响其他实验)。

### 3.4m 训练稳定性:warmup+cosine vs mean-loss(任务桶3 §3.2,2026-06-13)

**问题**:§3.4f 定的"LR=3e-4 稳、1e-3 种子不稳"是绕过而非根治。桶3 §3.2 给的两条 lever:(a) LR warmup+cosine 调度、(b) `_mse_loss_per_asset` L2-norm→mean。目标:能用回 LR≥1e-3 且种子稳。

**做法(env-gated,默认=旧行为,全部可逆)**:`ma_gaussian_diffusion._mse_loss_per_asset` 加 `MSE_REDUCE`(默认 `norm` = `torch.norm(p=2)`,`mean` = 每资产 mean-of-squares,把 loss 尺度从 ~√(K·F) 压到 O(1));`ma_diffusion_engine.configure_optimizers` 加 `LR_SCHED`(默认 `plateau` = 旧的 on_validation 手动砍半,`cosine` = 线性 warmup(`WARMUP_FRAC`=0.05)后 cosine 衰减,逐 step,且跳过手动砍半)。备份 `*.bak_stab`。补丁脚本 `_c38/patch_stability.py`,驱动 `_c38/stab_{ab,r3}_driver.sh`。MA no-graph BTC+ETH,5ep,LR=1e-3。

**关键方法点**:`MSE_REDUCE=mean` 改变 val_ema **量纲**(mean≈norm/√(K·F)),故 mean 跑出来的 val 数字小是假象,**不可与 norm 的 val 横比**。公平横比一律用 `eval_p0`(teacher-forced 逐笔预测 z 空间 L1,与 loss reduction 无关),ETH(BTC peer)+ BTC(ETH peer),N_POS=3000,DDIM-10。

**结果(`eval_p0` cont_l1,越低越好;val_ema 仅同量纲内可比)**:

| 配方 | LR | reduce | sched | val_ema | ETH cont_l1 | BTC cont_l1 | seeds |
|---|---|---|---|---|---|---|---|
| anchor(旧默认) | 3e-4 | norm | plateau | 0.963 | 0.221 | 0.253 | 1 |
| B | 1e-3 | norm | plateau | 0.927 | 0.186 | 0.231 | 1 |
| A | 1e-3 | mean | plateau | →1.5e16 | 6.33 | 6.40 | 1(发散) |
| T | 1e-3 | mean | cosine | 0.145* | 0.230/0.238 | 0.275/0.281 | 2 |
| **nc(推荐)** | 1e-3 | norm | **cosine** | **0.897/0.898** | **0.185/0.185** | **0.225/0.220** | 2 |

\*mean 量纲,不可比。

**结论(诚实,定稿)**:
- ✅ **cosine warmup 是对的 lever**:`nc`(norm+cosine,LR=1e-3)val_ema 0.897(< B 0.927 < anchor 0.963),cont_l1 ETH 0.185 / BTC 0.22 —— **追平历来最好的 B、并优于旧 3e-4 默认(0.221/0.253)**;**2 seed 极稳**(ETH cont_l1 0.1849 vs 0.1853,差 0.0004;val 0.897/0.898);且**从 epoch 0 平滑下降,没有 B 在 1e-3 下头 ~1.5 epoch 的瞎晃**(B:5.31→4.87→2.47 才被手动砍半救回)。**桶3 §3.2 目标(用回 LR≥1e-3 且种子稳)达成。**
- ❌ **mean reduction 是错的 lever**:`T` 的 val 0.145 是量纲假象,**实测 cont_l1 反而更差**(ETH 0.234 vs norm 0.185,BTC 0.278 vs 0.22)——mean 只让 price 通道略好(l1_price 0.023 vs 0.026),size/time 更差,净更差;`A`(mean+旧 plateau)直接**数值发散到 1.5e16**(固定 0.002 砍半阈值对 mean 小尺度误触发/失配)。**这推翻了 memory 里"L2-norm loss scale 是种子敏感主因"的猜测**——真因是缺 warmup,不是 loss reduction。
- 📌 **附带发现**:B(norm+plateau@**1e-3**)cont_l1 0.186 比 anchor(norm+plateau@**3e-4**)0.221 **更好**——起始 LR 高 + 衰减比固定低 LR 找到更好的解(anchor 仅 n=1,nc 的 2 seed 0.185 已坐实此结论)。
- **推荐默认**(未改 launcher 默认值,保持可复现;新跑显式带):`MSE_REDUCE=norm LR_SCHED=cosine LR=1e-3`。ckpt:`*_c38_{nc,mc,base,mean}_lr1e3_s{1234,42}.ckpt`;eval `AIQuant_logs/stab_{eval,nceval}.log`。
- **lob_bench AR 复核(nc s1234,8窗×500,DDIM-10,vs §3.4i 旧 baseline ng_s1234)**:**ETH MEAN(all) 0.482→0.444(真实改善 ~0.04)**,BTC 0.456→0.461(持平)。改善来自 spread(ETH 0.976→0.947)、imbalance(0.265→0.220)、**vol_per_min(BTC 0.341→0.243、ETH 0.390→0.238 两资产大降)**;**inter_arrival 略差**(BTC 0.80→0.90、ETH 0.90→0.92,此次未挂点过程头,时序仍短板)。即 **nc 配方在分布层面至少不亏、ETH 小赚**,稳定性增益渗到部分盘口动态;但单训练种子,ETH 的 0.04 建议多 seed 复核。输出 `data/lobbench/ar_nc/<asset>/`,驱动 `_c38/stab_lobbench_nc.sh`。

### 3.4n 结构化多资产数据:管线 + lead-lag 验证(任务桶4 数据部分,2026-06-13)

桶4 §4.2(三方案数据 + 两前置小工具)落地为可复现工具(本地单测 + 真 Tardis 端到端通过)。本地 Mac 写代码 commit/push,采集在 cluster38(经 `~/.aiquant_c38.sh` wrapper;直连 SSH 仍 publickey 拒)。

**新增/改动(均已入 git):**
- `preprocessing/lead_lag.py` — 滞后互相关(LOCF 规则网格 mid 对数收益)+ Hayashi-Yoshida(异步原生)+ HRY 平移 lead-lag 扫描。CLI:`python -m preprocessing.lead_lag A.npy B.npy --grid-ms 10 --max-lag-ms 500`,输出 verdict JSON。
- `preprocessing/structured_universes.py` — 三方案命名注册表(`btc_spot_perp`/`eth_spot_perp`/`btc_cross_exchange`/`qqq_basket`),每条含 `AssetUniverse` + 数据源 `LegSpec`(单一真相源)。
- `preprocessing/AssetUniverse.py` — 加 `basis_pair`(spot/perp、跨所)、`cross_venue`(N 资产、N−1 basis 组、全连接图)、`etf_basket`(NAV 组)三个构造器 → 填 `etf_basket_weights`(=`spread_groups`)+ `relation_types`。
- `preprocessing/adapters/iex_or_databento_adapter.py` — 加 Tardis 原生列识别(`asks[N].price`/`asks[N].amount`/`bids[N].*`,0-indexed);`timestamp` µs 已由 `timestamps_to_seconds` 处理。
- `preprocessing/build_structured_pairs.py` — 下载 Tardis 日样本 → adapter → 可选 LOCF 时间对齐(补 DATA_SOURCES §3.4 的对齐缺口)→ 写 `_adapter_raw/<asset>.npy` + `.t.npy` 绝对秒 sidecar + `<name>.leadlag.json`。产物即 `build_real_datasets.py` 输入。
- `tests/structured_data_smoke.py` — 无网络单测(Tardis 列 / 三 universe spread_groups / 已知 lag 复原)。

**真数据结论(Binance BTCUSDT,2024-01-01,20 万行)**:
| 度量 | grid xcorr (10ms) | Hayashi-Yoshida |
|---|---|---|
| perp vs spot 相关 | 0.081 | **0.599** |
| lead-lag | **perp 领先 10ms** | (HRY 扫描同向) |

- ✅ **方案 A 结构成立**:perp 领先 spot ~10ms,符合"价格发现在衍生品端"。**HY(0.60)≫ grid xcorr(0.08)** —— 紧套利的同币种 spot/perp 在粗网格上被微结构噪声淹没,必须用异步原生的 HY 才看得清(本工具两个估计都给,正是为此)。
- ⚠️ **网格分辨率坑**:100ms 网格会把 perp/spot 误报成 `synchronous`(lead-lag 是 sub-100ms);故 build 默认在 **event-native tick + 10ms 网格**上测 lead-lag,**不**在对齐后的 100ms 数据上测。
- **数据事实复核**:Tardis spot=`binance`、perp=`binance-futures`,同日同币;免费样本 = 每月 1 号全天(GET 可下,**HEAD 会误返 404**——之前误判"URL 失效"的坑)。

**未做(属桶4.3,非数据部分)**:实际训练头对头、`run_ma_c38.py` 加 `UNIVERSE` 钮、跨资产一致性指标③、no-arb finite guard + 放开 gamma。

### 3.4n-扩展 三方案数据全部落地(2026-06-13,扩量 + 付费 C)

在 §3.4n 工具基础上,按"多抓几个月 / 干净 USDT 跨所 / 用 Databento 拉 C"扩展,**三方案训练就绪数据全部在 cluster38**(均 100ms 对齐 + z-score + split + 多日 lead-lag 验证):

| 方案 | 资产 | train 行/资产 | 结构(lead-lag,多窗) |
|---|---|---|---|
| A | BTCUSDT perp/spot | **909k** | perp 领先 spot,HY 0.33–0.82(6 月全显著) |
| A | ETHUSDT perp/spot | **890k** | perp 领先 spot,HY 0.29–0.86(6 月全显著) |
| B | BTC binance/okex | **1.33M** | **okex 领先 binance**(负 lag 一致),HY 0.25–0.63(6 月全显著) |
| C | QQQ+AAPL/MSFT/NVDA/AMZN | **1.44M** | QQQ↔成分同期 corr~1.0、HY~1.0(NAV 共动);index/成分无干净亚秒 lead-lag(正常) |

- **工具新增**:`build_structured_pairs.py` 加 `--months YYYY-MM:YYYY-MM`(抓多个月 1 号免费样本 → 按资产拼接,逐日 lead-lag);`structured_universes` 加 `btc_cross_usdt`(binance+okex,**同 USDT 报价无污染**)、`btc_cross_usdt3`(+bybit,练 N>2 图);`preprocessing/fetch_databento.py`(方案 C:**先 get_cost 估价、仅 `--pull` 才下载**,LOCF 对齐篮子 + lead-lag)。
- **A 多月**:2024-01~06 各月 1 号,perp 领先 spot 在 **6 个月全部显著**(HY 跨月 0.29–0.86)→ 结构稳健,非单日偶然。
- **B coinbase 报价坑解法**:coinbase 是 BTC/USD,与 binance/okex(BTC/USDT)做 basis 会混入 USDT/USD 漂移。解法=**改用同 USDT 的 binance+okex(已采,干净)**;或拿 USDT/USD 参考序列把 coinbase 价换算后再差分;或注意 **lead-lag 用对数收益近似与报价无关**(污染只在价位 basis,不在收益)→ coinbase 仅 no-arb basis 项有问题,图的 lead-lag 信号不受影响。`btc_cross_exchange`(含 coinbase)保留但已标注。
- **C Databento**:mbp-10、XNAS.ITCH、QQQ+4 成分;**实拉 3 个跨 regime 交易日(2024-03-01/06-03/09-03),实际花费 $5.50**(get_cost 估价一致;$1.5–2.5/日,$125 预算用 4%)。注意:① order 仍是 mbp-10 proxy(真逐笔需 mbo + 新 adapter);② 含盘前盘后(~15h/日),NAV 在 RTH 最干净,后续可裁到 RTH;③ **篮子权重 wᵢ 目前是近似值(0.09/0.08/0.08/0.05),正式 no-arb 实验前需换 QQQ 官方持仓权重**;④ databento 0.79 `to_df` API 是 `price_type=` 不是 `pretty_px=`(已修)。cluster `deep_market` env 已 `pip install databento`;key 放在 cluster `DeepMarket/.env`(chmod 600,gitignore)。
- **GitHub 仍不通**(id_rsa 带密码、id_ed25519 未注册;直连 SSH 也拒)→ 代码改动靠 base64 经 `~/.aiquant_c38.sh` 传到 NAS;已 commit/push 到 GitHub(b4a1572)。

### 3.5 当前状态

- ✅ 环境打通；p0/p1 smoke + 单卡 MA_TRADES 训练在 GPU 上通过
- ✅ 真实数据已就位：`data/{INTC,BTCUSDT,ETHUSDT,AAPL_A,AAPL_B}/{train,val,test}.npy`
- ✅ E1（INTC+SYNTH）已跑并停止（合成数据，已被真实数据取代）
- 🟢 **E2b**（真实 BTC+ETH，graph ON，arb-guidance OFF，GPU0）+ **E3**（BTC+ETH，graph OFF，GPU1）正在双卡训练，5 epoch，0 报错
  - 日志：`AIQuant_logs/run_e2b_btceth_graph_noarb.log` / `run_e3_btceth_nograph.log`
- ⬜ 后续：E4（INTC+AAPL 真实股票）、E5 单资产 baseline、E7 lob_bench 评测；修 P3 guidance NaN；修 DDP 设备硬编码以做真数据并行

### 3.5 辅助脚本（均在 NAS 仓库，未入 git）

- `build_c38_data.py` — 真实 INTC（LOBSTER）+ 合成 SYNTH → npy
- `build_real_datasets.py` — 把 rsync 上来的 adapter npy（BTC/ETH/AAPL）z-score+切分 → 每资产 `{train,val,test}.npy`
- `run_ma_c38.py` — MA_TRADES 启动器（env 驱动，不改 WIP 配置）
- 原始 adapter npy 暂存：`data/_adapter_raw/`
