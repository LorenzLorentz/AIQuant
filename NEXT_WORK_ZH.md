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

> **✅ 完成(2026-06-11,见 EXPERIMENT_PLAN_ZH.md §3.4k)**:tiling 法做"ETH 数据贫瘠 / BTC 满量"的数据效率头对头(独立 TRADES vs 共享 MA no-graph,full/p20/p10×2seed)。**P0 卖点成立,集中在"价格放置"维度**:独立随数据缩小退化 ~26%(几乎全在 price,l1_price 0.027→0.15),共享主干钉在满量水平(cont_l1~0.19,l1_price~0.03);**2 seed lob_bench `limit_bid_order_depth` 确证 rescue**(p20/p10 共享 0.826/0.751 vs 独立 0.966),但 lob_bench 21 项聚合 MEAN 仍被 spread/timing 噪声淹没看不出。
> **✅ 完成(2026-06-11,§3.4l)**:option (a) 时间加权 + (b) log-time **都没降 inter_arrival**(~0.9);**option (c) 条件神经点过程时间头(`pp_model.py`)✅ 奏效**:时间字段由点过程采样、其余字段仍由扩散生成,**inter_arrival L1 0.886→0.840、Wasserstein 1.21→0.50(2 seed 稳)**;代价是 vol_per_min 等时间耦合项变差(MEAN ~持平),后续可把点过程头并入主干联合训练。
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
>
> **✅ §3.2 已完成(2026-06-13,见 `EXPERIMENT_PLAN_ZH.md` §3.4m)**;**§3.1 评估后判定为 data-blocked,未跑(理由见下)**。

### 3.1 【图】换图能起作用的战场 — ⚠️ data-blocked,未执行(2026-06-13)
> **判定**:§3.4h 已用 3 paired seed 坐实图在 BTC+ETH 上是 dead weight(gamma 自收敛到 ±0.01、符号 seed 间翻转)。本任务唯一能翻盘的前提是**换强 lead-lag 标的对**(指数/期货 vs 成分股、ADR、同交易所多市场),而这类数据尚未采集/归一化(`data/_adapter_raw/` 只有 BTC/ETH/AAPL,且 AAPL 与 crypto 无 lead-lag 关系)。在没有新数据前,纯改结构(更强耦合 / 跨资产辅助目标)极可能仍被多 seed null 淹没。**ROI 低,故不烧 A100;留作"先采 lead-lag 数据"的后续研究,不属本轮稳定性提升点。** 原始任务描述保留供参考。

- **判断**:当前 BTC+ETH 同步性不足以让跨资产耦合产生增益;残差小门控太弱;图只从去噪损失拿梯度,信号不够。
- **怎么做**:
  - **数据**:换**强 lead-lag 标的对** —— 指数/期货 vs 成分股、ADR、同一资产多交易所(此项也属 §1 数据采集;`data/_adapter_raw/` + `build_real_datasets.py` 是现成的归一化/切分管线)。
  - **结构**:用比 gamma 残差门控更强的耦合;或加**跨资产辅助目标**(如让图分支预测另一资产下一步),给它去噪损失之外的梯度。
  - **训练**:更长 epoch;别用 `GRAPH_LR=1e-5` 钉死 gamma。
- **验收**:在新标的/新结构上,graph−nograph 的 Δ 在多 seed 上**符号一致且超出方差**(对照 §3.4h 的判定标准)。

### 3.2 【稳定性】降种子敏感 — ✅ 已完成(2026-06-13,见 `EXPERIMENT_PLAN_ZH.md` §3.4m)
> **结论**:**(a) cosine warmup 是对的 lever,(b) mean reduction 是错的 lever**。推荐配方 **`MSE_REDUCE=norm LR_SCHED=cosine LR=1e-3`**:val_ema 0.897(< 旧 3e-4 默认 0.963)、`eval_p0` cont_l1 ETH 0.185 / BTC 0.22(优于旧默认 0.221/0.253)、**2 seed 极稳(差 0.0004)**、从 epoch 0 平滑下降无瞎晃 —— 用回 LR=1e-3 且种子稳,验收达成。
> - ❌ (b) mean reduction:val 小是量纲假象,实测 cont_l1 更差;`mean`+旧 plateau-halving 直接**发散到 1.5e16**(固定 0.002 砍半阈值对 mean 小尺度失配)。**推翻了"L2-norm loss scale 是种子敏感主因"的旧猜测——真因是缺 warmup。**
> - (c) `LossSecondMomentResampler` 冷启动:复查无碍(warmup 前返回 uniform,且已有 NaN-guard);非瓶颈,未改。
> - 补丁 env-gated 全可逆(`MSE_REDUCE`/`LR_SCHED` 默认=旧行为,backup `*.bak_stab`);脚本 `_c38/patch_stability.py` + `stab_{ab,r3}_driver.sh`;ckpt `*_c38_nc_lr1e3_s{1234,42}.ckpt`。**未改 launcher 默认值**(保持旧实验可复现),新跑显式带上述 env。
>
> 原始任务描述保留供参考:
> - **目标**:消除 0.94–0.96 间的 seed 抖动 + 历史上的 LR 敏感。
> - **怎么做**:(a) **LR warmup + cosine 调度**;(b) `_mse_loss_per_asset` 的 **L2-norm 改成 mean / 归一化**;(c) 检查 `LossSecondMomentResampler` 冷启动。
> - **验收**:固定超参下多 seed 的 val 方差下降;能用回 LR≥1e-3 而不发散。

---

## 4. 结构化多资产数据 → 激活图 & no-arb(2026-06-13 新增)

> **一句话动机**:桶3 把图(P1)和 no-arb(P3)判成"无收益/跑不通",但读了 `arbitrage/energy.py` 后看清——**这两套机制本来就是为"有经济结构的资产篮子"设计的,我们却一直拿 BTC+ETH 跑**,这对资产**既没有强 lead-lag、也没有套利关系**,所以两套机制都在空转(图 gamma 自收敛到 0;no-arb energy ill-posed → NaN → 一直 `DISABLE_ARB_GUIDANCE=1`)。**桶4 = 换上有结构的数据,给图和 no-arb 一个公平且必然有信号的战场**;这也是把整个工作从"增量 multi-asset"抬成"经济结构化多资产 LOB 生成 + 无套利引导"的关键一步(见桶3 后的讨论)。

### 4.1 当前数据的局限性(诚实盘点)

按"影响多严重"排序:

1. **【致命·针对图/no-arb】资产对没有经济结构**。BTC+ETH 是两个弱相关的独立币种:没有 lead-lag 主导关系(同步性来自共同市场 beta,不是一个领先另一个),更没有套利约束(不存在"BTC=f(ETH)"的定价关系)。→ 图的跨资产耦合**没有可学的有向信号**(§3.4h:gamma 多 seed 符号翻转、自收敛到 ±0.01);no-arb 的 `spread_groups(asset_universe)` **为空**(没有 ETF↔成分 / spot↔perp 关系可填)→ energy 退化/NaN。**这是桶4 要解决的头号局限。**
2. **【严重】order 流是"合成代理",不是真实逐笔**。我们的 crypto(Tardis)和股票(Databento mbp-10)都是**盘口快照**;order token 是用 `synthesize_orders_from_lob` 从快照差分**合成**的(见 `DATA_SOURCES_ZH.md` §"真·逐笔 vs 快照")。→ `cond_lob`(多档 book)是真实的,但**生成目标本身的 add/cancel/trade 序列是 proxy**,撤单/成交的微结构(order_id 关系、真实 cancel 行为)是我们在评测引擎里**补出来的**,不是数据里学到的。想要真·逐笔需 LOBSTER 或 Databento **`mbo`**(**目前没有 MBO adapter**,需新写一个)。
3. **【中】crypto adapter 的 `depth` 列恒 0**(桶1 §3.4i 发现):只能用绝对价解码 + tick 量化,不能用 ABIDES 的 depth-relative 定价。换数据时要复核新 instrument 的 depth 列是否同样退化。
4. **【中】覆盖面小**:Tardis 免费样本 = **每月 1 号全天**(单日);训练用的是单日 144 万行切分。跨日/跨 regime 泛化没测过。Databento $125 赠额 ≈ QQQ+4成分×3月(够,但要花)。
5. **【轻】尺度/归一化坑**:adapter 输出未 z-score(manifest `normalized:false`),训练前必须用**训练集统计量** z-score,否则静默学坏(已有 `build_real_datasets.py` 处理,换数据沿用)。

### 4.2 采集结构化数据(三方案,按性价比执行)

> 现成管线:`preprocessing/adapters/iex_or_databento_adapter.py`(L2/MBP→adapter schema)+ `build_real_datasets.py`(z-score + 切分 → `data/<asset>/{train,val,test}.npy`)。三方案都复用它,主要新增 = 拉数 + lead-lag 验证 + 填 `spread_groups`。

**方案 A(首选,零成本,信号最强):同币种 spot vs perpetual。** 例:`BTC-USDT spot` vs `BTC-USDT-PERP`(同所,如 Binance/OKX)。
- **为什么强**:永续/期货**领先**现货(价格发现在衍生品端);basis=(perp−spot)**均值回归到 ~0** = 一个**真·软套利带**,正好喂 no-arb energy(把 perp 当"ETF/NAV"、spot 当"成分",或反之;`spread_groups` 填这一对)。
- **怎么做**:Tardis 同时有 spot 和 perp 的月样本 → 各拉一份过 `iex_or_databento_adapter`(注意两边时间戳对齐、同一交易日)→ `build_real_datasets.py`。`DATA_SOURCES_ZH.md` 第 118 行已标"spot/perp basis 作 ETF-NAV 类比"。
- **风险/我倾向**:我**比较确信**这对能给图一个有向信号(lead-lag 确实存在);no-arb 的 basis 关系也确实存在但**带可不是常数**(资金费率、持有成本让 basis 有漂移),所以 energy 的 `delta` 用动态(`StressHead`)是对的,但要确认它别把正常 basis 漂移当违例。

**方案 B(零成本,次选):同币种跨交易所。** 例:`BTC@Binance` vs `BTC@Coinbase` vs `BTC@OKX`。
- **为什么**:跨所**价格相等**被套利者钉住(强 no-arb:三角/跨所);流动性最高的所通常**领先**。Tardis 覆盖 30+ 所。
- **怎么做**:同 instrument、不同 exchange 各拉一份;`spread_groups` 可填"两两价差→0"。三个所还能测**多于 2 资产**的图(目前只跑过 N=2)。
- **风险/我倾向**:跨所 lead-lag 比 spot/perp **弱且时变**(谁领先随时段变),图可能学到的是"对称耦合"而非"有向"。作为方案 A 的补充/稳健性检验更合适。

**方案 C(付费 $0–125,做股票主线 + 直接对标 no-arb 设计):ETF vs 成分股。** 例:`QQQ` vs `AAPL,MSFT,NVDA,AMZN`(top 成分)。
- **为什么**:`energy.py` 的 `relu(|ETF_mid − Σ wᵢ·成分_mid| − δ)²` **就是为这个写的** —— ETF mid ≈ 加权成分 mid(NAV)是教科书级套利约束。直接把 `spread_groups` 填 `(QQQ_idx, [(AAPL,w1),(MSFT,w2),...])`。
- **怎么做**:Databento `mbp-10` 一次拉 QQQ+成分($125 赠额够 ~3 月)。注意 order 仍是 proxy(mbp-10);若要真逐笔得上 `mbo`(新 adapter)。权重 wᵢ 用 QQQ 官方成分权重。
- **风险/我倾向**:这是**与 no-arb 模块最对口**的设定,审稿人也最认"ETF/NAV 套利"。但股票有**收盘/开盘、停牌、tick size 分层**等麻烦;且成分多(N 大)→ 图和 batch 变重。**我倾向先用方案 A 把机制跑通(便宜、干净),再用方案 C 做"上规模 + 经济学最硬"的主结果。**

**两个必做的前置小工具**(任一方案都要):
- **lead-lag 量化脚本**(训练前验证,别假设):对两资产 mid-price 对数收益做**滞后互相关**和 **Hayashi-Yoshida**(异步成交友好)估计领先滞后符号与量级(如"perp 领先 spot ~X ms,峰值相关 ρ")。**只挑确有显著、可观 lead-lag 的对进训练**。挑剩的记录下来(负结果也有用)。
- **填 `spread_groups`**:在 asset_universe 配置里声明 ETF↔篮子(或 perp↔spot)关系 + 权重,否则 P3 的 `group_rolling_stats` 返回空、energy=0。

### 4.2-DONE 数据部分已完成(2026-06-13,见 `EXPERIMENT_PLAN_ZH.md` §3.4n)

桶4 的**数据部分**(§4.2 全部:三方案管线 + 两个前置小工具)已落地为可复现工具,本地单测 + 真 Tardis 数据端到端验证通过:
- **前置小工具 1(lead-lag 验证)** = `preprocessing/lead_lag.py`:滞后互相关(规则网格)+ Hayashi-Yoshida(异步原生,无重采样偏差)+ HRY 平移扫描;CLI 输出 verdict JSON(leader / peak_lag_ms / corr / hy_corr / significant)。**真数据结论:Binance BTCUSDT perp 领先 spot ~10ms,HY 相关 0.60(grid xcorr 仅 0.08——证明此场景必须用 HY)→ 方案 A 结构成立、可进训练。** 网格 100ms 太粗会显示"synchronous",lead-lag 须在 event-native tick 上以 ≤10ms 网格测(已设为 build 默认)。
- **前置小工具 2(填 spread_groups)** = `AssetUniverse` 新增 `basis_pair`(A/B)、`cross_venue`(B,N>2,N−1 个 basis 组)、`etf_basket`(C,NAV 组);`preprocessing/structured_universes.py` = 三方案命名注册表(universe + 数据源 leg 单一真相源)。每个 universe 均产出非空 `spread_groups`(单测覆盖)。
- **方案 A/B 采集管线** = `preprocessing/build_structured_pairs.py`:下载 Tardis 日样本(GET;**注意 HEAD 会误报 404**)→ Tardis 原生列(adapter 已加 `asks[N].price/amount` 识别)→ top-10 → 可选**时间对齐**(LOCF 共享网格,补上 DATA_SOURCES §3.4 缺的对齐)→ 写 `_adapter_raw/<asset>.npy` + `.t.npy` 绝对秒 sidecar + lead-lag verdict。产物即 `build_real_datasets.py` 的输入(z-score+切分)。
- **方案 C(ETF/成分)**:universe(`qqq_basket`)+ 数据源 spec 已配,但 `free=False` → 不自动拉,需 Databento key + `get_cost` 审批后单独拉,再指向 `build_real_datasets.py`。
- 单测:`tests/structured_data_smoke.py`(无网络;adapter Tardis 列 / 三 universe 的 spread_groups / 已知 lag 的 lead-lag 复原)。
- **下一步(桶4.3,非数据部分)**:在 cluster38 上对 A/B 真数据跑 §4.3 的 2×2×多 seed 头对头(coupled vs independent × no-arb ON/OFF),配方 `MSE_REDUCE=norm LR_SCHED=cosine LR=1e-3`;`run_ma_c38.py` 需加 `UNIVERSE` 钮从注册表选资产。

### 4.3 跑图 + no-arb:期待收益、debug、找模型不足

**前置修复(必做)**:
- **no-arb NaN 守卫**:P3 在 BTC+ETH 上 NaN 的**表层**是数值(energy/guidance 里某项 inf/nan),**根层**是没有 spread_group。换上有结构数据后 energy 才有定义;但仍要在 `arbitrage/energy.py` + guidance 加 **finite guard**(`torch.nan_to_num` / clamp `decode_mid_price`、`rho` 的输入,给 `delta` 下限),并复用 `Sampler.py` 已有的 importance-sampler NaN-guard(桶2 已加)。**先单 batch smoke 确认 energy 有限、量级合理,再放进训练。**
- **释放 gamma**:别再用 `GRAPH_LR=1e-5` 钉死(那是 BTC+ETH 时代为防跑飞的权宜);用 `GRAPH_LR=3e-4`(=trunk LR,§3.4h 验证不会跑飞),让图在有信号时真能学。

**主实验(2×2×多 seed 头对头)**:
| 维度 | 取值 |
|---|---|
| 耦合 | **coupled**(图 ON,gamma 释放)vs **independent**(每资产独立 TRADES) |
| no-arb | **P3 ON** vs **OFF** |
| seed | ≥3(1234/42/7),配对 |
| 配方 | 桶3 的 **`MSE_REDUCE=norm LR_SCHED=cosine LR=1e-3`**(已确立的稳定配方) |

- **评测**:① `eval_p0`(逐笔预测,reduction-invariant);② **lob_bench AR**(分布);③ **新增·针对性指标 = 跨资产一致性** —— 生成 rollout 里 basis/NAV 偏离的分布,对比 real 和 independent-baseline(no-arb 该把它压向 real)。**第 ③ 个是这一桶的核心卖点指标,要专门写。**

**我期待 / 我倾向相信**:
- **图**:在 spot/perp 上**有可观正收益的概率中等偏上**——因为这里 lead-lag 是真的、有向的,图的 message-passing 有东西可传。但**我留个心眼**:残差小门控(`eps_fused=eps_local+gamma·MLP`)可能仍太弱,需要更强结构(让图分支预测对手资产下一步的**辅助损失**,给它去噪损失之外的梯度——§3.1 早就提过)。
- **no-arb**:在有真实套利带的数据上,**最可能赢在"跨资产一致性"指标(③)而非 lob_bench 总分**——类比桶2.1 共享权重的收益藏在 price-placement、被 21 项聚合淹没。所以**别只看 MEAN(all),要看 ③**。
- **失败模式预案(我观察到的同类陷阱)**:(a) 量纲/假象——像桶3 的 mean-loss,别被某个"看着小"的数字骗,一律用 reduction-invariant + 专项指标交叉验证;(b) 聚合淹没——真实收益常在单一维度,要拆指标看;(c) 数据无结构的"伪 null"——若图/no-arb 仍无收益,**先回去验 lead-lag 脚本确认这对数据真有结构**,再下"机制无效"的结论(别重蹈 BTC+ETH 的覆辙)。

**找模型不足(debug 路线,复用桶3 的诊断器)**:
- `diag_graph.py`(forward 切 disable_graph 看 delta 量级)、`diag_grad.py`(trunk 梯度 cosine ON/OFF)——确认图在新数据上**真的改变了 forward 且梯度有信号**(不是又退化成恒等)。
- 监控 gamma 轨迹:有结构数据上 gamma 该**稳定收敛到非 0**(对比 §3.4h 的符号翻转 ~0)。
- no-arb:打印每步 energy 量级 + guidance λ + violation 率,确认"ON 比 OFF 真的降低了 basis 违例"。
- 若图有效但不稳:接桶3 的 `LR_SCHED=cosine`;若 no-arb 数值脆:收紧 guard + 退火 schedule(`guidance_schedule.py`)调慢。

### 4.4 顺带把贡献抬上去(独立于数据,可并行)

- **点过程时间头并入主干联合训练**:桶2.2 的 `pp_model.py`(GRU→log-normal 混合,NLL)目前是**外挂**(评测时替换时间字段),导致 `vol_per_min` 退化。把它**焊进 `SharedScoreNet` 做联合训练**(时间用 TPP-NLL、其余字段用扩散去噪,共享上下文),消除外挂导致的时间耦合退化 —— 这本身是一个**可独立成立的方法贡献**(neural temporal point process + diffusion marks)。
- **强基线对标**:在我们自己的 AR lob_bench 协议上跑 **CGAN / Market-Replay** baseline(桶2.3 取的 TRADES 论文数字因协议不同不可直接比),给改进一个同协议靶子。

**验收(整桶)**:在一个**经过 lead-lag 验证的结构化资产对**上,拿到 **coupled+no-arb > independent** 的证据(至少在 `eval_p0` 或跨资产一致性指标 ③ 上多 seed 符号一致且超方差),并能说清"耦合/无套利在资产经济相关时才有用"。若结论仍是 null,则产出"受控的、带 lead-lag 验证的负结果",也是可写的科学结论。

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
