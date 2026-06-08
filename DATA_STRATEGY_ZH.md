# 低成本数据替代方案说明

本文说明在不购买 LOBSTER 多资产数据的前提下，如何保住当前项目主线：**多资产图耦合扩散模型 + spread-aware conditioning + quasi-no-arbitrage guidance**。核心原则是：不重写模型，不改变训练主接口，而是把可获得的数据源统一转换为现有 `MultiAssetLOBDataset` 能读取的 LOBSTER-like `.npy` 格式。

## 1. 核心判断

当前项目不能把成败押在付费 LOBSTER 数据上。论文原始设想使用同步的 ETF 与成分股 Level-3 order book 数据，这是理想设置，但数据成本和授权风险较高。

更稳妥的做法是把数据方案分成三层：

1. **权益主线**：优先尝试 Databento 或 IEX HIST，做小规模 ETF/成分股 cluster。
2. **公开 sanity check**：使用 FI-2010 验证 dataloader、训练循环、P0/P1 图耦合和 cross-asset metrics。
3. **兜底实验**：若权益 LOB 数据仍然不可用，切换到 crypto spot/perp 或 synthetic/ABIDES，把论文表述收缩为 LOB-style multi-market generation。

这样即使拿不到 LOBSTER，也能完成 P0/P1/P2/P3/P4 ablation，只是论文必须如实说明数据限制。

## 2. 统一数据接口

所有 adapter 的目标输出都是当前 DeepMarket 可读取的 `.npy`：

```text
[time, event_type, size, price, direction, depth,
 ask_price_1, ask_size_1, bid_price_1, bid_size_1,
 ...
 ask_price_10, ask_size_10, bid_price_10, bid_size_10]
```

也就是：

- `orders`: 6 列，形状为 `(T, 6)`
- `lob`: top-10 order book，40 列，形状为 `(T, 40)`
- 拼接后单资产数组形状为 `(T, 46)`
- 多资产训练时由 `MultiAssetLOBDataset` 读取多个资产 `.npy`，返回 `(B, N, K, F)` 风格的 batch

> **归一化注意**：adapter 输出的是与 `LOBSTERDataBuilder` 相同的**列布局**，但**没有做 z-score 标准化**（只按 `price_divisor` 缩放价格）。`LOBSTERDataBuilder` 的正式产物是 z-score 之后的数组，模型也是在 z-score 数据上训练的。因此 adapter 产物在喂入训练前**必须**先做 z-score（用训练集统计量），否则尺度不匹配会让模型静默学坏。每个 `.npy` 旁的 manifest 里记录了 `"normalized": false` 作为提醒。

已新增的 adapter 位于：

```text
DeepMarket/preprocessing/adapters/
```

其中：

- `lobster_adapter.py`：继续支持原始 LOBSTER/DeepMarket 路径。
- `iex_or_databento_adapter.py`：把权益市场 L2/MBP snapshot 转成 LOBSTER-like `.npy`。
- `fi2010_adapter.py`：把 FI-2010 feature-level 数据转成 sanity-check 用 `.npy`。
- `common.py`：统一校验、保存、split、manifest 和 snapshot-derived proxy order 生成。

## 3. 推荐数据路径

### 3.1 首选 A：Databento 小样本权益数据

Databento 当前提供美国股票与 ETF 历史数据接口，支持股票与 ETF、order book 相关 schema，并公开列出 `mbo`、`mbp-10`、`mbp-1` 等数据格式。对本项目来说，最务实的选择不是 full MBO，而是先用 `mbp-10` 做 top-10 book 实验。

建议范围：

```text
QQQ + AAPL + MSFT + NVDA
1-3 个完整交易日
schema 优先 mbp-10
```

优点：

- 最接近原始 ETF/成分股设想。
- 能保留“small economically meaningful ETF-constituent cluster”的论文主线。
- 数据格式和现有 top-10 LOB 接口比较贴近。

限制：

- 仍然可能有成本和 license 限制。
- 小样本实验不能声称覆盖完整 ETF basket。
- 如果只用 `mbp-10`，就不是 Level-3 order-by-order replay。

论文表述建议：

```text
We evaluate on a small economically meaningful ETF-constituent cluster
using top-10 displayed order book snapshots.
```

### 3.2 首选 B：IEX HIST 免费权益订单簿数据

IEX 官方说明 HIST 历史数据可在 T+1 基础上免费下载；IEX DEEP 提供 price-aggregated depth of book，DEEP+ 提供 order-by-order displayed resting orders。这个方案适合在预算极低时保留权益市场实验。

建议范围：

```text
QQQ + AAPL + MSFT + NVDA
或 SPY + AAPL + MSFT
5-10 个交易日
```

优点：

- 免费或低成本。
- 是真实权益市场订单簿数据。
- 可以支撑 P0/P1/P2/P3/P4 的小规模实验。

限制：

- IEX 是单交易所 book，不是 consolidated market book。
- displayed-only book 不能代表全部隐藏流动性。
- 若使用 aggregated depth，不能声称是完整 Level-3 message stream。

论文表述必须收缩为：

```text
IEX-specific displayed order book
```

不要写成：

```text
consolidated U.S. equity LOBSTER Level-3 data
```

### 3.3 公开 sanity check：FI-2010

FI-2010 是公开的 limit order book benchmark，包含 Nasdaq Nordic 5 只股票、10 个交易日、约 400 万个样本。它适合作为代码闭环和 sanity check，但不适合证明 ETF-NAV quasi-arbitrage。

可验证内容：

- dataloader 是否能读取多资产 `.npy`
- P0 shared backbone 是否能训练
- P1 graph coupling 是否有梯度
- cross-correlation / lead-lag metrics 是否输出 finite JSON

不能用它证明：

- ETF basket NAV gap
- 成分股套利关系
- Level-3 message replay realism

论文中应写清楚：

```text
FI-2010 is used only as a public sanity-check benchmark for LOB-style
multi-asset modeling and cross-asset metrics.
```

### 3.4 兜底：Crypto spot/perp 或 synthetic/ABIDES

如果权益订单簿数据仍卡住，可以切到 crypto spot/perp，例如 BTC spot/perp 或 ETH spot/perp。Binance public data 提供公开下载的日度和月度市场数据文件，适合快速构建低成本实验。

优点：

- 数据获取成本低。
- spot/perp basis 可以替代 ETF-NAV gap，作为 price-relation sanity metric。
- 适合验证 P2/P3 的 spread-aware conditioning 和 energy guidance。

限制：

- 论文标题和动机要收缩，不再强调 equity ETF LOBSTER。
- crypto microstructure 和股票 ETF basket 有显著制度差异。
- 若只有 trade/klines 而没有 depth snapshots，则不能声称 LOB 生成，只能作为弱兜底。

论文表述建议：

```text
LOB-style multi-market generation with spot-perpetual basis consistency.
```

## 4. 数据决策流程

建议把数据可行性判断限制在 48 小时内：

```text
第 0-1 天：
  尝试 Databento 或 IEX，获取 2-4 个 ticker 的至少 1 个完整交易日。

如果成功：
  走权益主线，先做 QQQ + 2-3 个成分股。

如果失败：
  立即切 FI-2010 sanity check + crypto/synthetic 兜底。

不要继续等待：
  不要因为 LOBSTER 未购买而阻塞模型训练、指标和论文 ablation。
```

最小训练闭环：

```text
raw data
  -> adapter
  -> per-asset .npy
  -> MultiAssetLOBDataset
  -> P0/P1/P2/P3 training smoke
  -> P4 metrics real-vs-real split
  -> ablation report
```

## 5. 当前代码使用方式

### 5.1 Adapter smoke test

```bash
cd DeepMarket
conda run -n deep_market python -m tests.data_adapters_smoke
```

该测试会验证：

- LOBSTER adapter 输出 `(T, 46)`。
- IEX/Databento snapshot adapter 输出 `(T, 46)`。
- FI-2010 adapter 输出 `(T, 46)`。
- 保存后的 `.npy` 能被 `MultiAssetLOBDataset` 正常读取。

### 5.2 主线 smoke tests

```bash
cd DeepMarket
conda run -n deep_market python -m tests.p0_smoke
conda run -n deep_market python -m tests.p1_smoke
conda run -n deep_market python -m tests.p2_smoke
conda run -n deep_market python -m tests.p3_smoke
conda run -n deep_market python -m tests.p4_smoke
```

通过这些测试后，说明新增数据层没有破坏：

- P0 shared score backbone
- P1 graph coupling
- P2 spread-aware conditioning
- P3 annealed energy guidance
- P4 posterior consistency metrics

## 6. 论文写作边界

数据替代方案的关键不是“装作有 LOBSTER”，而是把 claim 写准确。

如果使用 Databento/IEX top-10 depth：

```text
small multi-asset LOB-style cluster
top-10 displayed order book snapshots
ETF-constituent price-relation sanity metric
```

如果使用 IEX：

```text
single-venue IEX displayed order book
```

如果使用 FI-2010：

```text
public sanity-check benchmark
not used as ETF no-arbitrage evidence
```

如果使用 crypto spot/perp：

```text
spot-perpetual basis consistency
not equity ETF NAV arbitrage
```

必须避免的表述：

```text
full synchronized LOBSTER Level-3 ETF basket data
consolidated U.S. market order book
exact no-arbitrage enforcement
complete ETF NAV arbitrage experiment
```

除非这些数据和实验确实完成。

## 7. 最低可交付版本

即使没有购买 LOBSTER，项目仍然可以交付：

- P0：多资产 shared backbone diffusion。
- P1：图耦合 message passing。
- P2：spread-aware conditioning。
- P3：默认关闭、可选启用的 quasi-no-arbitrage energy guidance。
- P4：posterior consistency 和 cross-asset sanity metrics。
- 数据层：将 LOBSTER、IEX/Databento、FI-2010 转成统一 `.npy`。
- 论文实验：基于小规模真实数据或公开 sanity data 的 ablation。

关键是把贡献聚焦在方法和可复现实验闭环上，而不是把论文成败绑定到昂贵数据。

## 8. 参考链接

- Databento stock data: <https://databento.com/stocks>
- Databento pricing: <https://databento.com/pricing/>
- IEX market data / HIST: <https://www.iex.io/products/equities/market-data-connectivity>
- FI-2010 benchmark paper: <https://arxiv.org/abs/1705.03233>
- Binance public data: <https://github.com/binance/binance-public-data>
- LOB-Bench paper: <https://arxiv.org/abs/2502.09172>
