# 数据来源、代码支持与使用评估

> 配套文档：`DATA_STRATEGY_ZH.md`（策略）、`workflow.md`（D0 阶段）。
> 本文记录**实测结果**、**可选数据源清单**，以及对**数据使用方式的评估**。
> 最后更新：2026-06-08。

## 0. 统一数据契约

所有数据源最终都转成**每个资产一个 `.npy`**，每行 46 列：

```
[time, event_type, size, price, direction, depth,
 ask_px_1, ask_sz_1, bid_px_1, bid_sz_1, ... ask_px_10, ask_sz_10, bid_px_10, bid_sz_10]
```

- order 6 列 + lob 40 列；`event_type ∈ {0=limit, 1=cancel, 2=market/exec}`
- 多资产由 `MultiAssetLOBDataset` 读取，返回 `(N, K, F)`
- 旁挂 manifest 记录来源与 `"normalized": false`（**未做 z-score**）

代码位置：

| 文件 | 作用 |
|---|---|
| `preprocessing/adapters/common.py` | 校验 / 保存 / split / manifest / 快照合成 proxy order / 时间戳解析 |
| `preprocessing/adapters/lobster_adapter.py` | 原始 LOBSTER message/orderbook CSV |
| `preprocessing/adapters/fi2010_adapter.py` | FI-2010 特征矩阵 |
| `preprocessing/adapters/iex_or_databento_adapter.py` | L2/MBP 快照（列名自动推断 + 快照合成 order） |
| `tools/binance_recorder.py` | Binance 20 档深度实时录制器（输出即 adapter schema） |
| `tests/data_adapters_smoke.py` | 冒烟测试 |

---

## 1. 已实测的数据集 + 代码如何支持

| 数据集 | 实测内容 | 深度 | 代码路径 | 需要的预处理 | 结果 |
|---|---|---|---|---|---|
| **LOBSTER**（仅合成） | 合成 message/orderbook | 10 档 | `lobster_adapter`（复用 `preprocess_data`） | event_type `{1,3,4}→{0,1,2}`、价格 `/100` | ✅ smoke；**未跑真实 LOBSTER** |
| **FI-2010**（真实 / Kaggle） | `Test_NoAuction_DecPre_CF_7`：149×**55,478** | 10 档（DecPre 归一化） | `fi2010_adapter` | **需转置** `raw[:40].T`；快照合成 proxy order | ✅ `(55478, 46)` |
| **Binance bookTicker**（真实） | BTCUSDT 1 天：**1850 万行** | **仅 L1（4/40）** | `iex_or_databento`（列重命名） | `best_*→ask/bid_px/sz`、ms 时间戳 | ✅ 但 book 退化 |
| **Tardis 25 档**（真实 / 免费样本） | BTC+ETH 1 天：**144 万行/天** | **25 档（取前 10 → 40/40）** | `iex_or_databento` | `asks[i].*→ask_px_{i}`、µs 时间戳 | ✅ 双资产 |
| **Databento mbp-10**（真实 / **$0.067**） | AAPL 30min：**48.5 万行** | **10 档（40/40）** | `iex_or_databento`（**原生列名**） | `reset_index` 出 ts_event；tz-aware 时间戳 | ✅ 真·多档股票 |
| **Binance 录制器 depth20**（真实 / 实时） | BTC+ETH live 18s | **20 档（取前 10 → 40/40）** | `tools/binance_recorder` → `iex_or_databento` | 无（原生 schema） | ✅ 录→转→载 全通 |

> 实测过程中发现并修复 2 个真实 bug：① ms 时间戳被当成 µs（阈值低一个数量级）；② tz-aware `datetime64`（Databento 的 `ts_event`）导致 `timestamps_to_seconds` 崩溃。冒烟测试用整数时间戳，两个都没覆盖到。

**关键观察**：`levels_nonzero` 指标显示 Databento / Tardis / 录制器都是 **40/40**（满 10 档），而 Binance bookTicker 只有 **4/40**（退化成 L1）——这是选源的核心判据。

---

## 2. 可考虑的数据来源清单

### 2.1 股票 / ETF

| 源 | 付费 | 类型 | 深度 | 覆盖 / 量 | 可获取多少 | 接入 |
|---|---|---|---|---|---|---|
| **Databento `mbp-10`** | 用量付费（**$125 赠额**） | 事件级盘口快照 | 10 档 | 全美整合，多年 | $125 ≈ **4 标的 × ~76 交易日** / QQQ+4 成分 × ~3 月 | API key（已配 `.env`） |
| **Databento `mbo`** | 用量付费 | **真·逐笔消息(L3)** | 全深度 | 同上 | 比 mbp-10 贵（用 `get_cost` 估） | API key；**需新写 MBO adapter** |
| **IEX DEEP** | 免费 | 价位聚合深度 | 多档 | 单一交易所(IEX，~2–3% 量)，近 12 月 | 全市场 pcap **5.9 GB/天** | HIST 下载 + pcap 解析 + 重建 book |
| **LOBSTER** | 样本免费 / 学术付费 | **真·L3** | 1/5/10 档 | 样本=1 天 5 只 Nasdaq；多则付费 | 样本即取；ETF 需付费 | 下载 |
| **WRDS TAQ** | 学校订阅 | 报价 + 成交 | **仅顶档 NBBO** | 全美股（含 ETF），1993– | 看学校额度，量很大 | 机构登录导出 |
| **LSEG/Refinitiv Tick History — Market Depth** | 学校（若购） | 多档深度 | ~10 档 | 全球，1996– | 看学校 | 机构登录导出 |
| **AlgoSeek / Nasdaq TotalView-ITCH** | 学校（若购） | 全深度 | 全 | 全美 | 看学校 | 机构 |

### 2.2 加密

| 源 | 付费 | 类型 | 深度 | 覆盖 / 量 | 可获取多少 | 接入 |
|---|---|---|---|---|---|---|
| **Tardis.dev** | 免费样本 / 学术 **$650/月** | 快照 + 增量 L2 | 最多全深度（样本 25 档） | 30+ 交易所，多年 | 免费=**每月 1 号全天**；更多需付费/试用 | 直链 / API |
| **自录 Binance depth20** | **免费无限** | 20 档快照 @100ms | 20 档 | 任意币种，**向前采集** | 取决于跑多久（12h ≈ 数十万行/币） | `tools/binance_recorder.py` |
| **Binance public**（data.binance.vision） | 免费 | bookTicker=L1 / bookDepth=分桶 | L1 / 分桶 | 历史多年，海量 | 海量但浅 | 直链 |
| **Crypto Lake / Kaiko / Amberdata / CoinAPI** | 付费 | 全 L2 | 全 | 多交易所 | 付费 | API |

### 2.3 公开 benchmark

| 源 | 付费 | 类型 | 深度 | 量 | 接入 |
|---|---|---|---|---|---|
| **FI-2010** | 免费（Kaggle） | 特征矩阵 | 10 档（归一化） | 5 只 Nordic 股 × 10 天，~394k 事件 | Kaggle token（已配） |

---

## 3. 数据使用方式评估

### 3.1 快照 vs 逐笔消息（最关键）

模型本质是**订单消息生成器**（`event_type/size/price/direction/depth`）。但数据源分两类：

- **真·逐笔(L3)**：LOBSTER、Databento `mbo` —— order 列是真实的 add/cancel/trade 序列。
- **盘口快照**：Databento `mbp-10`、Tardis、FI-2010、自录 depth —— 我们用 `synthesize_orders_from_lob` 从快照差分**合成 proxy order**。→ `cond_lob`（多档 book）是真实的，但 **order 流是合成代理**，不是真实成交/撤单序列。

**结论**：
- 想要 order 生成的真实性 → 用 LOBSTER 或 Databento **`mbo`**（需补一个类似 `lobster_adapter` 的 MBO→`{0,1,2}` 适配器，**目前没有**）。
- 只关心多档 book 的条件与跨资产耦合（P1 图耦合 / P2 spread / `cond_lob`）→ 快照源已够用，proxy order 当弱监督。
- **改进点**：Databento `mbp-10` 每行其实**自带触发事件**（`action/side/price/size`），可据此生成更忠实的 order token，而不必从 book 差分硬合成——值得为 Databento 单独写一个 adapter。
- 论文措辞要诚实：快照源**不能**声称 L3 message replay realism。

### 3.2 深度
- 主线实验需 ≥10 档：Databento `mbp-10` / Tardis / 录制器 = 40/40 ✓。
- **避免 L1-only 源**（bookTicker、TAQ）做主线——book 退化 4/40，P1/P2 失去意义。

### 3.3 归一化
- adapter 输出**未 z-score**（manifest `normalized:false`）；训练前必须用**训练集统计量** z-score（对齐 `LOBSTERDataBuilder`），否则尺度不匹配会让模型静默学坏。
- FI-2010 已是 DecPre 归一化（另一套方案），仅作 sanity。

### 3.4 多资产对齐
- 当前 `MultiAssetLOBDataset` **仅按最短长度截断**，无时间戳对齐。
- 真做 ETF-NAV gap / lead-lag → 需按时间戳重采样对齐（跨资产事件频率不同）。这是 equity / crypto-basket 主线的**待补点**。

### 3.5 量是否够 + 成本
- 所有实测源行数都**远超**训练所需；真正约束是**成本（Databento）**或**采集时间（录制器）**，不是可得性。
- Databento `$125` 足够 **QQQ + 4 成分 × ~3 月** 的 `mbp-10`。
- 免费深档路线：Tardis 月样本 + 自录 depth20 篮子，**零成本**拿到真·多档 crypto。

### 3.6 推荐组合（按用途）

| 用途 | 首选 | 备注 |
|---|---|---|
| CI / sanity | **FI-2010** | 免费，已通 |
| 加密主线（免费深档） | **Tardis 月样本 + 自录 depth20 篮子** | spot/perp basis 作 ETF-NAV 类比 |
| 股票主线（便宜真实多档） | **Databento `mbp-10`** | $125≈3 月 QQQ+4 成分；order 为 proxy |
| 股票 + 真实逐笔 | **Databento `mbo` / LOBSTER** | 需新 MBO adapter 或付费 LOBSTER |
| 跳过 | IEX DEEP（5.9GB/天，重）、bookTicker/TAQ（L1） | |

---

## 附录：复现命令

```bash
# 冒烟（合成，三 adapter + dataset）
cd DeepMarket && conda run -n deep_market python -m tests.data_adapters_smoke

# Binance 20 档录制（自己跑；输出即 adapter schema）
python tools/binance_recorder.py --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --out data/crypto_raw --market futures --hours 12

# 录制结果转 .npy（train/val/test split）
python -c "from preprocessing.adapters import equity_file_to_splits; \
    equity_file_to_splits('data/crypto_raw/BTCUSDT.csv', 'data/crypto/BTCUSDT')"

# Databento 估价（免费）后再拉，避免超支
#   client.metadata.get_cost(dataset='XNAS.ITCH', schema='mbp-10', symbols=[...], start=, end=)
# Tardis 免费样本（每月 1 号，无需 key）
#   https://datasets.tardis.dev/v1/binance-futures/book_snapshot_25/2024/01/01/BTCUSDT.csv.gz
```

> 实测用的一次性脚本与原始数据放在仓库外的 `~/aiquant_data_verify/`，不入库。
