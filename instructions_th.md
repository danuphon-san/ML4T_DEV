# คู่มือ ML4T Research Pipeline (ภาษาไทย)

อัปเดต: 2026-06-05

---

## 1. ภาพรวม Pipeline

```
data → engineer → diagnostic → backtest
         ↑                        ↑
     research/                research/
  (build dataset)           (run backtest)
```

Pipeline วิจัยฉบับเต็มแบ่งเป็น 2 ส่วน:

| ส่วน | หน้าที่ |
|---|---|
| **Core packages** (`data`, `engineer`, `diagnostic`, `backtest`) | Library ที่ใช้ซ้ำได้ |
| **`research/`** | Scripts วิจัย — ประกอบ core packages เข้าด้วยกัน ตั้งแต่ดึงข้อมูลจนถึง backtest |

---

## 2. Setup

### macOS requirement

```bash
brew install libomp   # จำเป็นสำหรับ diagnostic (LightGBM/XGBoost)
```

### ติดตั้ง core packages

```bash
cd /Users/mit/Project/ML4T/data       && uv sync --dev
cd /Users/mit/Project/ML4T/engineer   && uv sync --python 3.12 --extra ta --extra store
cd /Users/mit/Project/ML4T/models     && uv sync
cd /Users/mit/Project/ML4T/diagnostic && uv sync
cd /Users/mit/Project/ML4T/backtest   && uv sync --dev
```

> `engineer` ใช้ Python 3.12 เสมอ — 3.14 มี intermittent crash บน macOS arm64

### ติดตั้ง research environment

```bash
cd /Users/mit/Project/ML4T/research
uv sync --python 3.12
```

`research/pyproject.toml` references `engineer`, `models`, `diagnostic` แบบ path dependency
ทำให้ scripts ทุกตัวใน `research/` ใช้ library เวอร์ชันเดียวกับ core packages

---

## 3. ลำดับการทำงาน (ตั้งแต่ต้นจนถึง Backtest)

```
Step 1  ดึงข้อมูลราคา              fetch_sp500_chunks.py
Step 2  สร้าง PIT composition      build_sp500_pit_composition.py
Step 3  ดึงข้อมูลประวัติศาสตร์     fetch_sp500_historical_extended.py
Step 4  Build model frame          build_sp500_10yr_dataset.py
Step 5  Screen signals             screen_sp500_all_signals.py
Step 6  Cluster signals            cluster_signals.py
Step 7  Backtest                   backtest_composite_signals.py
                               หรือ backtest_walkforward_10yr.py
```

---

## Step 1: ดึงข้อมูลราคา (`fetch_sp500_chunks.py`)

### ทำอะไร

ดึงข้อมูล OHLCV รายวันจาก Yahoo Finance สำหรับทุก ticker ใน universe และจัดเก็บใน Hive-partitioned Parquet format ที่ `~/ml4t-data/equities_daily_{symbol}/year={Y}/month={M}/data.parquet`

### ทำไมต้องทำ

`analyze_signal` และ backtest engine ต้องการข้อมูลราคาครบถ้วนก่อน feature engineering จะทำงานได้ การแยก "ดึงข้อมูล" ออกจาก "สร้าง feature" ทำให้ re-run feature ได้เร็วโดยไม่ต้องดึงซ้ำ

### คำสั่ง

```bash
cd /Users/mit/Project/ML4T/research

# ดึงข้อมูลสำหรับ universe ที่ขาดอยู่
uv run python fetch_sp500_chunks.py --prefix sp500_full --scope missing

# ดึงใหม่ทั้งหมด (full refresh)
uv run python fetch_sp500_chunks.py --prefix sp500_full --scope all \
    --update-mode backfill --start 2015-01-01 --end 2026-06-01
```

### ข้อควรรู้

- ข้อมูลถูกเก็บใน `~/ml4t-data/` แบบ Hive partition — อ่านด้วย `load_symbol_history()` ใน `build_research_dataset.py`
- Yahoo Finance มี rate limit — script แบ่งเป็น chunk ๆละ 25-50 ตัว
- `--scope missing` จะข้ามตัวที่มีข้อมูลแล้ว ประหยัดเวลามาก

---

## Step 2: สร้าง Point-in-Time Composition (`build_sp500_pit_composition.py`)

### ทำอะไร

ดึงตาราง "S&P 500 Changes" จาก Wikipedia (การเพิ่ม/ลบ ticker พร้อม effective date) แล้วสร้างตาราง `(date, ticker)` ว่าในแต่ละวัน มีหุ้นตัวใดบ้างที่อยู่ใน S&P 500 จริงๆ

### ทำไมต้องทำ — ปัญหา Survivorship Bias

ถ้าใช้ list S&P 500 ปัจจุบัน (500 ตัว) ไป backtest ย้อนหลัง 10 ปี จะได้ผลลัพธ์ที่ดูดีเกินจริง เพราะ:

- **หุ้นที่ถูกถอดออก** (บริษัทล้ม, ถูกซื้อกิจการ, ผลงานแย่) **ไม่อยู่ใน list** แต่ประวัติศาสตร์ list นั้นมีพวกเขาอยู่
- Portfolio ย้อนหลังจะเหมือน "รู้อนาคต" ว่าบริษัทไหนรอดมาถึงปัจจุบัน
- ผลตอบแทนที่ได้จะสูงกว่าความเป็นจริงอย่างมีนัยสำคัญ

**Point-in-Time (PIT) composition** แก้ปัญหานี้: ในแต่ละวัน ใช้เฉพาะหุ้นที่ถูก include ใน index ณ วันนั้นจริงๆ

### อัลกอริทึม

1. เริ่มจาก current S&P 500 (500 ตัว ณ วันที่ snapshot)
2. วิ่งย้อนหลังผ่าน change events — undo การ add → ลบออก, undo การ remove → เพิ่มกลับ
3. Forward replay events ตามวันที่จริงเพื่อสร้าง `(date, ticker)` pairs

```bash
uv run python build_sp500_pit_composition.py \
    --start 2016-01-01 --end 2026-06-01
```

**Output:** `outputs/sp500_pit/sp500_pit_composition.parquet`

| Metric | ค่า |
|---|---|
| Unique tickers 10 ปี | 716 ตัว |
| Current constituents | 500 ตัว |
| Historical-only (removed) | 216 ตัว |
| Avg constituents/day | ~508 ตัว |

### ข้อจำกัด

Wikipedia tracks เฉพาะ S&P 500 — ถ้าต้องการ index อื่น หรือต้องการ intraday precision ต้องใช้ข้อมูลเชิงพาณิชย์ (Compustat, Bloomberg MEMB)

---

## Step 3: ดึงข้อมูลหุ้นประวัติศาสตร์ (`fetch_sp500_historical_extended.py`)

### ทำอะไร

ดึงข้อมูลราคาสำหรับ:
1. **Historical-only tickers** (216 ตัว) — หุ้นที่เคยอยู่ใน index แต่ถูกถอดออกแล้ว
2. **Current 500** — ขยายย้อนหลังไปถึง 2015 (ปกติมีข้อมูลแค่จาก 2022)

### ทำไมต้องทำ

หลังจากมี PIT composition แล้ว ยังต้องมีข้อมูลราคาของหุ้นเหล่านั้นด้วย ไม่งั้น PIT filter จะกรอง row เหล่านั้นออกไปเองเพราะไม่มีข้อมูล และ survivorship bias ก็ยังคงอยู่

### คำสั่ง

```bash
cd /Users/mit/Project/ML4T/research

# Mode 1: ดึงเฉพาะ historical-only tickers (216 ตัว ใช้เวลา ~5-10 นาที)
uv run python fetch_sp500_historical_extended.py --mode historical-only

# Mode 2: ดึงทั้งหมด รวมถึงขยาย current 500 ย้อนหลัง (ใช้เวลา ~20-30 นาที)
uv run python fetch_sp500_historical_extended.py --mode extend-all

# ดูว่าจะดึงอะไรบ้าง (ไม่ดึงจริง)
uv run python fetch_sp500_historical_extended.py --mode extend-all --dry-run
```

### ผลที่ได้

| กลุ่ม | ความสำเร็จ | ไม่ได้ข้อมูล | เหตุผลที่ไม่ได้ |
|---|---|---|---|
| Historical-only (216) | 116 ตัว | 100 ตัว | ถูก acquire จริงๆ (CELG→BMY, ATVI→MSFT, XLNX→AMD) — Yahoo ลบข้อมูลหลัง delist |
| Current 500 extend | 492 ตัว | 8 ตัว | Spin-off ใหม่ (CEG, GEHC, GEV, KVUE) ยังไม่มีข้อมูลก่อน 2022 |

**การสูญเสียนี้ยอมรับได้**: หุ้นที่ถูก acquire ส่วนใหญ่ถูก acquire ในราคาพรีเมียม (ราคาสูงขึ้น) — การไม่มีข้อมูลเหล่านี้ทำให้ bias เล็กน้อยในทิศทาง conservative ไม่ใช่ optimistic

### หมายเหตุ: Treasury Yields

ก่อน build dataset ต้องขยาย macro data ด้วย:

```bash
cd /Users/mit/Project/ML4T/data
uv run python -c "
import yfinance as yf, polars as pl, pandas as pd
from pathlib import Path

symbols_map = {'^IRX': 'DGS2', '^FVX': 'DGS5', '^TNX': 'DGS10', '^TYX': 'DGS30'}
series_list = []
for yf_sym, col_name in symbols_map.items():
    df = yf.download(yf_sym, start='2015-01-01', end='2026-06-05', progress=False)
    close = df['Close'].squeeze(); close.name = col_name
    series_list.append(close)

merged = pd.concat(series_list, axis=1).reset_index()
merged.columns = ['timestamp', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
merged['timestamp'] = pd.to_datetime(merged['timestamp']).dt.tz_localize(None)
merged['YIELD_CURVE_SLOPE'] = merged['DGS10'] - merged['DGS2']
merged['YIELD_CURVE_5_10'] = merged['DGS10'] - merged['DGS5']
merged = merged.ffill().dropna()

pl_df = pl.from_pandas(merged).with_columns(pl.col('timestamp').cast(pl.Datetime('us')))
pl_df.write_parquet(Path.home() / 'ml4t-data' / 'treasury_yields.parquet')
print(f'Saved: {len(pl_df)} rows, {pl_df[\"timestamp\"].min()} -> {pl_df[\"timestamp\"].max()}')
"
```

**ทำไม:** `build_model_frame()` ทำ `drop_nulls` บน CONTEXT_COLS ซึ่งรวม DGS yield columns ถ้า treasury yields มีข้อมูลแค่จาก 2022 ทุก row ก่อนปี 2022 จะถูกตัดทิ้งโดยไม่มี error — model frame จะดูเหมือนทำงานปกติ แต่ได้ข้อมูลแค่ 4 ปีแทนที่จะเป็น 10 ปี

---

## Step 4: สร้าง Model Frame (`build_sp500_10yr_dataset.py`)

### ทำอะไร

ประกอบข้อมูลทั้งหมดเข้าด้วยกัน:
1. โหลดราคาจาก `~/ml4t-data/` สำหรับทุก ticker ใน PIT composition
2. คำนวณ features ทั้ง 105 ตัว (ต่อหุ้น ต่อวัน)
3. สร้าง Triple-barrier labels
4. Merge Fama-French factors + Treasury yields
5. ทำ `drop_nulls` เพื่อเอาเฉพาะ rows ที่มีข้อมูลครบ
6. **PIT filter**: inner-join กับ `sp500_pit_composition.parquet` เพื่อเก็บเฉพาะ `(date, ticker)` ที่ ticker อยู่ใน index จริงๆ ณ วันนั้น

```bash
cd /Users/mit/Project/ML4T/research
uv run python build_sp500_10yr_dataset.py
```

### ทำไม PIT Filter ถึงสำคัญ

โดยไม่มี PIT filter: model จะเห็น AAPL, MSFT ฯลฯ ทุกวันตั้งแต่ 2016 — แต่บางช่วงเวลาหุ้นเหล่านี้อาจไม่ได้อยู่ใน index (หรือหุ้นที่ถูกถอดออกในปี 2022 ก็ยังอยู่ใน dataset ของปี 2016-2021)

โดยมี PIT filter: แต่ละวันมีเฉพาะหุ้นที่ถูก include ใน S&P 500 จริงๆ — backtest จะ realistic ที่สุดเท่าที่ทำได้ด้วยข้อมูลฟรี

### ผลลัพธ์

| Metric | 4-year (sp500_full) | 10-year (sp500_10yr) |
|---|---:|---:|
| Raw price rows | ~660K | 1,658,952 |
| Model frame rows | ~403K | **1,188,550** |
| Symbols | 498 | **592** |
| Trading dates | 814 | **2,576** |
| Date range | 2023-02-01→2026-04-30 | **2016-02-01→2026-04-30** |
| Survivorship bias | ใช่ | **ไม่มี** |

**Output:** `outputs/sp500_10yr/sp500_10yr_model_frame.parquet`

### หมายเหตุเรื่อง Feature Warmup

แม้ raw data เริ่มต้นจาก 2015-01-02 แต่ model frame เริ่มจาก 2016-02-01 เพราะ:
- Features บางตัว (เช่น `volatility_percentile_rank`) ต้องการ lookback 252 วัน (1 ปี)
- `drop_nulls` ตัด rows ที่ feature ยังคำนวณไม่ได้ออกโดยอัตโนมัติ
- ดึงข้อมูลตั้งแต่ 2015 เพื่อให้ features พร้อมตั้งแต่ต้นปี 2016

---

## Step 5: Signal Screening (`screen_sp500_all_signals.py`)

### ทำอะไร

ทดสอบทุก feature column ใน model frame เป็น cross-sectional signal ทั้งทิศทาง raw และ inverse เพื่อหาว่า signal ไหนมีความสามารถทำนาย forward return ที่มีนัยสำคัญทางสถิติ

### ทำไมต้องทำ

มี 105 features = 210 potential signals (raw + inverse) ไม่ใช่ทุกตัวจะ predictive และ direction ก็สำคัญ (เช่น `-bollinger_bands_lower` ดีกว่า `bollinger_bands_lower`) — screening กรองเอาเฉพาะตัวที่ผ่าน threshold ทั้ง 4:

| Metric | Threshold | ความหมาย |
|---|---|---|
| IC 21D | > 0 | ทิศทางถูกต้อง (correlation กับ forward return เป็นบวก) |
| Spread 21D | > 0 | Top quantile ได้ผลตอบแทนสูงกว่า bottom quantile |
| Spread t-stat 21D | > 2.0 | นัยสำคัญทางสถิติ (ไม่ใช่ noise) |
| Monotonicity 21D | ≥ 0.5 | ผลตอบแทนเรียงตาม quantile อย่างสม่ำเสมอ |

### ทำไมต้อง Screen ทั้ง 3 Horizon (1D, 5D, 21D)

- **21D** คือ primary selection criterion — ตรงกับธรรมชาติของ volatility/liquidity signals ที่มีข้อมูลที่ noisier
- **5D** ตรงกับ rebalance cadence ของ strategy (WeeklyLongOnly5 rebalance ทุก 5 วัน) — IC 5D บอกว่า signal predict ได้ดีในช่วงเวลาที่ portfolio ถือจริง
- **1D** บอก signal decay speed — signal ที่ IC 1D ต่ำแต่ IC 21D สูงหมายความว่าเป็น slow-moving signal เหมาะกับ weekly rebalance มากกว่า daily

```bash
cd /Users/mit/Project/ML4T/research

# Screening ปกติ (full data, ทั้ง 3 horizon)
uv run python screen_sp500_all_signals.py --prefix sp500_10yr \
    2>&1 | tee /tmp/screen_10yr.log

# ดู progress (script print ทุก signal)
tail -f /tmp/screen_10yr.log
```

### ผลลัพธ์ (sp500_full, 4-year)

**81 signals ผ่าน** จาก 250 ที่ทดสอบ

Top signals โดย Spread t-stat 21D:

| Signal | t-stat | กลุ่ม |
|---|---:|---|
| volatility_regime_probability_prob_high_vol | 32.74 | Volatility regime |
| kyle_lambda | 30.09 | Liquidity/Vol |
| ewma_volatility | 29.14 | Volatility |
| garch_forecast | 28.95 | Volatility |
| risk_adjusted_returns_sharpe_ratio | 25.95 | Risk-adjusted |
| -maximum_drawdown_time_underwater | 22.31 | Drawdown |
| log_volume | 21.28 | Volume |
| plus_di | 14.21 | Momentum |

**Key insight:** Volatility signals ครองอันดับต้น — high-volatility stocks outperform low-volatility stocks ใน cross-section ซึ่งสอดคล้องกับ volatility-as-risk-premium ไม่ใช่ low-vol anomaly

### ข้อควรรู้เรื่อง Performance

Dataset 10yr (1.18M rows) ใช้เวลานานกว่า 4yr มาก ถ้า process ดูเหมือนค้างให้เช็ค:

```bash
# ดูว่า process ยังทำงานอยู่ไหม
ps aux | grep screen_sp500 | grep -v grep

# ดู log (script print ทุก signal ตั้งแต่ version ล่าสุด)
tail -f /tmp/screen_10yr.log

# เช็คว่า output file สร้างแล้วหรือยัง
ls outputs/sp500_10yr/sp500_10yr_all_signal_screen.parquet
```

---

## Step 6: Signal Clustering (`cluster_signals.py`)

### ทำอะไร

จัด group signals ที่ให้ข้อมูลซ้ำกันเข้าด้วยกัน โดยใช้ Spearman rank-correlation ของ cross-sectional ranks — ถ้า |r| ≥ 0.8 ถือว่าสัญญาณสองตัวนั้น "เหมือนกัน" และเลือกแค่ตัวที่ดีที่สุด (spread_t_21d สูงสุด) เป็น representative

### ทำไมต้องทำ

ถ้าใส่ signals ที่ correlation สูงเข้า composite พร้อมกัน (เช่น SMA, EMA, DEMA, TEMA, TRIMA ซึ่ง |r| ≈ 0.99) มันแทบไม่ได้เพิ่ม diversification แต่เพิ่ม complexity — clustering ทำให้ composite signal แต่ละ component เป็น "independent information source" จริงๆ

**ผลที่ได้ (sp500_full):** 80 signals → 32 clusters

- Cluster ใหญ่ที่สุด [26 ตัว]: `-bollinger_bands_lower` เป็น representative — ทุก price-level/trend indicator (SMA, EMA, SAR, linearreg ฯลฯ) แทบเหมือนกันทั้งหมด
- Cluster [6 ตัว]: `kyle_lambda` — volatility/liquidity signals
- Cluster [4 ตัว]: `risk_adjusted_returns_sharpe_ratio` — risk-adjusted metrics

```bash
cd /Users/mit/Project/ML4T/research

uv run python cluster_signals.py --prefix sp500_10yr --threshold 0.8
```

**Output:** `outputs/sp500_10yr/sp500_10yr_signal_clusters.json`

---

## Step 7A: Composite Signal Backtest (`backtest_composite_signals.py`)

### ทำอะไร

นำ cluster representatives มาสร้าง composite signals และ backtest เพื่อหาว่า "dimension" ไหนของข้อมูลที่ drive return จริงๆ

**Composite definitions:**

| Signal | Components | หลักการ |
|---|---|---|
| `vol_composite` | kyle_lambda + garch_forecast + coefficient_of_variation | Volatility/liquidity dimension |
| `risk_composite` | sharpe_ratio + (−time_underwater) + (−max_drawdown) | Risk-adjusted/drawdown dimension |
| `combined` | average rank ของ vol + risk | ผสมทั้งสอง dimension |

การสร้าง composite ใช้ **cross-sectional rank averaging**:
1. Rank แต่ละ signal ข้ามทุก asset ในแต่ละวัน (ลบ scale differences)
2. Average ranks จากทุก component
3. ใช้ composite rank เป็น signal ในการเลือกหุ้น

```bash
uv run python backtest_composite_signals.py --prefix sp500_10yr
```

### Strategy: WeeklyLongOnly5

| Parameter | ค่า | เหตุผล |
|---|---|---|
| long_count | 5 | concentrate ใน top signals, ไม่กระจาย noise |
| position_size | 15% | max leverage ≈ 75%, buffer สำหรับ cash |
| rebalance_frequency | 5 วัน | weekly — balance ระหว่าง signal decay และ turnover cost |
| short_count | 0 | long-only ง่ายกว่าในการ implement จริง |

### ผลลัพธ์ (sp500_full)

Test period: 2025-09-08 → 2026-04-30 (7 เดือน, 6 rebalances)

| Strategy | Return % | Sharpe | DSR |
|---|---:|---:|---:|
| **plus_di** | 74.64 | **3.49** | **0.999** |
| log_volume | 62.61 | 2.74 | — |
| combined | 91.21 | 2.46 | — |
| risk_composite | 44.75 | 1.64 | — |
| vol_composite | 17.85 | 1.13 | — |

**Counter-intuitive insight:** Volatility signals มี IC สูงสุด (t≈29-33) แต่ Sharpe ต่ำสุดในการ backtest เหตุผล:
- High-vol stocks มี higher return แต่ Sharpe ratio ลดลงเพราะ risk ก็สูงตาม
- Momentum signal (`plus_di`) เลือกหุ้นที่ "กำลังวิ่ง" — Sharpe สูงกว่าเพราะ risk-per-unit-return ดีกว่า
- **สำคัญที่สุด:** 6 trades ไม่เพียงพอสำหรับ statistical conclusion — การ ranking ระหว่าง strategies อาจเป็น noise ทั้งหมด

### DSR (Deflated Sharpe Ratio)

ใช้แทน Sharpe Ratio ธรรมดาเพื่อ account for selection bias: ถ้าทดสอบ 6 strategies พร้อมกัน โอกาสที่ strategy ที่ดูดีที่สุดจะดีด้วยความบังเอิญมีสูงกว่าการทดสอบครั้งเดียว DSR ปรับ Sharpe ให้สะท้อน "probability of being truly profitable after adjusting for number of trials"

DSR = 0.999 หมายความว่า `plus_di` มีโอกาส 99.9% ที่ Sharpe จะยังเป็นบวกแม้หลัง adjust — แต่ **6 trades ยังถือว่าน้อยเกินไปสำหรับ conclusion ที่แข็งแกร่ง**

---

## Step 7B: Walk-Forward Backtest (`backtest_walkforward_10yr.py`)

### ทำอะไร

แก้ปัญหา "test period สั้นเกินไป" โดยใช้ rolling window: ทดสอบ 5 folds โดยแต่ละ fold train บน 5 ปี และ test บน 1 ปีที่ไม่ทับกัน

### ทำไมต้องทำ

Backtest แบบ single split มีปัญหา:
- **ตัวเลข DSR ขึ้นกับ test period ที่เลือก** — ถ้าช่วง test บังเอิญเป็น bull market ผลดีทุก strategy
- **6 trades ไม่พอ** สำหรับ meaningful DSR — ต้องการอย่างน้อย 20-30 trades
- **Look-ahead bias ใน signal selection** — เราเลือก composite definition หลังจากเห็น 4 ปีทั้งหมด

Walk-forward CV แก้ทั้งสามปัญหา:
- แต่ละ fold มี 5-10 trades → รวม 25-50 trades ทั้ง 5 folds
- Signal screening ทำบน training data ของ fold นั้นเท่านั้น
- Test period ไม่เคยถูกเห็นตอน train

### Fold Design

| Fold | Train | Test | หมายเหตุ |
|---|---|---|---|
| 1 | 2016 → 2021 | 2021 | Pre-COVID recovery |
| 2 | 2017 → 2022 | 2022 | Rising rate environment |
| 3 | 2018 → 2023 | 2023 | Post-rate-hike recovery |
| 4 | 2019 → 2024 | 2024 | AI boom |
| 5 | 2020 → 2025 | 2025 | Late cycle |

**ครอบคลุม 5 regime ที่แตกต่างกัน** — ถ้า strategy ทำงานดีใน 4-5 folds มั่นใจได้ว่าไม่ใช่ curve-fitting

```bash
cd /Users/mit/Project/ML4T/research

# รันทุก fold
uv run python backtest_walkforward_10yr.py --prefix sp500_10yr

# รัน fold เดียว (ทดสอบ)
uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --fold 3
```

**Output:**
- `outputs/sp500_10yr/walkforward/fold_01/` ... `fold_05/`
- `outputs/sp500_10yr/sp500_10yr_walkforward_report.md`

---

## 8. Macro Concepts & Design Principles

### Cross-Sectional vs. Time-Series Signal

Pipeline นี้ใช้ **cross-sectional signals** เท่านั้น:
- ในแต่ละวัน: rank หุ้นทุกตัวในกันและกัน — ไม่ใช่เทียบกับประวัติศาสตร์ตัวเอง
- ผลลัพธ์: signal บอกว่า "หุ้น A น่าสนใจกว่าหุ้น B ณ วันนี้" ไม่ใช่ "หุ้น A ดีกว่าตัวเองเมื่อเดือนที่แล้ว"
- ข้อดี: ลด regime dependency, เปรียบเทียบได้ข้ามช่วงเวลา, ง่ายต่อการสร้าง composite

### Triple-Barrier Labeling

ใช้แทน fixed-horizon labels เพราะ:
- หุ้นแต่ละตัวมี volatility ต่างกัน — fixed 3% target ง่ายสำหรับหุ้น volatile แต่ยากสำหรับหุ้น stable
- Triple barrier: profit target, stop loss, หรือ time limit — whichever comes first
- Labels ที่ได้สะท้อน "quality of trade" ได้ดีกว่า return ดิบ

### Composite > Single Signal

- Single best signal เสี่ยง overfitting กับ regime เดียว
- Composite ที่ประกอบจาก signals ที่ **low correlation กัน** ได้ประโยชน์จาก diversification ที่ information level
- ตัวอย่าง: vol_composite (volatility risk) + risk_composite (drawdown risk) มีข้อมูล orthogonal กัน → combined ดีกว่าทั้งคู่ในหลาย regime

### DSR ดีกว่า Sharpe Ratio สำหรับ Selection

เมื่อทดสอบหลาย strategies พร้อมกัน Sharpe ratio ของ winner จะ inflated จาก selection การใช้ DSR ทำให้:
- รู้ probability ที่แท้จริงว่า strategy นี้จะ profitable จริงๆ (ไม่ใช่แค่โชค)
- Compare ได้ข้ามงานวิจัยที่มีจำนวน strategies ทดสอบต่างกัน

---

## 9. คำสั่งสรุป (Quick Reference)

```bash
cd /Users/mit/Project/ML4T/research

# --- ดึงข้อมูล ---
uv run python fetch_sp500_chunks.py --prefix sp500_full --scope missing
uv run python fetch_sp500_historical_extended.py --mode extend-all

# --- สร้าง PIT composition ---
uv run python build_sp500_pit_composition.py --start 2016-01-01 --end 2026-06-01

# --- Build dataset ---
uv run python build_sp500_10yr_dataset.py                   # 10yr + PIT filter
uv run python build_research_dataset.py --prefix sp500_full # 4yr standard

# --- Signal screening ---
uv run python screen_sp500_all_signals.py --prefix sp500_10yr 2>&1 | tee /tmp/screen.log

# --- Clustering ---
uv run python cluster_signals.py --prefix sp500_10yr --threshold 0.8

# --- Backtest ---
uv run python backtest_composite_signals.py --prefix sp500_10yr        # single period
uv run python backtest_walkforward_10yr.py --prefix sp500_10yr         # walk-forward
uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --fold 3 # single fold

# --- เช็ค output ---
cat outputs/sp500_10yr/sp500_10yr_all_signal_screen_report.md
cat outputs/sp500_10yr/sp500_10yr_walkforward_report.md
```

---

## 10. ข้อควรระวัง

| ปัญหา | วิธีตรวจจับ | วิธีแก้ |
|---|---|---|
| Survivorship bias | Model frame ขนาดเล็กกว่าที่คาด | ใช้ PIT composition + fetch historical tickers |
| Treasury yields truncation | Model frame เริ่มต้นจาก 2022 แทนที่จะเป็น 2016 | ขยาย treasury_yields.parquet ย้อนหลังก่อน build |
| Signal screening ค้าง | Process ไม่ print อะไรเลย หลายนาที | เช็ค `ps aux`, ดู log — script print ทุก signal ตั้งแต่ version ล่าสุด |
| 6 trades ไม่พอ | DSR สูงแต่ confidence interval กว้าง | ใช้ walk-forward backtest เพื่อรวม trades ข้าม folds |
| Feature warmup ทำให้ model frame สั้นกว่า raw data | Date range ใน summary ไม่ตรงกับ raw data | ดึงข้อมูล raw ให้เริ่มก่อน model start 1 ปี (ดึงจาก 2015 เพื่อ model จาก 2016) |
| libomp ไม่ได้ติดตั้ง | ImportError จาก lightgbm/xgboost | `brew install libomp` |
