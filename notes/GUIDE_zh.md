# 311 reporting correction：方法说明与结果解读（零基础版）

> 写法约定：technical terms 用英文，直觉和解释用中文。
> 对应代码：`src/01_data.py` … `src/06_surface.py`；对应结果：`output/`。
> 本文假设读者没有任何 context。每个文件、每个数字都从头解释。

---

## 0. 问题是什么

**我们想知道的：** 费城每个地方真实发生了多少 illegal dumping（非法倾倒垃圾）。

**我们有的三份数据：**

| 数据 | 是什么 | 优点 | 缺点 |
|---|---|---|---|
| 311 dumping reports | 居民打电话举报倾倒，116,772 条，每条有坐标和时间 | 便宜、全城、连续 | **有偏**：同样多的垃圾，爱打电话的社区举报多，不爱打的社区举报少 |
| Litter Index survey | 市政人员沿路线实地打分 1–4，72,084 次 visit | 接近真实情况（ground truth） | 贵、只有 2022–23 两年、只覆盖抽样路线 |
| Streetlight outage reports + poles | 路灯坏了的 311 举报 56,472 条，路灯 203,036 根 | 可以当"标尺"（见下） | 本身不是垃圾数据 |

**核心想法：** 路灯坏不坏和社区性格无关，全城每根路灯的真实故障率大致一样。所以"每根路灯对应多少条 outage report"的地区差异，**全部来自居民爱不爱打 311**。这就测出了每个地方的 **reporting propensity**（举报倾向），记作 w_R。然后把每条 dumping report 按当地的 w_R 打折：爱打电话的地方，一条举报算少一点；不爱打的地方，一条举报算多一点。打折的力度叫 **β**。β 不是拍脑袋定的，而是让打折后的地图和 survey 最吻合的那个值。

**一句话：** 用路灯测出"筛子的孔径"，把 311 举报按孔径还原，再用实地 survey 检验还原得对不对、顺便估出还原力度。

---

## 1. 整条链怎么走

六个脚本，严格按顺序跑，每个只做一件事：

```
原始数据
   │  01_data.py          整理成表格：点、路灯、街区、survey
   ▼
output/visits.parquet, points_*.parquet, poles_xy, support_grid, cbg*
   │  02_reporting.py     用路灯算 w_R
   ▼
output/wR_events.parquet, wR_visits.parquet, wR_raster.npz
   │  03_field.py         在每个 survey 点算 311 field（各种 β 下的版本）
   ▼
output/fields/h*.npz
   │  04_calibrate.py     用 survey 估 β，给 CI，做检验      ──►  output/calib/
   │  05_validate.py      留出一块地区不看，检验预测力       ──►  output/cv/
   ▼
   │  06_surface.py       三张图：311 原始、311 修正后、survey 自己   ──►  output/surface/
```

重要的分工：**03 只做几何（kernel 求和），04 和 05 只做统计（回归）。** 03 把所有可能用到的求和一次算完存下来，04/05 就不用再碰坐标。这就是为什么 `fields/` 里的 npz 文件那么重要，也是它"存那么多列"的原因，后面详细讲。

---

## 2. output 目录里每个文件是什么

### 2.1 根目录：01 的产出（整理好的原始数据）

| 文件 | 一行是什么 | 列 | 谁用 |
|---|---|---|---|
| `points_dumping.parquet` | 一条 dumping report | x, y（米，EPSG:26918）、ts（时间）、t_days（距 2021-01-01 的天数）、GEOID（所在街区） | 03, 04, 06 |
| `points_outage.parquet` | 一条 outage report | 同上 | 02, 03 |
| `poles_xy.parquet` | 一根路灯 | x, y, GEOID | 02 |
| `cbg.parquet` | 一个 census block group（街区，1,338 个） | 面积、exposure（SafeGraph 人流）、路灯数 | 02 |
| `cbg_geom.geojson` | 街区的多边形 | GEOID, geometry | 画图、空间 join |
| `support_grid.parquet` | 一个 50 m 网格 cell（105,979 个） | x, y, GEOID, E_cell | 03, 06 |
| `visits.parquet` | 一次 survey visit（72,084 次） | 见下 | 03, 04, 05, 06 |
| `cbg_blocks.parquet` | 一个街区属于哪个 block | GEOID, block（0–39） | 04, 05 |

**`support_grid`（built support）是什么：** 全城切成 50 m 的小格子，只保留"在某个街区内、离最近路灯 75 m 以内"的格子。这就是"有街道的地方"。河流、公园、机场、铁路场站都被排除了。每个街区的 exposure（SafeGraph 月均人流 stop 数）平均摊到它的格子上，得到 `E_cell`。后面所有"分母"都在这个网格上算。

**`visits.parquet` 的列：**
- `x, y`：米坐标；`lon, lat`：原始经纬度
- `ts, t_days`：survey 时间
- `rating`：1–4 的打分，这是 **outcome**（1 = 干净，4 = 最脏）
- `bulk`：这次 visit 是否记录了大件垃圾（0/1），备用
- `location_id, surveyor`：固定地点 ID、打分人
- `GEOID, exposure`：所在街区及其人流
- `loc, surv, month`：上面三个的整数编码（0, 1, 2…），回归用
- `block`：这个 visit 属于 40 个空间 block 中的哪一个

**`block` 是什么：** 把 1,335 个有 survey 的街区切成 40 块相邻的矩形区域，每块约 1,800 次 visit。它是 **bootstrap** 和 **cross-validation** 的基本单位（第 3.5 节解释为什么要按块）。

### 2.2 02 的产出：w_R

| 文件 | 内容 |
|---|---|
| `wR_events.parquet` | 每条 dumping report 所在位置的 w_R。两列：`wR_h400`（主用）、`wR_h600`（敏感性） |
| `wR_visits.parquet` | 每次 survey visit 所在位置的 w_R，同样两列 |
| `wR_raster.npz` | 全城 100 m 网格上的 w_R，一张图（301 行 × 277 列），城外是 NaN。给 torus-shift null 用（第 3.5 节） |
| `choices.json` | pipeline 自己选出来的参数和依据（h_w、h、β̂、CI、各种 sweep 表） |

### 2.3 `fields/`：03 的产出，整个 pipeline 的"中间仓库"

**先讲 `.npz` 是什么。** 它是 numpy 把多个数组打包成一个 zip 文件的格式。读法：

```python
import numpy as np
F = np.load("output/fields/h800.npz")
F.files            # 里面有哪些数组
F["num"].shape     # (72084, 23)
```

就像一个字典，key 是数组名。用 npz 而不用 csv 的原因：这些是纯数值矩阵（72,084 行 × 几十到上百列），npz 存取快、不丢精度、不需要列名解析。

**为什么存三个文件 `h500 / h800 / h1200`：** h 是 kernel 半径（第 3.2 节）。主用 800 m，另外两个是敏感性分析。三个文件内容结构一样，800 的多几个数组。

**`h800.npz` 里每个数组：**

| key | shape | 是什么 |
|---|---|---|
| `grid` | (23,) | β 的格点：−0.2, −0.1, 0, 0.1, …, 2.0。第 2 个（index 2）是 β = 0 |
| `num` | (72084, 23) | **核心。** 第 i 行第 j 列 = 第 i 次 visit 周围、用 β = grid[j] 打折后的 dumping report 加权和（kernel sum）。β = 0 那列就是不打折的 raw 版本 |
| `den` | (72084,) | 第 i 次 visit 周围的人流（exposure）加权和，分母 |
| `wbar` | (72084,) | 第 i 次 visit 周围那些 report 的 w_R 平均值（给 −c/b 对照用） |
| `lam_bar_raw` | 标量 | 全城平均的 "report per exposure" 率，在 support grid 上算的，shrinkage 用 |
| `mean_m` | (23,) | 每个 β 下打折权重 exp(−β w_R) 的全城平均，用来把 lam_bar 调到对应 β |
| `num_alt` | (72084, 23) | 同 `num`，但用 h_w = 600 的 w_R 打折（敏感性） |
| `num_null` | (72084, 50, 4) | 50 张 **平移过的** w_R 地图 × 4 个 β 值下的 kernel sum（torus-shift null 用） |
| `mean_m_null` | (50, 4) | 对应的权重平均 |
| `null_betas` | (4,) | 0.5, 1.0, 1.5, 2.0 |
| `num_outage` | (72084,) | 用 **outage reports** 代替 dumping reports 算的 kernel sum（placebo 用） |

**`num` 为什么有 23 列，这是整个设计的关键：** 后面要"转动 β 的旋钮，每转一格重算地图、重做回归"。重算地图是几何运算（找每个 visit 周围 800 m 内的所有 report，加权求和），慢；回归相对快。03 把 23 个 β 对应的地图**一次性**算好存成 23 列，04 和 05 只需要取某一列、除以 `den`、取 log，就得到那个 β 下的 field，然后做回归。换 β 等于换一列，不再碰坐标。

**从 npz 到 field 的公式**（04/05 里的 `field_log` 函数）：

```
log λ̂_β(visit i) = log( (num[i, j] + α) / (den[i] + α / λ̄_β) )      α = 1,  λ̄_β = lam_bar_raw × mean_m[j]
```

直觉：report 数 ÷ 人流 = "每单位人流有多少举报"，这是 **relative risk**。加 α = 1 是 **Gamma shrinkage**：假装每个地方先天有 1 条"虚拟举报"落在全城平均水平上，这样周围一条 report 都没有的 visit（0.35%）不会出现 log(0) = −∞，而是被拉到接近全城平均。

### 2.4 `calib/`：04 的产出，估 β 和各种检验

| 文件 | 内容 |
|---|---|
| `table1.csv` | **论文 Table 1。** 一行一个 specification：h、FE、α、h_w、ℓ(FE only)、ℓ(0)、ℓ(β̂)、Δℓ、β̂、−c/b、CI |
| `profile_curves.csv` | 每个 spec 在 23 个 β 格点上的 log-likelihood ℓ(β)，画 fig1 用 |
| `profile_*.npz` | **cache。** 每个 spec 一个：`ll` (23,) 和 `thetas` (23, 68)。thetas 是每个 β 下 fit 出来的全部 68 个系数。有了它，重跑 04 不用再 fit 一遍（2 分钟 vs 10 分钟），bootstrap 也用它做 warm start |
| `boot_context.npz` | 给并行 worker 进程的小包裹：β̂、它在 grid 上的 index、raw field、−c/b 模型的系数、ℓ(0)。纯技术文件 |
| `bootstrap.csv` | 一行一次 block resample（200 行）：`beta_prof`（这次重抽样下的 β̂）、`beta_ratio`（这次的 −c/b）、`moves / n_fits`（算法细节）。CI 就是 `beta_prof` 这列的 2.5% 和 97.5% 分位数 |
| `torus_null.csv` | 一行一张平移地图（50 行）：`dll_best` = 这张假地图能换来的最大 Δℓ |
| `summary.json` | 以上全部的汇总，含 torus 和 placebo 的数字 |

68 个系数是什么：4 个 spline 系数 + 16 个 month dummy + 45 个 surveyor 系数 + 3 个 cutpoint（第 3.3 节）。

### 2.5 `cv/`：05 的产出，out-of-sample 检验

| 文件 | 内容 |
|---|---|
| `folds.csv` | 一行 = 一个 fold × 一个 spec × 一个 h（10 × 4 × 3 减去不跑的组合）。`ll`：test 上的 log-likelihood 总和；`n`：test 的 visit 数；`ll_train`：training 上的；`somers_d`、`rmse`；`beta_fit`：这个 fold 在 training 上选出的 β |
| `table2.csv` | **论文 Table 2。** 按 spec × h 汇总：`ll_per_visit`（test 上平均每次 visit 的 log-lik）、`somers_d`、`gain_vs_S0`（比 raw 好多少）、`gain_se`（10 个 fold 的标准误）、`folds_better`（10 个 fold 里赢了几个）、`beta_fit_min / max` |
| `block_calibration_S0.csv / _S2.csv` | 40 个 block 各自：观测平均 rating vs 预测平均 rating |
| `block_calibration.csv` | 上面的回归 slope、intercept、correlation |

### 2.6 `surface/`：06 的产出，最终地图

三样东西，分开存，分开看：

| 文件 | 内容 |
|---|---|
| `dumping_2021-2025.tif` | **A 和 B。** GeoTIFF，50 m，5 个 band：`rate_raw`（311 原样的举报率，每年每 1,000 人流 stop）、`rate_corr`（每条举报乘 exp(−β̂ w_R) 后的"真实"率）、`log_rel_raw` / `log_rel_corr`（相对全城平均的 log，画图用）、`correction_factor`（corr ÷ raw） |
| `dumping_2022-2023.tif` | 同上，只用 survey 那两年的举报，和 survey 同期对照用 |
| `litter_survey.tif` | **C。** 只用 survey：`rating_adj`（去掉 month 和 surveyor 效应后的 rating，kernel 平滑）、`rating_mean`（不去效应的平均）、`n_eff`（kernel 里有效 visit 数；< 10 的格子是 NaN） |
| `litter_index_2023-06-30.tif`, `litter_index_2025-12-31.tif` | **附录。** 某日期前 120 天的 311 field 推过 sigmoid 得到的 E[rating]，4 个 band。回答的是"311 能多好地复现 survey"，不是"倾倒在哪" |
| `surface_cbg.csv` | 每个街区上面所有 band 的平均 + 它 2022–23 实际平均 rating |
| `summary.json` | 各窗口的全城率、分位数、修正幅度、A/B 与 C 的 Spearman |

**A 和 B 的单位：** 每年每 1,000 个 SafeGraph 月均 stop 的举报数。全城平均 1.19（raw）。它是 relative risk 的形式，和 calibration 用的 field 一致。

### 2.7 `figures/`

| 文件 | 内容 |
|---|---|
| `02_wR_map.png` | w_R 地图 + 和街区自身 outage 率的对照 |
| `03_sweep.png` | 不同 h、τ 下 raw field 和 rating 的 Spearman，选 h 的依据 |
| `fig1_profile.png` | **ℓ(β) − ℓ(0) 曲线**（估 β 的全部信息）+ bootstrap 直方图 |
| `fig2_linearisation.png` | rating 对 log field 的 S 形曲线，raw vs corrected |
| `05_cv.png` | out-of-fold log-lik 随 h 的变化 + block 层面的 calibration |
| `fig3_maps.png` | **主图。** (A) 311 原样，2021–25 平均；(B) 修正后；(C) survey 自己。A、B 同一色标 |
| `A2_windows_and_correction.png` | 附录：2022–23 窗口的 A、B；修正系数 log(B/A) 地图；w_R 地图 |
| `A1_rating_snapshot.png` | 附录：120 天 snapshot 推过 sigmoid 的 E[rating]，修正差，2025 extrapolation |

---

## 3. 方法逐步讲

### 3.1 Step 1：用路灯算 reporting propensity w_R（`02_reporting.py`）

**公式：**
```
w_R(s) = log[ Σ_outages K(|s − s_j|) / Σ_poles K(|s − s_p|) ] − 全城均值
```

**读法：** 站在地点 s，数周围 400 m 内的 outage reports（2021–2023 三年），再数周围 400 m 内的路灯；两者相除是"每根灯多少条举报"；取 log；减去全城平均，让 w_R = 0 表示全城平均水平、+1 表示比平均高 e ≈ 2.7 倍。

**K 是 kernel：** 不是硬性"400 m 内算 1、外面算 0"，而是越近权重越大、到 400 m 处降到 0 的平滑函数（quartic kernel，`K(d) = (1 − d²/h²)²`）。直觉是在每个 report 位置放一个小土堆，叠起来。

**假设（identifying assumption）：** 全城每根路灯的真实故障率一样。如果成立，outage 举报率的地区差异就只能是举报行为的差异。这是整个方法的前提，论文要明说并辩护。

**h_w = 400 m 是怎么选的：** 规则是"reliability ≥ 0.95 的最小半径"。reliability 的意思：kernel 里只有几十条 report 时，Poisson 随机噪声会很大；reliability = 1 − 噪声方差/信号方差。400 m 时 0.951，600 m 时 0.969。规则选 400，600 作为敏感性。

**实际数字：** w_R 在 report 位置的标准差 0.53。400 和 600 两个版本相关 0.94。在街区中心评估的 kernel w_R 和"该街区自己的 outage/poles"的 Spearman 是 0.86，说明 kernel 没算错。

### 3.2 Step 2：311 dumping field（`03_field.py`）

**公式：**
```
λ̂_β(s, t) = Σ_j exp(−β·w_R(s_j)) · K_h(|s − s_j|) · exp(−(t − t_j)/τ) · 1[t ≥ t_j]
            ───────────────────────────────────────────────────────────────
                            Σ_c K_h(|s − c|) · E_c
```

一项一项读：
- `exp(−β·w_R(s_j))`：第 j 条 report 的**打折权重**。它所在地 w_R 高（爱举报），权重 < 1；w_R 低，权重 > 1。β = 0 时全是 1，就是 raw 版本。
- `K_h(|s − s_j|)`：空间 kernel，半径 h = 800 m。
- `exp(−(t − t_j)/τ) · 1[t ≥ t_j]`：时间 kernel，只看 survey 之前的 report（causal），越久远权重越小，τ = 120 天。
- 分母：周围的人流 E，同样 kernel 平滑。整体是"每单位人流的举报率"。

**h = 800 是怎么选的：** 规则是"raw field 和 rating 的 Spearman 在最大值 0.01 以内的最小 h"。实际：300/500/800/1200/2000 m 对应 0.331 / 0.361 / 0.375 / 0.380 / 0.372。最大 0.380 在 1200，800 的 0.375 在 0.01 以内，选 800。（`03_sweep.png`）

**τ 为什么是"固定"而不是"估"：** 两个证据。第一，τ = 60/120/240/480 天的 Spearman 是 0.356 / 0.375 / 0.376 / 0.362，基本平的。第二，**placebo**：用 survey 之后 90 天的举报来"预测" rating，效果和之前 90 天一样好（没有 FE 时 past 系数 0.076、future 0.102；加 month + location FE 后 0.018 和 −0.010，差的 t 值分别是 −0.5 和 +0.9）。未来的举报不可能导致现在的垃圾，所以这个相关性全部来自"哪里一直脏"这种时间上不变的空间结构。结论：时间 kernel 只是把多长时间的举报汇总起来的窗口，不是垃圾的半衰期。

### 3.3 Step 3：把 field 映射到 1–4 rating（`design.py`, `ordinal.py`）

**模型：proportional-odds logit（ordinal logit）**
```
P(rating ≥ k) = σ( η − κ_k ),   k = 2, 3, 4
η = g(log λ̂_β) + month-of-sample effect + surveyor effect
```

**直觉：** 每个 visit 有一个看不见的"脏度分数" η。打分人心里有三条线 κ₂ < κ₃ < κ₄：η 超过第一条打 2，超过第二条打 3，超过第三条打 4。σ 是 sigmoid（S 形函数，把任意数压到 0–1 之间）。"三条 sigmoid 共用一个 η"就是 PI 说的 sigmoid。

**η 里的三项：**
- `g(log λ̂)`：field 对脏度的影响，用 **natural cubic spline**（5 个自由度）而不是直线，因为数据显示 rating 随 field 先升后趋平（`fig2`）。
- month-of-sample：17 个 survey 月份各一个截距。吸收"冬天普遍干净"之类的全城时间效应。
- surveyor：46 个打分人各一个截距。有人打分严、有人松。这项解释了 rating 方差的 27%，不能漏。编码用 deviation coding 加一点 ridge，这样 CV 里 training 没见过的打分人按平均水平预测。

**log-likelihood ℓ 是什么：** 模型给每次 visit **实际打出的那个分数**分配一个概率，取 log，72,084 个加起来。ℓ 越接近 0（越不负）模型越好。单独看一个 ℓ 没意义，**两个模型的 ℓ 之差 Δℓ** 才是比较标准。经验法则：Δℓ 加一个参数要超过 2 才算有意义；这里修正不加任何参数，Δℓ = +610。

### 3.4 Step 4：用 survey 估 β，profile likelihood（`04_calibrate.py`）

**直觉：** β 是旋钮。从 −0.2 转到 2.0，每转 0.1 一格，重算地图（取 `num` 的对应列）、重做 Step 3 的回归、记下 ℓ。ℓ 最高的那个位置就是 β̂。`fig1_profile.png` 左图画的就是 ℓ(β) − ℓ(0)，一条倒 U 形曲线，顶点在 0.86。

**为什么不直接做一个回归：** 可以，那就是 −c/b 对照（第 3.5 节）。但打折是作用在每条 report 上、在 kernel 求和**里面**的，直接回归只是它的一阶近似，而且对 α 很敏感。profile 用的是精确的非线性形式。

**三个关键点：**
- ℓ(0)：β = 0，不修正，这是 **baseline**。
- ℓ(β̂)：最高点。
- Δℓ = ℓ(β̂) − ℓ(0) = **修正带来的改善**。

**CI 怎么来的：spatial block bootstrap。** 普通 bootstrap 随机抽 visit，但相邻的 visit 高度相关（同一条街、同一个人打的分），抽出来的样本"信息量"比表面上少得多，CI 会假性地窄。所以按 **block** 抽：40 个 block 有放回地抽 40 个，被抽到 2 次的 block 权重为 2，没抽到的为 0，重新找 β̂。做 200 次，取 2.5% 和 97.5% 分位数。结果：sd 0.12，CI [0.63, 1.08]。

### 3.5 Step 5：检验（`04_calibrate.py`, `05_validate.py`）

**(a) −c/b 对照。** 一个线性回归：η = b·log λ̂_raw + c·w̄ + FE。推导（thinning）告诉我们 c = −b·β，所以 β = −c/b。这是"教科书版本"，一次拟合就出结果。它和 profile 的 β̂ 应该接近：0.94 vs 0.86，在 CI 内。它随 α 和 h 漂移（0.81 / 0.94 / 1.04），profile 不漂（0.88 / 0.86 / 0.83），这就是为什么 profile 是正式估计量、−c/b 只是对照。

**(b) Torus-shift null。** 质疑：会不会**任何**一张平滑的地图拿来打折都能改善拟合？检验：把 w_R 地图整体平移（像在环面上滚动，右边出去的从左边回来），平移后地图的平滑程度一模一样，只是位置错了。做 50 次，每张假地图也允许选自己最好的 β。结果：假地图最好的 Δℓ 是 +226，95% 分位 +51，平均 +11；真地图 +592（同一粗网格上）。p = 0。结论：改善来自路灯**把 w_R 放在了正确的位置**。

**(c) Outage placebo。** 质疑：dumping field 预测 rating，会不会只是因为它反映了"这个社区 311 活跃"这种通用信号？检验：用 outage reports 代替 dumping reports 建同样的 field，看它预测 rating 的 slope。结果：outage slope 0.25（se 0.09），dumping 0.95（se 0.08）；Δℓ 分别是 +218 和 +4,269。有一个小的通用成分（5%），但 dumping field 的信号主要是 dumping 本身。

**(d) Spatially blocked cross-validation。** 这是"修正有没有用"最硬的证据。
- 把 40 个 block 分成 10 组（fold），每组 4 块。
- 每次留出 1 组做 test，其余做 training。**Buffer**：training 里距离 test 任何一个 visit 1.2 km 以内的点也删掉（删掉 20–31%），否则 test 点的邻居在 training 里，等于偷看答案。
- 在 training 上 fit 四个模型：FE only、raw field（S0）、线性版（S1）、corrected（S2，**β 在 training 上选**，不用全样本的 β̂）。
- 在 test 上算每次 visit 的平均 log-likelihood。
- 10 个 fold 配对比较 S2 − S0。

### 3.6 Step 6：三张地图（`06_surface.py`）

**A：311 原样。** 把一个时间窗口（2021–2025 五年，或 2022–2023 两年）里的全部 dumping report 放到 support grid 上做 kernel 平滑，除以同样平滑的人流，再除以年数。单位是每年每 1,000 个 stop 的举报数。这是"居民报了多少"。

**B：修正后。** 同样的 report，每条先乘 exp(−β̂·w_R)，β̂ = 0.86。爱举报的地方一条举报算不到一条，不爱举报的地方一条算一条多。这是"按全城平均举报倾向还原后应该有多少"，也就是我们对真实倾倒分布的最佳估计。B ÷ A 是修正系数地图。

**C：survey 自己。** 只用 72,084 次 visit：先用一个 OLS 去掉 month 和 surveyor 的效应（它们解释了 rating 方差的 33%），再把调整后的 rating 在 visit 位置上用同样的 kernel 平滑。kernel 里有效 visit 少于 10 的格子留空（只有 1.5%）。这张图不含任何 311 信息。

**为什么要分开：** A 和 B 是我们的产出，C 是独立的参照。如果 B 比 A 更像 C，修正就是朝对的方向走的。

**附录：** 旧的做法是把某日期前 120 天的 311 field 推过 Step 3 的 sigmoid，得到 E[rating]（"如果 surveyor 来打分会打几分"）。它回答的是"311 能多好地复现 survey"，保留为附录对照（`A1_rating_snapshot.png`），不是主产出。

---

## 4. 结果解读

### 4.1 Table 1（`calib/table1.csv`）逐列读

| h | FE | α | h_w | ℓ(0) | ℓ(β̂) | Δℓ | β̂ | −c/b |
|---|---|---|---|---|---|---|---|---|
| 500 | month+surv | 1 | 400 | −49,492 | −48,888 | +605 | 0.88 | 0.81 |
| **800** | month+surv | 1 | 400 | −49,057 | −48,447 | **+610** | **0.86 [0.63, 1.08]** | 0.94 [0.66, 1.24] |
| 1200 | month+surv | 1 | 400 | −48,835 | −48,229 | +606 | 0.83 | 1.04 |
| 800 | month only | 1 | 400 | −61,271 | −60,850 | +421 | 0.93 | 1.02 |
| 800 | month+surv | 0.25 | 400 | −49,007 | −48,385 | +622 | 0.87 | 1.08 |
| 800 | month+surv | 1 | 600 | −49,057 | −48,310 | +748 | 1.03 | 0.94 |

ℓ(FE only) = −53,681（所有行相同）。

**怎么读：**
- 第一层比较：ℓ(FE only) −53,681 → ℓ(0) −49,057。只加入 raw 311 field，log-likelihood 提高 4,600。**311 数据本身对 litter 有很强的预测力。**
- 第二层比较：ℓ(0) → ℓ(β̂)，+610。**在此基础上，用路灯打折再提高 610，而且没有增加任何参数**（β 只是一个旋钮，不是回归系数）。
- β̂ 在三个 h 下是 0.83–0.88，稳定。−c/b 从 0.81 漂到 1.04，是一阶近似的正常表现。
- "month only" 那行 ℓ 差了 12,000，说明 surveyor effect 必须放。
- α = 0.25 几乎不改变 β̂（0.87），但 −c/b 变成 1.08，再次说明 −c/b 脆弱。
- h_w = 600 那行：β̂ = 1.03，Δℓ +748，比 400 更好。见 4.6。

**β̂ = 0.86 是什么意思：** 一条来自 w_R 比平均高一个标准差（0.53）的地方的 report，算 exp(−0.86 × 0.53) ≈ 0.63 条；来自低一个标准差的地方，算 ≈ 1.6 条。修正后的地图上，最爱举报和最不爱举报的地区之间，举报数被重新缩放了约 2.5 倍。

### 4.2 Bootstrap 和 CI（`bootstrap.csv`, `fig1` 右图）

200 次重抽样的 β̂ 分布：均值 0.84，sd 0.12，[0.63, 1.08]。**CI 不含 0**：修正方向是确定的。直方图里 −c/b（橙）比 profile（蓝）更分散（sd 0.15 vs 0.12），profile 更精确。

### 4.3 Torus null 和 placebo（`torus_null.csv`, `summary.json`）

见 3.5(b)(c)。`torus_null.csv` 里 50 个数，大部分接近 0（平移后的假地图没有帮助，甚至它自己最好的 β 就是 0），最大一个 226，仍然远低于真实的 592。

### 4.4 Table 2（`cv/table2.csv`）和 `05_cv.png`

| spec（h = 800） | OOF ℓ / visit | Somers' D | 比 raw 好多少 ± SE | 赢的 fold |
|---|---|---|---|---|
| FE only | −0.814 | 0.546 | | |
| raw field（S0） | −0.724 | 0.638 | — | — |
| 线性 raw + c·w̄（S1） | −0.710 | 0.653 | +0.0138 ± 0.0061 | 8/10 |
| **corrected，β in-fold（S2）** | **−0.710** | **0.652** | **+0.0137 ± 0.0042** | **9/10** |

**怎么读：**
- `ll_per_visit`：test 上每次 visit 的平均 log-likelihood。−0.814 → −0.724 → −0.710。数越大（越接近 0）越好。注意它比 in-sample 的 ℓ/visit（−48,447 / 72,084 ≈ −0.67）低，这是正常的：没见过的地区本来就更难预测。
- `Somers' D`：排序一致性。随机 = 0，完美 = 1。它回答"模型能不能把脏的地方排在干净的前面"。0.638 → 0.652。
- `gain_vs_S0 ± SE`：+0.0137 ± 0.0042，是 SE 的 3.3 倍。**这是修正有效的 out-of-sample 证据。**
- `folds_better` 9/10：只有 fold 4 输了（那个 fold 的区域对所有模型都最难，−0.82 到 −0.84）。
- `beta_fit`：10 个 fold 各自在 training 上选的 β，8 个选 0.8、2 个选 1.0，和全样本的 0.86 一致，说明 β 在不同区域之间是可以**transport** 的。
- `05_cv.png` 右图：40 个 block 的"观测平均 rating vs 预测平均 rating"，点贴着 45° 线，slope 1.01。模型在**区域层面**是 calibrated 的，不只是排序对。

**S1 和 S2 几乎一样好，怎么理解：** S1 是 S2 的线性近似，两者用了同一信息（field + w_R）。S2 的优势在于它的 β 有物理意义、可以用来画修正后的地图；S1 只能预测。

### 4.5 三张地图（`surface/summary.json`，`fig3_maps.png`）

| | 2021–2025（116,772 条） | 2022–2023（46,608 条） |
|---|---|---|
| 全城平均率，raw → corrected（每年每 1,000 stop） | 1.19 → 1.26 | 1.19 → 1.26 |
| 格子率 p1 / p50 / p99，raw | 0.01 / 0.90 / 5.85 | 0.02 / 0.89 / 6.16 |
| 格子率 p1 / p50 / p99，corrected | 0.01 / 0.89 / 8.62 | 0.02 / 0.86 / 8.84 |
| 修正系数 B/A，p1 / p50 / p99 | 0.49 / 0.97 / 3.88 | 0.50 / 0.97 / 3.73 |
| 街区层面与 C 的 Spearman，raw → corrected | 0.66 → 0.69 | 0.66 → 0.69 |
| 街区层面与实际平均 rating 的 Spearman，raw → corrected | 0.68 → 0.72 | 0.68 → 0.71 |

**怎么读：**
- **修正系数**是最直观的一行：最爱举报的地方（p1）举报被打到一半，最不爱举报的地方（p99）被放大将近 4 倍，中位数的格子几乎不动（0.97）。所以修正不是整体平移，而是把两端拉开。`A2` 图左下那张就是它：蓝 = 打折，红 = 放大。
- 修正后全城总量比原始高 6%（1.19 → 1.26），因为权重没有归一化：w_R 的零点定在"路灯加权的全城平均"上，而举报集中的地方 w_R 偏低一点。这是定义的结果，不是 bug，论文里说明即可。
- 两个窗口的数字几乎一样：举报的空间分布在五年里是稳定的。所以五年图作为产出，两年图用来和 survey 同期对照。
- **修正有效的证据在最后两行：** 不管和 C（survey 自己的平滑图）比还是和街区实际平均 rating 比，corrected 都比 raw 更接近 survey。幅度不大（+0.03 到 +0.04），和 CV 的结论一致：修正是"小而确定"的改善。
- C 本身：去掉 month 和 surveyor 效应后的平滑 rating，1% / 50% / 99% 分位是 0.93 / 1.47 / 2.03，覆盖 98.5% 的格子。
- 附录 snapshot（E[rating]）：2023-06-30 的中位数 1.26，街区 Spearman 0.71 vs 0.68；2025 年底 0.74 vs 0.70。

### 4.6 h_w = 400 还是 600？β 的单位问题

规则选了 400（reliability 刚过 0.95）。但 600 的敏感性行 fit 更好（Δℓ +748 vs +610），β̂ = 1.03。这不是矛盾：**β 的单位取决于 w_R 的尺度。** 600 m 的 w_R 更平滑，标准差 0.46（400 m 是 0.53），所以要更大的 β 才能达到同样的打折力度。可比的量是 β × sd(w_R)：400 m 下 0.86 × 0.53 = 0.45，600 m 下 1.03 × 0.46 = 0.47。几乎一样。论文里 β 应该连同 β·sd(w_R) 一起报。

为什么 600 fit 更好：更平滑的 w_R 噪声更少，打折的"误伤"更少。规则是事先定的，所以主结果用 400、600 作为敏感性；如果以后想改规则（比如 reliability ≥ 0.96），要在跑之前定，不能看了结果再改。

### 4.7 修正有效的四条证据（论文的骨架）

1. **In-sample**：Δℓ = +610，不加参数。
2. **不是随便一张地图都行**：torus null p = 0。
3. **Out-of-sample**：9/10 fold，+0.0137 ± 0.0042 ℓ/visit，in-fold β 和全样本一致。
4. **区域层面**：街区平均的 Spearman 0.68 → 0.71，block calibration slope ≈ 1。

---

## 5. 常见疑问

**Q：Δℓ = 610 大不大？**
对比：raw field 本身带来 4,600。修正是在那之上的 13%。另一个参照：torus null 里假地图平均只能带来 +11。

**Q：为什么 CV 里的数字（−0.71）和 in-sample 的（−0.67）不一样？**
In-sample 是模型在自己见过的数据上的表现，天然偏乐观。CV 是在没见过的区域上的表现，更诚实。论文应该报 CV 的。

**Q：surveyor effect 在 CV 里有没有偷看？**
有一点。打分人 A 的 test 区域留出了，但他在其他区域打的分还在 training 里，所以模型"认识"他。这对所有四个模型一视同仁，所以 S2 − S0 的**配对差**不受影响，但绝对水平（−0.71）略偏乐观。METHOD.md 里已声明。

**Q：为什么不直接用 CBG（街区）层面的模型？**
311 report 密度比 survey 高得多（每个 survey 点 37 m 内就有 report），用 kernel 可以在**每个 survey 点的精确坐标和日期**上评估 field，不受街区边界影响。旧 pipeline 做过对比，kernel 比最好的街区聚合高 10% 的相关性；新 pipeline 没有重做这个对比。

**Q：fields 里的 `num_null` 有 72084 × 50 × 4 = 1,400 万个数，为什么要存？**
因为 torus null 需要 50 张假地图 × 4 个 β 的 field，每个都是一次 kernel 求和。03 一次算完（3 分钟），04 只做 200 次回归。如果不存，04 每次重跑都要重新算几何。

**Q：profile_*.npz 的 thetas 是什么？**
每个 β 格点下，回归 fit 出的 68 个系数。bootstrap 每次重抽样要在附近的 β 重新 fit，从这些系数出发（warm start）只要几十步就收敛，从零开始要几百步。

**Q：β̂ 和之前旧 pipeline 的 1.04 为什么不同？**
旧的用 h_w = 600。新的 600 行给 1.03，一致。见 4.6。

---

## 6. 局限（论文里要写的）

- **假设不可直接检验：** "路灯真实故障率全城均匀"。老旧街区的灯可能真的坏得多。outage placebo 的小成分（5%）提示通用 311 活跃度确实混了一点进来。
- **w_R 是估的，不是已知的：** bootstrap 以 w_R 为给定（reliability 0.95），没有把 w_R 自身的不确定性传递进 CI。
- **β 的单位依赖 h_w**：报 β·sd(w_R)。
- **surveyor 在 CV 里的 leakage**：配对差免疫，绝对水平偏乐观。
- **距离是直线距离**，不是街道网络距离。
- **2025 地图是 extrapolation**：模型在 2022–23 校准，2025 的数字是"举报率变化推过 sigmoid"，不是验证过的垃圾测量。
- **1–4 rating 衡量的是一般 litter，不只是 dumping。** `visits.parquet` 里的 `bulk` 列（大件垃圾标记）可以做专门的检查，本版本没做。
