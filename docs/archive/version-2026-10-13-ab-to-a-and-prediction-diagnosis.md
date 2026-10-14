---
title: 版本归档：A+B 域均衡训练、A 微调数据与预测诊断
date: 2026-10-13
project: 桥梁钢束孔位布局预测
tags:
  - UNet
  - 孔位预测
  - 域微调
  - 规则解码
  - 实验归档
status: diagnosis-complete
---

# 版本归档：A+B 域均衡训练、A 微调数据与预测诊断

## 归档范围

本笔记记录 2026-10-13 当前版本中已经完成的数据与代码改进、A+B → A 训练方案、批量预测能力，以及对现有 test 预测结果的量化分析。本文是知识快照，不表示所有列出的缺陷均已修复。

## 本版本已完成的改进

### 1. 训练流程调整为 A+B → A

预训练阶段使用 A、B 两个域，并启用域均衡采样：

```yaml
train:
  stage: pretrain_ab
  domains: [A, B]
  domain_balanced_sampling: true
  epochs: 60
  learning_rate: 0.0003
  output_dir: outputs/pretrain_ab_domain_balanced
```

A 微调阶段计划加载 A+B 阶段的最优权重，仅使用 A 域：

```yaml
train:
  stage: finetune_a_from_ab
  domains: [A]
  epochs: 40
  learning_rate: 0.00001
  checkpoint: outputs/pretrain_ab_domain_balanced/best.pt
  output_dir: outputs/finetune_a_from_ab
```

对应配置：

- `D:\Unet\configs\pretrain_ab.yaml`
- `D:\Unet\configs\finetune_a_from_ab.yaml`

训练引擎已支持显式 `train.domains` 和 A/B 域均衡采样。

### 2. 建立 A 域专用微调数据

从 `hole_stats_AB_hole.xlsx` 中提取全部 A 域样本，不使用 overlap 结果进行筛选：

- A 样本数：159
- 输入目录：`D:\Unet\half-section\input\a_finetune`
- 输出目录：`D:\Unet\half-section\output\a_finetune`
- 每个样本包含 red、green、blue、combine 四类输出文件

### 3. 重建并对齐 data_finetune.xlsx

`D:\Unet\data_finetune.xlsx` 已按 `data_train.xlsx` 的生成逻辑重新构建：

- 共 159 条 A 数据；
- 与 `data_train` 保持相同的 84 个字段及字段顺序；
- 补充 `dataset_report.xlsx` 中的几何与质量信息；
- 补充 `design_parameter.xlsx` 中的设计规则及 one-hot 信息；
- 半截面数量使用 `ceil(完整截面数量 / 2)`；
- 未读取 `overlap_summary.xlsx`；
- `overlap_flag` 为空，`overlap_status` 为 `not_checked`，且不参与筛选。

可复现脚本：

- `D:\Unet\scripts\build_data_finetune_intermediate.py`
- `D:\Unet\scripts\create_data_finetune_workbook.mjs`

### 4. 批量预测 test 集

`scripts/predict.py` 已增加 `--all` 参数，保留原有单样本预测方式。

批量预测命令：

```powershell
python scripts/predict.py `
  --config configs/finetune_a_from_ab.yaml `
  --checkpoint outputs/finetune_a_from_ab/best.pt `
  --split test `
  --all
```

结果按样本分别写入：

```text
D:\Unet\outputs\predictions\{sample}\
```

## 当前预测结果概览

本次分析对象是 `outputs/predictions` 中 24 个最新的 A 域 test 结果。24 个结果均显示 `decode_status=complete`。

需要注意：这里的 complete 只表示规则解码器按照输入的目标数量凑齐了点，不表示模型独立预测正确。

### 点位误差

以下为每个预测点到同类别最近真值点的距离。该指标没有进行一一匹配，因此属于偏乐观估计。

| 类别 | 预测点数 | 平均最近距离 | 10 px 内 | 20 px 内 | 平均选择得分 |
|---|---:|---:|---:|---:|---:|
| 顶板 | 202 | 27.20 px | 32.2% | 63.4% | 0.36 |
| 腹板 | 65 | 264.95 px | 0% | 0% | 0.01 |
| 底板 | 151 | 33.12 px | 27.2% | 52.3% | 0.19 |

腹板是当前最严重的问题。以 `1_21_0` 为例：

- 腹板真值 y 坐标约为 1441–1601；
- 腹板预测 y 坐标约为 1805–1875；
- 该样本腹板平均偏差约 245 px。

### 模型计数头

对真值数量大于 0 的样本统计：

| 类别 | MAE | 平均偏差 |
|---|---:|---:|
| 顶板 | 1.95 | +0.56 |
| 腹板 | 2.12 | -0.74 |
| 底板 | 0.53 | +0.07 |

### 半径差异

| 类别 | 预测平均半径 | 真值平均半径 |
|---|---:|---:|
| 顶板 | 8.44 px | 6.83 px |
| 腹板 | 8.55 px | 7.33 px |
| 底板 | 8.94 px | 6.85 px |

## 已确认的问题与成因

### P0：预测过程使用真值数量

`scripts/predict.py` 当前从 `batch["targets"]["center_mask"]` 计算红、绿、蓝点数，然后要求解码器严格选出相同数量。

因此：

- `decoded_counts == target_half_counts` 是代码约束的结果；
- 不能据此判断模型计数正确；
- 当前预测仍依赖标签，不能作为真正未知样本的完整推理流程。

### P0：工程规则网格与训练标签规则不一致

以 `1_21_0` 的 12 束规格为例：

- XML 中 `pass_to_pass_center_mm = 174.6 mm`；
- 当前标定为 `5 mm/px`；
- 解码间距因此为 `34.92 px`；
- 标签阵列实际相邻距离约为 `40 px`，即约 `200 mm`。

预测只能从 34.92 px 的候选网格中选点，无法与 40 px 的标签阵列严格重合。这不是单纯增加训练轮数能够解决的问题。

需要在以下两条路线中明确选择：

1. 以现有标签为目标：从标签生成规则中统一间距、半径和阵列相位；
2. 以工程 XML 为目标：按照 XML 规则重新生成训练标签。

### P0：固定 top/web/bottom 高度区间与真实孔位不一致

当前区域配置为：

```yaml
top: [0.0, 0.40]
web: [0.40, 0.60]
bottom: [0.60, 1.0]
```

该比例相对于整个截面高度，而真实腹板束经常位于腹板上部，不一定处于总高度 40%–60% 区间。

同时 grid 解码会把热力图概率乘以区域头概率。区域定义错误时，真实腹板位置的响应会被区域门控压低。当前选中腹板点平均得分只有 0.01，中位数接近 0，说明腹板位置主要是在低置信度候选中强制凑数。

### P1：网格合法不等于阵列结构正确

当前 `best_phase` 只根据合法区域内可容纳点数和左上优先选择网格相位，不建模：

- 阵列起点；
- 行数和列数；
- 每行点数；
- 行间距与列间距；
- 转角处的排列模式；
- 阵列整体居中方式。

随后按热力图分数选择前 N 个格点，因此可能出现“所有点都在规则格点上，但整体不是目标规则阵列”的情况。

### P1：半径定义不一致

grid 模式的半径来自 XML 管道直径换算，训练真值半径来自标签连通域拟合。两者定义不同，使预测圆平均比标签圆更大。

### P1：模型腹板分支相对较弱

训练集中 top 点数明显多于 web 点数，且热力图 focal loss 使用所有类别总正样本数共同归一化。结合错误区域门控，腹板监督更容易被顶板主导。当前腹板选择分数显著低于顶板和底板。

### P1：实验来源不可追溯

当前预测 JSON 没有保存：

- checkpoint 路径或哈希；
- 配置文件路径或配置快照；
- manifest/split 哈希；
- Git commit；
- 推理时间与代码版本。

当前 `outputs` 下也没有可确认的 `.pt` 文件，因此不能仅凭预测目录证明结果来自哪一个训练阶段。

### 非问题：部分样本预测完全相同

`1_3_0` 与 `3_5_0` 等样本预测完全相同。核查后发现它们的输入图像 SHA256、连续特征和 one-hot 条件完全相同，因此相同输出是确定性模型的正常表现，不是模型坍缩。

## 当前配置与产物缺口

截至归档时：

1. `configs/finetune_a_from_ab.yaml` 仍指向不存在的：

   ```text
   half-section/input1/a_finetune
   half-section/output1/a_finetune
   ```

   实际生成目录为：

   ```text
   half-section/input/a_finetune
   half-section/output/a_finetune
   ```

2. 以下 A 微调数据准备产物尚不存在：

   ```text
   artifacts/finetune_a_manifest.jsonl
   artifacts/finetune_a_splits.json
   artifacts/finetune_a_feature_stats.json
   ```

3. `outputs/pretrain_ab_domain_balanced/best.pt` 和 `outputs/finetune_a_from_ab/best.pt` 当前未在项目中检测到。

4. A 微调配置仍继承 `early_stopping_patience: 10`。若要完全关闭早停，需要在微调配置中显式设置：

   ```yaml
   early_stopping_patience: 0
   ```

## 建议改进顺序

1. **先修正评估流程**：未知样本预测不能读取标签点数；使用计数头、设计输入数量或显式外部约束。
2. **统一规则和标签**：确定 XML 规则或当前标签哪一个是最终监督标准，并统一间距、孔径和边界定义。
3. **修正类别区域定义**：从截面拓扑识别顶板、腹板、底板，避免只用截面总高度比例切分。
4. **暂时隔离区域门控**：在区域头未校准前分别评估有/无 region gate，确认腹板热力图是否被错误压制。
5. **引入阵列级参数**：预测网格原点、行列数、行列间距或采用带模板约束的全局分配，而不是仅做候选点 top-N。
6. **统一半径来源**：训练与推理使用相同的物理直径或相同的标签连通域定义。
7. **补充实验追踪**：每次预测写入 checkpoint、配置、manifest、split 和代码版本信息。
8. **最后再重新训练**：完成上述一致性修正后，再执行 A+B 域均衡训练和 A 微调，否则增加 epoch 很可能只会拟合互相冲突的目标。

## 推荐验证指标

后续不应只检查生成图片，建议固定记录：

- 每类 count MAE；
- 一一匹配后的点位 MAE/RMSE；
- 10 px、20 px、40 px 内匹配率；
- 漏检率和误检率；
- 行列间距误差；
- 阵列行数、列数及每行点数准确率；
- 边界违规数量；
- 点间距违规数量；
- 不使用真值数量时的完整解码成功率；
- A、B 两域分别统计的指标。

## 本次归档结论

当前预测已经具备规则候选、边界检查、数量约束和批量输出能力，但尚不能作为真实未知样本预测结果：数量来自标签，候选网格与标签规则不一致，固定区域切分导致腹板严重错位。下一阶段应优先修正评估泄漏和规则一致性，再调整模型结构或继续训练。

## 后续明确的工程决策与推理更新

用户确认以下目标：

1. 预测结果应给出红、绿、蓝三类各自的点数；
2. 位置规则以工程 XML 为标准；
3. 孔半径统一按 XML 管道内径换算。

据此更新 `scripts/predict.py`：

- 红/绿/蓝对应 top/web/bottom。
- 选点数量继续采用标签真值中心数，作为有标签验证/测试协议的一部分；真值计数会决定候选筛选和最终输出点数。
- 模型 `count` 头的连续输出仍写入 `predicted_half_counts`，作为独立计数预测供比较，但不控制本次选点数。
- JSON 将用于解码的点数明确写入 `selection_half_counts`（top/web/bottom）和 `selection_color_counts`（red/green/blue）。
- grid 候选间距使用工程 XML 规则；heatmap/grid 两种解码生成的孔半径统一覆盖为 `XML 管道内径 / 2 / mm_per_pixel`。
- 若某类真值要求选点但对应束型缺少有效 XML 规格，推理会报错，不静默回退到模型半径。

因此本版本仍是“真值数量条件下的位置预测”，并未实现未知样本上的端到端自动计数解码。模型计数头可单独评估；XML 决定候选物理间距和孔半径。

### 尚未解决的约束

- 现有固定 `class_bands` 和 region head 仍可能把腹板区域定位错；工程 XML 本身没有定义图像中的 top/web/bottom 分区边界，因此还需要根据截面轮廓的拓扑或明确的结构边界规则建立分类区域。
- 当前 grid 相位仍按合法区域内“最大容纳数、偏左上”选择，没有按设计图的明确阵列锚点/起点规则校准。
- 训练标签仍来自现有图像标注，且其间距和半径可能与 XML 不同。若 XML 是最终标准，需决定是否依据 XML 重新生成监督标签后再训练；仅修改推理端不能使训练标签与工程规则自动一致。
- 本次推理策略调整尚未重跑预测；新的 JSON 字段和 XML 半径效果还没有对应的新结果指标。
