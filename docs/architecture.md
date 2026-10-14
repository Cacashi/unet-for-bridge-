# 架构索引

> 本文只记录已经在代码、配置或 `README.md` 中确认的结构。实验日志、临时调试结论和未复现的猜测不应写入这里。

## 项目目标

项目用于预测桥梁截面中的钢束孔位布局。输入同时包含半截面几何图像和工程表格参数，输出顶板、腹板、底板的允许区域，以及不同颜色类别孔位的中心热力图、亚像素偏移和半径。

## 端到端数据流

```text
half-section/input/*.png ─┐
                           ├─ 数据准备 ─> artifacts/manifest.jsonl
half-section/output/* ────┤             artifacts/splits.json
hole_stats_AB_hole.xlsx ──┘             artifacts/feature_stats.json

输入截面轮廓 + 实体 mask ─┐
                           ├─ Conditional U-Net + FiLM ─> 多任务输出头
工程参数 / one-hot 特征 ───┘

热力图、偏移、半径、区域预测 ─> 候选生成 ─> 几何/区域/重叠/数量约束 ─> 孔位 JSON 与可视化
```

## 主要模块

| 模块 | 位置 | 职责 |
| --- | --- | --- |
| 配置加载 | `src/bridge_hole/config.py` | 读取 YAML 并合并 `base` 配置 |
| 数据清单 | `src/bridge_hole/data/manifest.py` | 读取工作簿和标签，生成样本清单与特征统计 |
| 数据划分 | `src/bridge_hole/data/splits.py` | 按 `group_id` 划分，并绑定重复内容，避免泄漏 |
| 数据集 | `src/bridge_hole/data/dataset.py` | 读取图像、构造双通道输入、归一化表格特征和增强 |
| 目标编码 | `src/bridge_hole/data/targets.py` | 将孔位编码为热力图、偏移、半径和计数监督 |
| 几何处理 | `src/bridge_hole/geometry.py` | 构造实体区域 mask，检查孔中心和几何有效性 |
| 模型 | `src/bridge_hole/models/` | U-Net 编解码器、FiLM 条件融合和多任务预测头 |
| 训练引擎 | `src/bridge_hole/engine.py` | epoch、损失、AMP、checkpoint、W&B 和验证 |
| 后处理 | `src/bridge_hole/postprocess.py` | 候选解码、区域/边界/重叠/数量约束及渲染 |
| 命令脚本 | `scripts/` | 数据准备、训练、推理和颜色点分析入口 |

## 模型接口

- 图像输入为 2 个通道：截面轮廓和填充后的实体区域 mask。
- 表格特征由连续特征和 `top/web/bottom` one-hot 特征组成；当前 `tabular_dim` 配置为 69。
- 表格特征通过 MLP 生成 FiLM 条件，在 U-Net 瓶颈层调制视觉特征。
- 当前 `output_stride` 固定为 4，默认图像尺寸为 `1024×512`。
- 预测头覆盖孔位热力图、`dx/dy`、半径、类别区域和相关密度/计数信息。

## 运行阶段

1. `scripts/prepare_data.py` 生成固定 manifest、split 和特征统计。
2. `scripts/train_stage.py --config configs/pretrain_b.yaml` 对 train group 中的 B 数据进行预训练。
3. `scripts/train_stage.py --config configs/finetune_a.yaml` 从 B checkpoint 开始进行 A 微调。
4. `scripts/predict.py` 读取 checkpoint，解码孔位并写出 JSON、三类颜色图和 combine 图。

## 重要边界

- 标签是半截面；设计计数来自完整截面，但推理保持半截面，不自动镜像。
- `decode.class_bands` 相对于实际截面纵向包围盒，而不是整张含留白图片。
- 当前输入通道和区域预测头的变更会使旧 checkpoint 不兼容；代码或输入定义变化后应重新准备数据并重新训练。
