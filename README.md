# unet-for-bridge-

## 桥梁钢束孔位布局预测

## 项目文档

稳定的架构、设计决策、当前状态和故障排查索引位于 [`docs/`](docs/)：

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/decisions.md`](docs/decisions.md)
- [`docs/current-state.md`](docs/current-state.md)
- [`docs/troubleshooting.md`](docs/troubleshooting.md)

项目文档默认维护在代码项目内；只有明确要求时，才将经过确认的内容整理到独立的 Obsidian Vault。

本项目实现条件式 U-Net：截面轮廓及填充后的实体区域 mask 作为双通道图像进入编码器，表格工程参数进入 MLP，并在 U-Net 瓶颈层通过 FiLM 融合。模型同时预测顶板、腹板、底板的允许区域，以及各类孔的中心热力图、`dx/dy` 亚像素偏移和半径；推理时先用区域概率门控候选，再进行边界、类别区域、重叠和数量约束筛选。

## 环境

项目使用 Python 3.10 或 3.11：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

CUDA 12.8 环境需要先安装官方 CUDA wheel，再安装本项目：

```powershell
python -m pip uninstall -y torch
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一项必须输出 `True`。默认训练配置针对 4GB 显存：物理 batch 为 1，梯度
累积 2 次（等效 batch 为 2），使用 FP16 AMP 和 channels-last 内存格式。
编码器使用 stride-2 stem，避免在完整的 `1024×512` 分辨率上执行昂贵的深层
卷积；验证阶段使用 inference mode，CUDA 环境使用 fused AdamW。
AMP 从较保守的 loss scale `16` 开始；偶发溢出的 optimizer step 会被安全跳过并自动
降低 scale，连续多次溢出时则输出具体参数和各项 loss 后停止训练。

## 数据准备

默认读取：

- `half-section/input/{sample}.png`
- `half-section/output/{sample}_red.png`
- `half-section/output/{sample}_green.png`
- `half-section/output/{sample}_blue.png`
- `hole_stats_AB_hole.xlsx`

```powershell
python scripts/prepare_data.py --config configs/base.yaml
```

该命令生成 `artifacts/manifest.jsonl` 和固定的 `artifacts/splits.json`。划分单位是 `group_id`，并会额外按输入截面图像内容去重；具有相同输入内容的不同 group 会被绑定到同一集合，避免内容级泄漏。同组的 A/B 和所有 distance 不会跨集合。
默认还会剔除标注中心落在截面实体区域外的异常样本，避免几何监督互相冲突。
本次模型输入已由单通道轮廓改为“轮廓 + 实体 mask”双通道，并新增区域预测头，因此旧 checkpoint
不兼容；更新代码后必须重新执行数据准备并从 B 预训练开始重新训练。

## 训练

首次使用 W&B 时先安装依赖并登录：

```powershell
python -m pip install -e .
wandb login
```

```powershell
python scripts/train_stage.py --config configs/pretrain_b.yaml
python scripts/train_stage.py --config configs/finetune_a.yaml
```

B 预训练仅使用 train group 中的 B；A 微调仅使用同一批 train group 中的 A。验证组和测试组对应的 A/B 均不会参与训练。训练集默认按 group 反频率采样，并对 train 样本启用小幅同步几何增强；验证和测试不增强。

W&B 默认开启，项目名为 `bridge-hole-layout`。训练过程中会按 `wandb.log_interval`
记录 batch loss，并在每个 epoch 后记录训练集和验证集的汇总 loss。可在
`configs/base.yaml` 中修改项目名、设置运行名称，或通过 `wandb.enabled: false` 关闭上传。

## 推理

```powershell
python scripts/predict.py --config configs/finetune_a.yaml --checkpoint outputs/finetune_a_regularized/best.pt --sample 0_1_0
```

推理结果包括孔位 JSON、三个颜色图和保留原始黑色截面的 combine 图。

### 规则网格模式

当前规则网格基线使用半截面数量、5 mm/像素标定和铁路通过束参数。区域头为四分类
（背景、顶板、腹板、底板），推理配置 `decode.mode: grid` 时，先对截面区域按规格
腐蚀生成合法中心区，再按 `pass_to_pass_center_spacing` 生成规则候选格点，最后用
热力图分数选择每类前 n 个位置。`scripts/oracle_grid.py` 可在不依赖模型预测区域的
情况下评估规则阵列对现有标注中心的容量和覆盖率。

当前 manifest 明确使用 `count_scope: half`、`mm_per_pixel: 5.0` 和
`rule_mode: pass_only`。逐束 `tendon_id/role` 接入后，规则解码必须升级为带角色的
全局位置分配；在此之前不能把匿名孔位伪造为锚固束。

## 重要配置

- `data.label_scope`：标签图片是半截面；训练计数由标签连通域直接统计。
- `data.design_count_scope`：工作簿中的红/绿/蓝计数是完整截面设计计数，只作为条件特征及元数据保存。
- `data.radius_source`：当前实现为 `mask_component`，从彩色连通域拟合半径。
- 推理始终保持半截面到半截面，不执行镜像，也不会自动生成完整截面点分布。
- `decode.class_bands` 相对于实际截面的纵向包围盒定义，而不是相对于含有大量留白的整张图片。
- `model.output_stride`：当前架构固定为 4。
