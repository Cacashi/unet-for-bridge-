# 故障排查

## 安装与环境

### Python 版本不匹配

项目要求 Python `>=3.10,<3.12`。优先使用 Python 3.10 或 3.11 创建虚拟环境，并在虚拟环境中执行 `python -m pip install -e .`。

### CUDA 不可用

检查：

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果最后输出不是 `True`，确认安装的是与驱动匹配的 CUDA wheel，并确认训练配置中的 `train.device` 没有误设为 `cuda` 而实际环境只有 CPU。CPU 可用于小规模测试，但默认训练配置按 CUDA 设计。

## 数据准备

### 找不到输入或工作簿

检查 `configs/base.yaml` 中的 `data.root`、`workbook`、`input_dir` 和 `output_dir`。默认路径分别指向项目根目录下的 `hole_stats_AB_hole.xlsx`、`half-section/input` 和 `half-section/output`。

### 划分出现泄漏或结果不稳定

重新运行数据准备后检查 `artifacts/splits.json`。划分单位是 `group_id`，同组 A/B 和 distance 不应跨集合；启用 `deduplicate_content` 时，相同输入内容的 group 也应绑定在一起。不要手工编辑生成的 split 作为长期修复。

### 几何样本被剔除

默认启用 `exclude_invalid_geometry`。这表示标签中心落在实体区域外的样本会被排除，以避免几何监督冲突。先检查截面 mask、标签图和连通域提取结果，再决定是否修复源数据；不要直接关闭检查来掩盖数据问题。

## 训练与 checkpoint

### 加载 checkpoint 时报尺寸不匹配

当前模型使用双通道输入并包含区域预测头，旧模型可能无法加载。重新执行：

```powershell
python scripts/prepare_data.py --config configs/base.yaml
python scripts/train_stage.py --config configs/pretrain_b.yaml
```

之后再使用新的 B checkpoint 进行 A 微调。

### AMP 出现非有限梯度

默认 loss scale 从 16 开始，训练引擎会跳过偶发溢出的 optimizer step 并降低 scale；达到连续溢出上限时会停止并输出诊断。应先保留诊断信息，检查输入、学习率和 loss，再考虑降低学习率或关闭 AMP 做对照实验。

### 显存不足

默认物理 batch 为 1。优先保持输入尺寸和模型结构不变，检查是否启用了 channels-last、验证阶段是否使用 inference mode，并确认没有同时打开多个训练进程。若仍不足，再记录变更后的配置和效果。

## 推理与结果

### 预测结果超出截面或类别区域

检查 `decode.use_region_gate`、`enforce_class_bands`、`boundary_tolerance` 和 `min_clearance`。`class_bands` 是相对于实际截面纵向包围盒的比例，不是整张图片的比例。

### 结果数量或重叠不符合预期

确认输入的设计计数、标签范围和 checkpoint 阶段是否匹配。后处理会执行边界、类别区域、重叠和数量约束；先比较原始候选与约束后的结果，再调整解码参数。

### 误以为输出应包含完整截面

当前推理保持半截面到半截面，`geometry.mirror_full_section` 默认为 `false`。需要完整截面时，必须明确指定镜像策略、对称轴和容差，并单独验证镜像结果。

## 文档与同步

不要把 `wandb/`、`outputs/`、`artifacts/` 或临时检查文件直接复制到 Obsidian。应将复现步骤、根因、修复和验证结果整理成短条目，再同步到 Vault 的故障排查笔记。
