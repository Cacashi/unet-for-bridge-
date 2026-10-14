# 按截面位置分别微调与预测

样本 ID 最后一段按数值解析：0、0.0、-0 都属于 zero，其余有限数值
（包括负数、小数）属于 nonzero。不是按字符串是否包含 `_0` 判断。

两组共用 A 域 manifest，在 Dataset 中分别筛选，不需要复制图片或拆分 Excel。
远程微调输入来自 input1/a_finetune，红绿蓝标签来自 output1/a_finetune。
已人工清理的 5 个样本 ID 在配置中显式排除；清单不要求标签目录存在 combine 预览图。

两个微调任务分别从 outputs/pretrain_ab_domain_balanced/best.pt 初始化，
使用独立的优化器和输出目录。继承 40 epochs、1e-5 学习率，并关闭早停。
训练流程原有的每轮验证及 best.pt/last.pt 保存逻辑保留。

复用 A+B 预训练的 artifacts/splits.json 和 artifacts/feature_stats.json，
不重新划分工程组、不重算归一化参数。这两个文件必须来自对应预训练任务，
不可在预训练后用其他数据重建覆盖。若 A 域中有预训练划分未包含的组，准备脚本
会明确报错；需要先制定兼容的工程划分，不能静默重新随机划分。

在项目根目录执行以下命令（此次仅提供命令，未执行）：

```powershell
python scripts/prepare_data.py --config configs/finetune_a_position_base.yaml
python scripts/train_stage.py --config configs/finetune_a_zero.yaml
python scripts/train_stage.py --config configs/finetune_a_nonzero.yaml
```

预测使用 A 微调 manifest 和预训练时已留出的 test 工程组。
`--all --split test` 按固定种子从中选出 zero 2 条、nonzero 4 条，
其他 test 样本仍未进入微调训练。`--sample` 可显式预测其中任意样本。
真实值选点数量、红绿蓝计数输出及 XML 孔径换算规则保持原有协议。

```powershell
python scripts/predict.py --config configs/predict_position.yaml --all --split test
python scripts/predict.py --config configs/predict_position.yaml --sample 1_10_0 --split test
```

输出分别保存在 outputs/predictions/test_selected/zero/<sample>/ 和
outputs/predictions/test_selected/nonzero/<sample>/，JSON 记录分组和实际使用的 checkpoint。
旧版在 outputs/predictions/zero 和 nonzero 下生成的文件不会自动清除。
如果希望使用最后一轮模型，可以覆盖两组权重：

```powershell
python scripts/predict.py --config configs/predict_position.yaml --all --split test --checkpoint-zero outputs/finetune_a_zero/last.pt --checkpoint-nonzero outputs/finetune_a_nonzero/last.pt
```

原单模型预测接口（--config 配合 --checkpoint）继续保留。
