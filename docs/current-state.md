# 当前状态

更新时间：2026-10-12

历史进度快照：[`docs/archive/progress-2026-10-12.md`](archive/progress-2026-10-12.md)

## 当前目录与职责

| 路径 | 当前用途 | 是否应同步到 Vault |
| --- | --- | --- |
| `src/bridge_hole/` | 数据、模型、训练和后处理源码 | 否；只同步稳定结论 |
| `scripts/` | 数据准备、训练、推理和分析入口 | 否；可在架构文档中引用 |
| `configs/` | base、B 预训练和 A 微调配置 | 否；配置变更应总结为决策 |
| `tests/` | 几何、目标、清单和划分测试 | 否；失败模式可整理到 troubleshooting |
| `half-section/` | 输入、输出和分析中间数据 | 否 |
| `artifacts/` | manifest、split、特征统计等生成文件 | 否；已确认的数据规则可同步 |
| `outputs/`、`wandb/`、`*.inspect.ndjson` | checkpoint、预测、日志和检查产物 | 否 |
| `docs/` | 项目稳定知识索引 | 是：按需整理到 Vault |

## 已确认的运行方式

```powershell
# 安装
python -m pip install -e .

# 生成数据清单和固定划分
python scripts/prepare_data.py --config configs/base.yaml

# 训练
python scripts/train_stage.py --config configs/pretrain_b.yaml
python scripts/train_stage.py --config configs/finetune_a.yaml

# 推理示例
python scripts/predict.py --config configs/finetune_a.yaml `
  --checkpoint outputs/finetune_a_regularized/best.pt `
  --sample 0_1_0

# 测试
python -m pytest
```

CUDA 12.8 环境需先安装对应的 PyTorch wheel；默认配置针对约 4GB 显存，使用物理 batch 1、梯度累积 2、FP16 AMP 和 channels-last。

## 当前知识缺口

- 当前工作区未检测到 Git 元数据，因此文档暂不能引用 commit、分支或 PR 作为版本锚点。
- `README.md` 已覆盖主流程，但尚未记录模型输出张量的完整字段表和 checkpoint 兼容矩阵。
- 运行产物很多，尚未建立统一的实验命名、保留和归档规则。
- Vault 当前不作为本次整理的读取源；同步目标目录按用户约定为 `D:\notebook\unet-定位生成\项目\unet\`。

## 项目文档与 Vault 的同步流程

同步遵循“项目内先确认，Vault 后整理”的单向优先原则：

1. 在 `D:\Unet\docs\` 修改或新增内容，并在文档中标注更新时间。
2. 只选择稳定、可复用、已验证的段落；不复制日志、checkpoint、临时脚本、完整实验输出或未确认结论。
3. 用户明确要求同步时，将选定内容复制或改写到 `D:\notebook\unet-定位生成\项目\unet\`，保留 Vault 原有组织方式。
4. 同步后在项目文档中记录同步日期和目标笔记；若 Vault 中出现新结论，先人工确认，再回写项目 `docs/`。
5. 若两边冲突，以代码、配置、测试和本次已验证结果为准；不要自动覆盖 Vault 中的历史笔记。

建议的同步映射：

| 项目文档 | Vault 笔记建议 |
| --- | --- |
| `architecture.md` | `项目/unet/架构` |
| `decisions.md` | `项目/unet/决策记录` |
| `current-state.md` | `项目/unet/当前状态` |
| `troubleshooting.md` | `项目/unet/故障排查` |

当前不建议使用自动双向同步。若未来引入脚本或 MCP，应至少增加：明确的允许路径、只读默认值、变更预览、冲突检测和同步日志。

## Vault 同步记录

- 2026-10-13：将 `docs/archive/version-2026-10-13-ab-to-a-and-prediction-diagnosis.md` 同步到 `D:\notebook\unet-定位生成\项目\unet\版本归档-2026-10-13-A+B到A微调与预测诊断.md`。归档记录的当前推理约定为：保留真实值点数控制选点，模型计数头单独输出用于比较；工程 XML 决定网格间距和管径半径。区域分割、网格起点及训练标签规则仍待对齐。
