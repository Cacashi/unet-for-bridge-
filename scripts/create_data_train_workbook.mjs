import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from 'file:///C:/Users/%E8%A9%B9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const build = JSON.parse(await fs.readFile('artifacts/data_train_build.json', 'utf8'));
const excludedSamples = new Set(['0_7_-51']);
build.records = build.records.filter(record => !excludedSamples.has(String(record.sample_id)));
build.metadata.row_count = build.records.length;
build.metadata.filter = 'overlap == 0; exclude invalid geometry samples';
build.metadata.excluded_samples = [...excludedSamples];
await fs.writeFile('artifacts/data_train_build.json', JSON.stringify(build, null, 2), 'utf8');
const wb = Workbook.create();
const main = wb.worksheets.add('data_train');
const rules = wb.worksheets.add('design_rules');
const dict = wb.worksheets.add('field_dictionary');
const summary = wb.worksheets.add('build_summary');

const records = build.records;
const columns = Object.keys(records[0]);
const matrix = [columns, ...records.map(r => columns.map(c => r[c] ?? null))];

const headerFmt = { fill: '#1F4E78', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' }, horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true };
const bodyFmt = { font: { name: 'Arial', size: 9 }, verticalAlignment: 'center' };
main.getRangeByIndexes(0, 0, matrix.length, columns.length).values = matrix;
main.getRangeByIndexes(0, 0, 1, columns.length).format = headerFmt;
main.getRangeByIndexes(1, 0, records.length, columns.length).format = bodyFmt;
main.freezePanes.freezeRows(1);
main.getRangeByIndexes(0, 0, matrix.length, columns.length).format.wrapText = false;
main.getRange('A1:CH1').format.rowHeight = 32;
main.getRange('A:CH').format.columnWidth = 14;
main.getRange('A:A').format.columnWidth = 18;
main.getRange('B:B').format.columnWidth = 9;
main.getRange('C:C').format.columnWidth = 11;
main.getRange('H:H').format.columnWidth = 15;
main.getRange('AI:CH').format.columnWidth = 16;
main.getRange('A1:CH731').format.autofitColumns();
main.getRange('A1:CH1').format.columnWidth = 16;
main.getRange('A1:CH731').format.autofitRows();

const ruleCols = Object.keys(build.rules[0]);
const ruleMatrix = [ruleCols, ...build.rules.map(r => ruleCols.map(c => r[c] ?? null))];
rules.getRangeByIndexes(0, 0, ruleMatrix.length, ruleCols.length).values = ruleMatrix;
rules.getRangeByIndexes(0, 0, 1, ruleCols.length).format = headerFmt;
rules.getRangeByIndexes(1, 0, build.rules.length, ruleCols.length).format = bodyFmt;
rules.freezePanes.freezeRows(1);
rules.getRange('A1:N22').format.autofitColumns();
rules.getRange('A1:N1').format.rowHeight = 32;

const entries = [
  ['sample_id','样本唯一标识','由 sample 作为主键保留','overlap_summary + hole_stats + dataset_report'],
  ['group_id','重叠分组编号','原 id 重命名；仅保留 overlap=0','overlap_summary.id'],
  ['sample_order','样本顺序编号','原 id_num 重命名','overlap_summary.id_num'],
  ['dataset_domain','数据域 A/B','原数据集来源字段标准化','hole_stats_AB_hole.数据集来源AB'],
  ['count_scope','计数口径','固定为 half_section，表示半截面','规则约定'],
  ['mm_per_pixel','像素比例','固定为 5.0 mm/pixel','规则约定'],
  ['full_hole_count / half_hole_count','孔洞数量（全/半截面）','half=ceil(full/2)，保留中心线孔洞','hole_stats_AB_hole.holes_count'],
  ['full_*_component_count / half_*_component_count','颜色连通域数量（全/半截面）','half=ceil(full/2)','hole_stats_AB_hole.*_components'],
  ['full_*_region_count / half_*_region_count','区域数量（全/半截面）','half=ceil(full/2)','dataset_report.*_regions'],
  ['full_*_entity_count','实体/文件分组数量（全截面）','不做除二；明确其为实体分组计数','dataset_report.*_entities'],
  ['geometry_*','几何窗口字段','保持源全截面几何，不伪造半截面坐标','dataset_report'],
  ['*_strand_specs','钢束规格原始/归一化文本','保留逗号、波浪号等混合规格','hole_stats_AB_hole.top/web/bottom'],
  ['*_single_strand_count','单一规格束数','仅单值规格可解析；混合规格置空','design_parameter 对照'],
  ['*_rule_lookup_status','规则查找状态','single_spec 或 mixed_spec_requires_tendon_level','design_parameter'],
  ['*_onehot_specs','规格 one-hot/组合信息','保留原 one-hot 字段用于追溯','hole_stats_AB_hole.*_onehot'],
  ['*_pass_* / *_anchor_* / *_jack_*','设计规则距离（mm）','来自通用设计参数；角色级数据缺失时仅作候选约束参考','design_parameter.钢束空间(铁路) 加大波纹管内径'],
  ['rule_mode','规则模式','第一版固定按通过束处理','规则约定'],
  ['overlap_flag / overlap_status','重叠质检状态','只保留 overlap_flag=0 的样本','overlap_summary'],
  ['label_quality_status / *_count_mismatch','标签质量字段','保留源质检结果','dataset_report'],
];
dict.getRangeByIndexes(0, 0, entries.length + 1, 4).values = [['字段名','中文含义','修改/转换说明','来源'], ...entries];
dict.getRange('A1:D1').format = headerFmt;
dict.getRangeByIndexes(1, 0, entries.length, 4).format = bodyFmt;
dict.freezePanes.freezeRows(1);
dict.getRange('A1:D20').format.wrapText = true;
dict.getRange('A:A').format.columnWidth = 34;
dict.getRange('B:B').format.columnWidth = 28;
dict.getRange('C:C').format.columnWidth = 52;
dict.getRange('D:D').format.columnWidth = 42;
dict.getRange('A1:D20').format.autofitRows();

const summaryRows = [
  ['项目','值'],
  ['输出表','data_train'],
  ['筛选条件','overlap == 0；排除无效几何样本 0_7_-51'],
  ['保留样本数', records.length],
  ['计数口径','half_section（半截面）'],
  ['半截面换算','ceil(full_count / 2)，保留中心线项'],
  ['mm_per_pixel', 5.0],
  ['规则模式','pass_only（第一版全部按通过束处理）'],
  ['角色级逐束表','未接入；混合规格标记为 mixed_spec_requires_tendon_level'],
  ['锚固边距/千斤顶边距','暂不作为硬约束启用，保留通用设计规则列供后续角色数据接入'],
  ['来源','overlap_summary.xlsx; hole_stats_AB_hole.xlsx; design_parameter.xlsx; dataset_report.xlsx'],
];
summary.getRangeByIndexes(0, 0, summaryRows.length, 2).values = summaryRows;
summary.getRange('A1:B1').format = headerFmt;
summary.getRangeByIndexes(1, 0, summaryRows.length - 1, 2).format = bodyFmt;
summary.getRange('A:A').format.columnWidth = 28;
summary.getRange('B:B').format.columnWidth = 92;
summary.getRange('A1:B11').format.wrapText = true;
summary.getRange('A1:B11').format.autofitRows();

wb.recalculate();
await fs.mkdir('artifacts/data_train_preview', { recursive: true });
const preview = await wb.render({ sheetName: 'data_train', range: 'A1:O14', scale: 1, format: 'png' });
await fs.writeFile('artifacts/data_train_preview/data_train_top.png', new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save('data_train.xlsx');
console.log(JSON.stringify({ output: 'data_train.xlsx', rows: records.length, columns: columns.length, rules: build.rules.length }));
