import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "file:///C:/Users/%E8%A9%B9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = path.resolve(".");
const build = JSON.parse(await fs.readFile(path.join(root, "artifacts/data_finetune_build.json"), "utf8"));
const wb = Workbook.create();
const main = wb.worksheets.add("data_finetune");
const rules = wb.worksheets.add("design_rules");
const dict = wb.worksheets.add("field_dictionary");
const summary = wb.worksheets.add("build_summary");

const records = build.records;
const columns = Object.keys(records[0]);
const matrix = [columns, ...records.map((record) => columns.map((column) => record[column] ?? null))];
const headerFmt = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
const bodyFmt = { font: { name: "Arial", size: 9 }, verticalAlignment: "center" };

for (const sheet of [main, rules, dict, summary]) sheet.showGridLines = false;
main.getRangeByIndexes(0, 0, matrix.length, columns.length).values = matrix;
main.getRangeByIndexes(0, 0, 1, columns.length).format = headerFmt;
main.getRangeByIndexes(1, 0, records.length, columns.length).format = bodyFmt;
main.freezePanes.freezeRows(1);
main.getRangeByIndexes(0, 0, matrix.length, columns.length).format.wrapText = false;
main.getRangeByIndexes(0, 0, matrix.length, columns.length).format.autofitColumns();
main.getRangeByIndexes(0, 0, 1, columns.length).format.rowHeight = 32;

const columnName = (number) => {
  let result = "";
  for (let value = number; value > 0; value = Math.floor((value - 1) / 26)) {
    result = String.fromCharCode(((value - 1) % 26) + 65) + result;
  }
  return result;
};
const ruleColumns = Object.keys(build.rules[0]);
const ruleMatrix = [ruleColumns, ...build.rules.map((rule) => ruleColumns.map((column) => rule[column] ?? null))];
rules.getRangeByIndexes(0, 0, ruleMatrix.length, ruleColumns.length).values = ruleMatrix;
rules.getRangeByIndexes(0, 0, 1, ruleColumns.length).format = headerFmt;
rules.getRangeByIndexes(1, 0, build.rules.length, ruleColumns.length).format = bodyFmt;
rules.freezePanes.freezeRows(1);
rules.getRangeByIndexes(0, 0, ruleMatrix.length, ruleColumns.length).format.autofitColumns();
rules.getRangeByIndexes(0, 0, 1, ruleColumns.length).format.rowHeight = 32;

const entries = [
  ["sample_id", "样本唯一标识", "由 sample 作为主键保留", "hole_stats_AB_hole.sample"],
  ["group_id", "分组编号", "沿用源表 id；用于按组划分数据", "hole_stats_AB_hole.id"],
  ["sample_order", "组内样本顺序", "沿用源表 id_num", "hole_stats_AB_hole.id_num"],
  ["dataset_domain", "数据域 A/B", "标准化数据集来源字段；仅保留 A", "hole_stats_AB_hole 数据集来源字段"],
  ["count_scope", "计数口径", "固定为 half_section，表示半截面", "规则约定"],
  ["mm_per_pixel", "像素比例", "固定为 5.0 mm/pixel", "规则约定"],
  ["full_*_count / half_*_count", "孔位数量（全/半截面）", "half=ceil(full/2)，保留中心线项", "hole_stats_AB_hole"],
  ["full_*_component_count / half_*_component_count", "颜色连通域数量", "半截面值按 ceil(full/2) 计算", "hole_stats_AB_hole"],
  ["full_*_region_count / half_*_region_count", "区域数量", "半截面值按 ceil(full/2) 计算", "dataset_report.xlsx"],
  ["full_*_entity_count", "实体/文件分组数量", "保留报告中的全截面数量", "dataset_report.xlsx"],
  ["geometry_* / clipped_*", "几何窗口字段", "沿用报告值，不换算为半截面坐标", "dataset_report.xlsx"],
  ["*_strand_specs", "钢束规格", "保留原规格并抽取规格数字", "hole_stats_AB_hole"],
  ["*_single_strand_count", "单一规格束数", "单一规格可查规则；混合规格不做单规格规则查找", "design_parameter.xlsx"],
  ["*_rule_lookup_status", "规则查找状态", "记录 none、single_spec 或 mixed_spec_requires_tendon_level", "规格解析规则"],
  ["*_duct_* / *_pass_* / *_anchor_* / *_jack_*", "设计规则距离（mm）", "按钢束规格匹配空间参数库", "design_parameter.xlsx"],
  ["*_onehot_specs", "规格 one-hot 摘要", "将激活位置转换为规格编号，便于核对", "hole_stats_AB_hole"],
  ["rule_mode", "规则模式", "按通过束规则记录", "规则约定"],
  ["overlap_flag / overlap_status", "重叠检查", "不读取 overlap 表；flag 留空，status 标记 not_checked，不参与筛选", "未使用 overlap_summary.xlsx"],
  ["label_quality_status / *_count_mismatch", "标签质检字段", "保留源质检报告，不用作筛选条件", "dataset_report.xlsx"],
];
dict.getRangeByIndexes(0, 0, entries.length + 1, 4).values = [["字段名", "中文含义", "修改/转换说明", "来源"], ...entries];
dict.getRange("A1:D1").format = headerFmt;
dict.getRangeByIndexes(1, 0, entries.length, 4).format = bodyFmt;
dict.freezePanes.freezeRows(1);
dict.getRangeByIndexes(0, 0, entries.length + 1, 4).format.wrapText = true;
dict.getRange("A:A").format.columnWidth = 38;
dict.getRange("B:B").format.columnWidth = 27;
dict.getRange("C:C").format.columnWidth = 58;
dict.getRange("D:D").format.columnWidth = 40;
dict.getRangeByIndexes(0, 0, entries.length + 1, 4).format.autofitRows();

const summaryRows = [
  ["项目", "值"],
  ["输出表", "data_finetune"],
  ["纳入条件", "dataset_domain == A 且保留对应 combine 文件"],
  ["重叠筛选", "不考虑；未读取 overlap_summary.xlsx"],
  ["A 域记录数", records.length],
  ["人工清理剔除数", build.metadata.excluded_after_manual_review_count],
  ["人工清理剔除样本", (build.metadata.excluded_after_manual_review ?? []).join(", ")],
  ["计数口径", "half_section（半截面）"],
  ["半截面换算", "ceil(full_count / 2)，保留中心线项"],
  ["mm_per_pixel", 5.0],
  ["规则模式", "pass_only；钢束规格映射到 design_parameter.xlsx"],
  ["报告未匹配记录", build.metadata.unmatched_dataset_report_rows],
  ["来源", build.metadata.sources.join("; ")],
];
summary.getRangeByIndexes(0, 0, summaryRows.length, 2).values = summaryRows;
summary.getRange("A1:B1").format = headerFmt;
summary.getRangeByIndexes(1, 0, summaryRows.length - 1, 2).format = bodyFmt;
summary.freezePanes.freezeRows(1);
summary.getRange("A:A").format.columnWidth = 28;
summary.getRange("B:B").format.columnWidth = 84;
summary.getRangeByIndexes(0, 0, summaryRows.length, 2).format.wrapText = true;
summary.getRangeByIndexes(0, 0, summaryRows.length, 2).format.autofitRows();

wb.recalculate();
const previewDir = path.join(root, "artifacts/data_finetune_preview");
await fs.mkdir(previewDir, { recursive: true });
const previewRanges = [
  ["data_finetune", "A1:O14", "data_finetune_top.png"],
  ["data_finetune", "P1:AD14", "data_finetune_features_1.png"],
  ["data_finetune", "AE1:AS14", "data_finetune_features_2.png"],
  ["data_finetune", "AT1:BH14", "data_finetune_features_3.png"],
  ["data_finetune", "BI1:CF14", "data_finetune_rules.png"],
  ["design_rules", "A1:N22", "design_rules.png"],
  ["field_dictionary", "A1:D19", "field_dictionary.png"],
  ["build_summary", "A1:B13", "build_summary.png"],
];
for (const [sheetName, range, filename] of previewRanges) {
  const preview = await wb.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, filename), new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(path.join(root, "data_finetune.xlsx"));
const check = await wb.inspect({
  kind: "table",
  sheetId: "data_finetune",
  range: `A1:${columnName(columns.length)}8`,
  include: "values",
  tableMaxRows: 8,
  tableMaxCols: columns.length,
  maxChars: 5000,
});
console.log(JSON.stringify({
  workbook: path.join(root, "data_finetune.xlsx"),
  rows: records.length,
  columns: columns.length,
  rules: build.rules.length,
  sheetNames: [main.name, rules.name, dict.name, summary.name],
  inspect: check.ndjson,
}));
