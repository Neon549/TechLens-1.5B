# 冻结评估集

`test.jsonl` 是现有的程序化合成评估集（7 类、每类 30 条），保留它以便与既有 M0–M3 结果可比。

`real_test_v2.jsonl` 由 `python scripts/freeze_real_eval.py` 生成。脚本从公开的 Yahoo Finance A 股日线下载数据，按真实 OHLCV 规则计算标签，并在 `bullish`、`bearish`、`neutral`、`no_levels`、`edge` 五类中等量、按股票 round-robin 抽样。它拒绝覆盖已冻结文件，也会在任何类别不足时失败，而不会悄悄生成失衡数据。`real_test.jsonl` 是首版冻结集，仅保留作审计，不再作为默认基准。

真实评估数据严禁进入 `data/train/` 或定向 DPO 的输入。下载时须在实验日志中记录命令、日期范围、股票池、样本数量与数据源可用性。
