# Phase 03-04 SFT + DPO

## SFT结果 (M1)
- train_loss: 0.0588, eval_loss: 0.0124
- 格式遵循: 100%, OK/ABORT: 100%, 复制保真度: 100%
- 分类字段: 50%, 价位纪律: 83.3%, 编造率: 10%

## DPO结果 (M2)  
- train_loss: 0.6657, 1epoch, 25steps
- 编造价位率: 10% -> 0% (DPO核心贡献)
- 其他指标与SFT持平

## 关键发现
- Qwen3思考模式输出think标签需要剥离
- DPO eval阶段OOM(12GB显存不够双模型)，关闭eval_strategy解决
