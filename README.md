# TechLens-1.5B：金融分析模型

> 针对 StockMind 多 Agent 系统中 technical_analyst 模块的云端 API 替换方案。  
> 完整后训练工程：数据合成 → SFT → DPO → 量化 → FastAPI 服务 → Agent 集成。

---

## 一、项目背景与痛点

StockMind 是一个基于 LangGraph 的 A 股多 Agent 投研系统，每次分析并行调用 DeepSeek 三次。其中 `technical_analyst` 模块负责读取 K 线/行情/KDJ 数据并输出技术面分析报告，是三路分析师中**输入最结构化、输出最格式化**的一路。

**三个核心痛点：**

| 痛点 | 具体表现 |
|---|---|
| 延迟高 | 云端 API P50 延迟 8.5s，拖慢整体分析流程 |
| 成本高 | 三路分析师并行，每次分析消耗 3 次 API 调用 |
| 幻觉 | 模型编造支撑位/压力位数字、KDJ 数值抄错 |

**解决思路：** 该任务不需要通用大模型的开放推理能力，只需要"读数、判断、按格式输出"。用 SFT+DPO 微调 1.7B 小模型专门接管这个任务，本地 GPU 推理，FastAPI 服务化，无缝替换云端 API 调用。

---

## 二、项目亮点（与同类微调项目的差异）

### 亮点一：训练数据完全程序化合成，不依赖强模型蒸馏

大多数微调项目的数据来源是"用 GPT/DeepSeek 生成 + 人工抽检"（知识蒸馏路线），存在：
- 教师模型幻觉污染训练数据的风险
- API 调用成本
- 标签质量依赖教师模型能力上限

TechLens 的数据完全程序化生成：
- K 线数据：随机游走模拟，控制七种行情类型
- 指标计算：KDJ/MA20 公式与线上 `_calc_kdj_signal` **逐行对齐**
- 标签推导：规则函数 `derive_label()` 是单一事实来源，100% 确定性
- 零 API 成本，无教师幻觉，可无限扩量

### 亮点二：幻觉被量化为两个客观指标

金融场景的幻觉很难量化，TechLens 把它拆成两个 100% 规则可判的指标：

- **KDJ 复制保真度**：输出的 K/D/J 三值必须与输入数值逐字相等，抄错即幻觉
- **价位纪律**：该输出"暂不设定"时输出具体数字即编造，可精确统计

### 亮点三：评估体系 100% 规则判分，零 LLM Judge

所有评估角度均由代码断言完成，不依赖任何大模型打分，结果客观可复现。

### 亮点四：危险行为单独计量

工具报错/数据不足时硬着头皮输出分析 = **危险分析率**，金融场景最不可接受的失败模式，DPO 火力重点针对这类行为。

---

## 三、技术架构

```
用户/StockMind Agent
        ↓ POST /analyze
TechLens FastAPI Service
        ├── 输入：history_result + price_result + kdj_result + stock_code
        ├── 推理：Qwen3-1.7B + DPO LoRA Adapter
        └── 输出：结构化 JSON（status=OK/ABORT）
                ↓ 失败时自动降级
        DeepSeek 云端 API（兜底）
```

**输出格式（status=OK）：**
```json
{
  "status": "OK",
  "stock_code": "600036",
  "trend": "bullish",
  "volume_price": "缩量上涨",
  "support": "暂不设定",
  "resistance": 12.80,
  "kdj": {"K": 20.5, "D": 25.3, "J": 10.8, "signal": "buy_ready"},
  "confidence": "high",
  "summary": "趋势向好，缩量上涨，KDJ满足超卖买入条件。"
}
```

**输出格式（status=ABORT）：**
```json
{
  "status": "ABORT",
  "reason": "历史K线工具返回错误，无法完成技术面分析"
}
```

---

## 四、数据工程

### 4.1 七类任务设计

| 任务类型 | 数量 | 说明 | 难点 |
|---|---|---|---|
| ok_bullish | 400 | 上升趋势行情 | 量价关系判断 |
| ok_bearish | 400 | 下降趋势行情 | 同上 |
| ok_neutral | 350 | 震荡行情 | trend=neutral 的判断 |
| ok_no_levels | 350 | 价位无法判断 | **核心难例**：必须输出"暂不设定" |
| ok_edge | 200 | 含涨跌停/零成交异常日 | confidence 应为 low |
| abort_tool_error | 180 | 工具返回 [TOOL_ERROR] | **必须 ABORT，不得硬分析** |
| abort_insufficient | 120 | K 线不足 60 根 | 同上 |
| **合计** | **2000** | | |

**配比设计逻辑：** ok_no_levels（350条）和 abort 两类（300条）刻意加重，因为这两类是基座模型最容易出错的场景（编造价位、该停却分析），也是 DPO 的火力重点。

### 4.2 冻结评估集

- 每类抽取 30 条，共 **210 条**
- `gen_data.py` 运行时自动切割，物理写入 `data/eval/test.jsonl`
- 脚本有冻结保护：文件存在时拒绝覆盖，防止无意间重建污染基准

### 4.3 SFT/DPO 数据

- **SFT**：1790 条，alpaca 格式，`data/train/sft_train.json`
- **DPO**：400 对 chosen/rejected，`data/train/dpo_train.json`

**DPO 偏好对构造（程序化定向扰动）：**

| 扰动类型 | 占比 | 针对痛点 |
|---|---|---|
| 编造支撑位/压力位 | 40% | 价位幻觉（最核心） |
| KDJ 数值漂移 | 25% | 复制保真度 |
| 该 ABORT 却硬分析 | 25% | 危险行为 |
| markdown 格式包裹 | 10% | 格式遵循 |

---

## 五、训练流程

### 5.1 SFT（监督微调）

```yaml
模型: Qwen3-1.7B
方法: LoRA（r=16, alpha=32, lora_target=all）
损失: Response-only loss（train_on_prompt=false）
Epochs: 3
学习率: 1e-4，cosine decay
硬件: RTX 5070 Ti 16GB
耗时: 约 1.5h
```

**关键设计 - Response-only loss：**  
只对模型输出的 JSON 部分计算损失，屏蔽 prompt（system + 三段工具结果）的梯度。避免模型学习"复述输入"，专注学习"如何从输入推导正确的 JSON 输出"。

**训练结果：**
```
train_loss: 1.034 → 0.059（收敛）
eval_loss:  0.012（无过拟合）
```

### 5.2 DPO（偏好对齐）

```yaml
基础: 在 SFT checkpoint 上继续
方法: DPO sigmoid loss
β: 0.1（约束 KL 散度，防过度优化）
Epochs: 1
学习率: 5e-6（比 SFT 低一个数量级）
数据: 400 对偏好数据
耗时: 约 45min
```

**β=0.1 的意义：**  
β 控制微调后策略与参考模型（SFT checkpoint）的 KL 散度距离。β 太小会过度优化，在拉大 chosen/rejected 分数差时损伤已收敛的格式能力；β=0.1 在"纠正幻觉"和"保持格式"之间取得平衡。

**注意事项：** DPO 训练时评估阶段需要同时加载 policy + reference 两个模型，12GB 显存不够。解决方案：去掉 `eval_strategy`，训完后单独用推理脚本评估。

---

## 六、评估体系

### 6.1 六维分数卡

| 评估角度 | 判分方式 | 含义 |
|---|---|---|
| 格式遵循 | JSON 可解析 + schema 合规 + 无 markdown 围栏 | 模型是否输出合法 JSON |
| OK/ABORT 决策 | status 字段与期望对比 | 该分析时分析，该停时停 |
| KDJ 复制保真度 | K/D/J 三值与输入逐字比对 | 幻觉直接度量 |
| 分类字段准确率 | trend/volume_price/signal/confidence 四分类 | 判断质量 |
| 价位纪律 | 该"暂不设定"时是否输出数字 | 编造价位检测 |
| 克制与安全 | 该 ABORT 却输出 OK 的比率 | 危险行为率 |

**全部使用代码断言，零 LLM Judge。**  
原因：输出是纯结构化 JSON，每个字段的期望值都能写进 `expected`，规则可 100% 覆盖。

### 6.2 四阶段对比结果

| 指标 | M0 基座 | M1 SFT | M2 DPO | M3 INT8量化 |
|---|---|---|---|---|
| 格式遵循 | 0.0% | 100.0% | 100.0% | 100.0% |
| OK/ABORT 决策 | 0.0% | 100.0% | 100.0% | 100.0% |
| KDJ 复制保真度 | 0.0% | 100.0% | 100.0% | 100.0% |
| 分类字段全对率 | 0.0% | 50.0% | 50.0% | 53.3% |
| 价位纪律 | 0.0% | 83.3% | 83.3% | 76.7% |
| 编造价位率 | - | 10.0% | **0.0%** | 6.7% |
| 危险分析率 | - | 0.0% | 0.0% | 0.0% |
| 推理延迟 P50 | 8.54s | 7.25s | 6.62s | 16.6s |

**关键结论：**
- SFT 让基座从"完全不懂格式"跳到"核心指标全 100%"，一次收敛
- DPO 的核心贡献：编造价位率 10% → 0%，其他指标不退化
- INT8 量化在 GPU 上反而比 bf16 慢（16.6s vs 6.6s），符合"INT8 收益在 CPU 侧"预期
- Qwen3 输出含 `<think></think>` 思考标签，需在 `parse_model_output` 中剥离

---

## 七、部署与集成

### 7.1 FastAPI 推理服务

```bash
python api_server.py
# 服务启动在 http://0.0.0.0:8088
```

接口：
- `GET /health` → 健康检查
- `POST /analyze` → 技术面分析

### 7.2 接入 StockMind

改动两处，下游 Agent 节点零改动：

**llm_config.py** 新增 `TechLensClient`，封装本地服务调用与健康检查。

**technical_analyst.py** 改造为双 backend 架构：
```
TechLens 服务在线 → 调用本地模型（低延迟、低成本）
TechLens 服务离线 → 自动降级 DeepSeek 云端 API（兜底）
```

---

## 八、项目结构

```
TechLens-1.5B/
├── data/
│   ├── analysis_schema.json      # 输出协议与标注规则（单一事实来源）
│   ├── eval/test.jsonl           # 冻结评估集（210条，不参与训练）
│   └── train/
│       ├── clean.jsonl           # 训练原料（1790条）
│       ├── sft_train.json        # SFT alpaca格式
│       ├── dpo_train.json        # DPO偏好对
│       └── dataset_info.json     # LLaMA-Factory注册文件
├── src/techlens/
│   ├── schemas/output.py         # 四种action结构校验（全项目复用）
│   ├── prompts/template.py       # 训练/推理共用prompt（分布一致纪律）
│   ├── datagen/
│   │   ├── engine.py             # K线模拟+指标计算+规则标注
│   │   └── builder.py            # SFT/DPO训练集构造
│   ├── inference/backends.py     # 推理后端抽象（mock/llama_server）
│   └── evaluation/
│       ├── scorers/all_scorers.py # 六维规则scorer
│       └── runner.py              # 评估runner+分数卡
├── scripts/
│   ├── gen_data.py               # 一键生成全部数据（秒级，零API成本）
│   └── run_eval.py               # 评估入口
├── configs/training/llamafactory/ # SFT/DPO/export训练配置
├── models/
│   ├── adapters/                  # LoRA adapter权重
│   └── merged/                    # 合并后全量权重
├── experiments/                   # 各阶段评估结果与分数卡
├── project-log/                   # 各阶段训练日志与踩坑记录
└── api_server.py                  # FastAPI推理服务入口
```

---

## 九、复现步骤

```bash
# 1. 安装依赖
pip install pandas numpy llamafactory transformers peft fastapi uvicorn bitsandbytes

# 2. 生成数据（秒级完成，无需API key）
python scripts/gen_data.py

# 3. 验证评估链路
python scripts/run_eval.py --backend mock-gold   # 应全100%
python scripts/run_eval.py --backend mock-noisy  # 验证scorer灵敏度

# 4. 下载基座模型
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', cache_dir='./models/base')"

# 5. M0 baseline（记录训练前基准）
python run_baseline.py

# 6. SFT训练
llamafactory-cli train configs/training/llamafactory/sft_qwen3_1p7b.yaml

# 7. SFT评估
python run_sft_eval.py

# 8. DPO训练（去掉eval_strategy避免OOM）
llamafactory-cli train configs/training/llamafactory/dpo_qwen3_1p7b.yaml

# 9. DPO评估
python run_dpo_eval.py

# 10. 合并LoRA权重
llamafactory-cli export configs/training/llamafactory/export_merge.yaml

# 11. 启动推理服务
python api_server.py

# 12. 测试接口
curl http://localhost:8088/health
```

---

## 十、面试要点速记

**Q：为什么不用强模型蒸馏生成训练数据？**  
A：金融K线数据可以程序化精确模拟，KDJ/MA20公式是确定性计算，Ground Truth由规则推导，不存在"教师模型也会出错"的问题。蒸馏的价值在于迁移难以规则化的知识（如自然语言理解），本任务不需要。

**Q：DPO 为什么选 β=0.1？**  
A：β 控制策略与参考模型的 KL 散度。β 太小过度优化会损伤 SFT 已收敛的格式能力（格式遵循从100%退化）；β 太大纠偏力度不够。0.1 是经验值，实验验证 DPO 后格式指标未退化。

**Q：INT8 量化为什么在 GPU 上反而更慢？**  
A：INT8 的收益来自减少内存带宽压力，在 CPU 上效果显著。GPU 本身算力足够，INT8 的矩阵运算还需要额外的量化/反量化开销，反而增加延迟。结论：量化部署目标是 CPU 侧，GPU 推理保持 bf16。

**Q：评估集为什么要物理冻结？**  
A：防止"刷分"——如果评估集参与训练，模型相当于在"考已经背过的题"，指标虚高失去意义。冻结后评估集成为不变的标准尺子，每个训练阶段都用同一把尺子量，对比才有意义。

**Q：Qwen3 的 think 标签是什么，怎么处理？**  
A：Qwen3 默认开启"思考模式"，推理时会在正式输出前生成 `<think>...</think>` 推理过程。本任务只需要最终 JSON，在 `parse_model_output` 中用正则剥离 think 块即可。如果不处理，JSON 解析会失败导致所有指标归零。

**Q：DPO 训练 OOM 怎么解决？**  
A：DPO 评估阶段需要同时加载 policy 模型和 reference 模型（SFT checkpoint），两个 1.7B 模型加起来超出 12GB 显存。解决方案是去掉训练配置中的 `eval_strategy`，训练结束后单独跑推理评估脚本，两者不同时在显存里。

---

## 十一、踩坑记录

| 问题 | 原因 | 解决方案 |
|---|---|---|
| SFT后评估全0% | Qwen3输出含`<think></think>`标签导致JSON解析失败 | `parse_model_output`用正则剥离think块 |
| DPO训练OOM | 评估阶段同时加载两个模型超出显存 | 去掉`eval_strategy`，训完单独评估 |
| YAML训练配置乱码 | Windows PowerShell编码问题 | 用Python直接写文件而非PowerShell重定向 |
| llama-cpp-python安装失败 | Windows路径过长限制 | 改用预编译wheel或bitsandbytes量化 |
| Ollama不支持Qwen3 | Ollama 0.31.2尚未支持Qwen3架构 | 改用bitsandbytes INT8量化方案 |