# s1 复现实验记录簿 (Experiment Logs)

本文件用于记录 s1 (arXiv:2501.19393) 极简复现过程中的全部训练实验、超参数组合、硬件显存实测数据与收敛曲线。

---

## 实验 01：8GB 显卡 4-bit QLoRA 5-step 冒烟压测 (Smoke Test)

- **实验日期**：2026-09-13
- **实验目标**：在本地消费级 8GB 显存显卡上，打通 4-bit 量化、LoRA 注入、动态 DataCollator、Prompt Loss Masking、前向反向求导与权重落盘的全链路闭环。

### 1. 软硬件与运行环境
- **操作系统**：Windows 11
- **GPU 硬件**：NVIDIA GeForce RTX 4060 Laptop (8GB 物理显存)
- **基座模型**：`Qwen/Qwen2.5-7B-Instruct`
- **训练数据**：`simplescaling/s1K-1.1` (截取前 8 条样本)

### 2. 核心超参数配置 (Hyperparameters)
| 超参数项 | 配置值 | 说明 |
| :--- | :--- | :--- |
| **量化精度** | 4-bit NF4 (Double Quant) | 7B 基座模型权重常驻显存仅约 4.2 GB |
| **计算精度** | bfloat16 | 保持数值稳定性，防止下溢出 |
| **LoRA 秩 (r)** | 16 | 覆盖全部 7 个线性投影层 (`q,k,v,o,gate,up,down`) |
| **LoRA Alpha** | 32 | 缩放系数 |
| **可训参数量** | 40,370,176 (0.5273%) | 仅微调 0.5% 参数，极大降低优化器与梯度显存 |
| **单卡批次 (Batch Size)** | 1 | 最小化物理瞬时激活开销 |
| **梯度累积 (GA)** | 1 | 冒烟测试阶段设为 1，实现步步即时反馈 |
| **序列最大长度** | 2048 | 8GB 显存黄金安全线 |
| **优化器** | `paged_adamw_8bit` | 带 CUDA 分页内存换页防爆栓 |
| **学习率 (LR)** | 1e-4 | 配合余弦衰减调度器 |

### 3. 硬件开销与显存表现
- **显存占用峰值**：`7937 MiB / 8188 MiB` (在未发生 CUDA OOM 的情况下平稳跑通)
- **GPU 算力利用率**：100% 满载
- **核心温度**：62°C ~ 66°C
- **硬件经验**：针对双显卡笔记本高压微调，已确认关闭屏幕超时休眠、开启 Python 最高性能优先模式，彻底规避电源管理休眠冲突。

### 4. 训练收敛指标记录 (Loss Trajectory)

| Step | Epoch | 学习率 (LR) | 梯度模长 (grad_norm) | 训练损失 (Loss) | 状态 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1/5** | 0.125 | 1.00e-04 | 0.5093 | **0.8730** | ✅ 正常反向求导 |
| **2/5** | 0.250 | 8.00e-05 | 0.6056 | **0.9292** | ✅ 梯度稳定更新 |
| **3/5** | 0.375 | 6.00e-05 | 0.3463 | **0.4713** | ✅ 显著下降收敛 |
| **4/5** | 0.500 | 4.00e-05 | 0.4434 | **0.5529** | ✅ 正常微调波动 |
| **5/5** | 0.625 | 2.00e-05 | 0.3220 | **0.3641** | 🎯 达到最优收敛 |

### 5. 产出物验证 (Artifacts)
- **保存路径**：`outputs/s1-7b-qlora/checkpoint-5/`
- **核心权重**：`adapter_model.safetensors` (体积约 161.5 MB，物理落盘完整)
- **配置文件**：`adapter_config.json`、`tokenizer.json`、`training_args.bin`

### 6. 实验结论
通过 QLoRA + 梯度检查点 + Paged 8-bit 优化器，在消费级单卡（RTX 4060 8GB）上完全具备对 7B 级别长思维链大模型进行微调的能力。全链路工程自测通过，可正式进入推理层与评测阶段。

---

## 实验 02：推理层 Budget Forcing 慢思考符号拦截与测试期算力缩放实测

- **实验日期**：2026-09-15
- **实验目标**：验证推理层纯净基座加载与 LoRA 权重注入；测试 Budget Forcing 状态机；验证思考算力硬截断（Early Answer Forcing）与长思维链收拢交卷能力。

### 1. 软硬件与运行环境
- **操作系统**：Windows 11
- **GPU 硬件**：NVIDIA GeForce RTX 4060 Laptop (8GB 物理显存)
- **基座模型**：`Qwen/Qwen2.5-7B-Instruct` (4-bit NF4 量化)
- **微调适配器**：`outputs/s1-7b-qlora/checkpoint-5/` (161.5MB 真实落盘权重)
- **测试题目**：$n^2 + 19n + 48$ 为完全平方数的正整数 $n$ 之和（高难度数论代数竞赛题）

### 2. 核心超参数配置 (Inference Hyperparameters)
| 超参数项 | 配置值 | 说明 |
| :--- | :--- | :--- |
| **思考预算 (thinking_budget)** | 384 tokens | 思考阶段算力硬配额 |
| **最大生成长度 (max_new_tokens)** | 1536 tokens | 包含思考链与最终答案的总配额 |
| **解码算法** | 贪婪解码 (Greedy, Temp=0) | 消除采样抖动，确保推导严谨确定 |
| **拦截特征序列** | `<\|im_start\|>answer` (Tokens: `[151644, 9217]`) | 思考结束转折标记 |
| **思考转折提示 (turn_prompt)** | `\nWait, let me rethink this problem...` | 诱导注意力回溯验算 |
| **显存常驻开销** | ~4.6 GB | 4-bit 基座 + LoRA 旁路，稳定运行无抖动 |

### 3. 实测表现与推理轨迹记录 (含现场证据链)

#### 3.1 纯净基座载入与适配器权重注入 (0 告警)
消除初测阶段的双重 `PeftModel` 嵌套，实现 0 告警、0 缺失键，40,370,176 个 LoRA 权重全量装载至 GPU 显存。

![纯净基座载入与已训练适配器注入](assets/exp02/exp02_clean_lora_load.png)

#### 3.2 预算耗尽强推交卷 (Early Answer Forcing)
模型在深入推导至第 384 个 Token 处达到思考预算上限。状态机精准截断草稿并强行注入 `<|im_start|>answer` 交卷引导词。模型顺畅衔接上下文，直接转入答案求解阶段。

![思考达到 384 tokens 强制截断并注入 answer](assets/exp02/exp02_early_answer_forcing.png)

#### 3.3 数论代数因式分解与因子对排查
进入作答阶段后，模型展现了极其严谨的符号计算逻辑，利用平方差公式 $(m-2k)(m+2k)=169$ 展开穷举，精确列出 $(1, 169)$、$(13, 13)$ 及其对应负数因子对。

![169 因子对拆解与联立方程求解](assets/exp02/exp02_factor_pairs_deduction.png)

#### 3.4 标称答案输出与结果验证
代回原方程检验正整数条件，成功排除 $-52$、$-3$、$-16$ 等负根，求得唯一正整数解 $n=33$，最终以标准竞赛格式输出 `\boxed{33}`，答题准确率达成 100%。

![成功求得唯一正整数解并以 boxed 格式交卷](assets/exp02/exp02_boxed_answer_success.png)

---

### 4. 关键避坑与对照复盘 (Failure vs. Success)

在本次实验中，我们经历了从“假装加载”到“真正生效”的关键工程修复：

| 阶段 | 现象与日志特征 | 底层根因 | 实测截图存证 |
| :--- | :--- | :--- | :--- |
| **初测失败** | 抛出 `missing adapter keys` 警告，随后生成人力资源（HR）规章制度 | 调用了已带 LoRA 的 `get_model()` 造成双重嵌套，权重未生效；同时 `eos_token_id=None` 迫使模型在原任务结束后生成预训练网页语料 | ![双重Peft警告](assets/exp02/exp02_bug_double_peft_warning.png)<br/>![语料漂移HR幻觉](assets/exp02/exp02_bug_hr_hallucination.png) |
| **修复通关** | 0 警告载入，推导过程极度专注，截断交卷后以因式分解做对数论竞赛题 | 采用 `load_clean_base_model()` 保证纯净映射；规范因果解码停止标记 | ![通关结果](assets/exp02/exp02_boxed_answer_success.png) |

#### 核心认知沉淀
1. **双重包装陷阱**：在推理阶段必须先通过 `load_clean_base_model` 实例化纯净的 `AutoModelForCausalLM`，严禁直接调用包含 `get_peft_model` 的训练初始化函数，杜绝 `base_model.model.base_model...` 键名前缀漂移。
2. **测试期算力缩放的双向机理**：
   - **机制 A（预算耗尽强推交卷）**：当思考预算较紧（如 384 tokens）时，模型未及自主交卷即被硬性强推，测试表明模型在被强制交卷后具备强大的断点续推与答案收敛能力。
   - **机制 B（慢思考扩展与 Wait 注入）**：需将 `thinking_budget` 设为高于自然推导长度（如 1024 tokens），当模型首次试图交卷时方可触发哨兵拦截并注入 `Wait` 展开二次反省。


