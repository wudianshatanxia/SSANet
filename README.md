# AI 驱动的多 TRP 无线资源调度

面向 120 个 UE、24 个 TRP、每个 TRP 8 根天线的预测感知调度实验。模型融合当前 PMI、预测的 5 ms 后 PMI/速率、历史调度速率和 UE 速度，输出每个 UE 的调度分数。

## 目录与入口

```text
config.py                 统一实验参数、特征布局和默认路径
run_demo.py               数据准备 → 训练 → 评价 → Fig.3–Fig.5
requirements.txt          运行依赖
requirements-dev.txt      运行依赖 + pytest
src/
  models/                 LSTM、SSA-Net
  official_data.py        官方 MAT 检查与转换
  official_like_generator.py  官方格式仿真数据生成
  data_generator.py       合成数据生成
  dataset.py              数据集加载
  mobility.py             移动性与预测模型
  baselines.py            规则调度器和集合解码器
  train.py                训练与验证
  evaluate.py             方法对比与指标导出
  metrics.py              Q、相关性、集合 SINR 和训练损失
  plot_results.py         论文图与持续用户 episode 评价
  emergency_experiments.py  应急优先级、TRP 失效和时延实验
  official_score.py       官方示例评分逻辑的可复用实现
  checkpointing.py        checkpoint 加载与配置/数据兼容性检查
  reproducibility.py      随机种子与确定性设置
  stress_experiments.py   预测误差压力实验与推理计时
scripts/                  按阶段运行的命令行入口
tests/                    原有核心逻辑、论文评价与应急测试
docs/                     原始交接资料与本次整理记录
data/                     原始 MAT、生成 MAT 和转换缓存
outputs/                  checkpoint、CSV、图片和结果打包文件
```

当前实现以本 README 为使用说明。原始交接资料保存在 `docs/handoff_doc.docx`，其中的历史实验描述不代表当前实现。清理清单见 [整理记录](docs/cleanup_notes.md)。

## 环境与数据

使用 Python 3.10 或更高版本，在项目根目录运行：

```bash
python -m pip install -r requirements.txt
```

官方参考文件路径：

```text
data/official_raw/Training_Data.mat
data/official_raw/Test_Data.mat
```

MATLAB v7.3 字段：

| 字段 | 形状 |
| --- | --- |
| `N` | `(1, 1)` |
| `PMI_0ms`、`PMI_5ms` | `(2, 192, 120, N)` |
| `eigenvalue_0ms`、`eigenvalue_5ms`、`rate_old` | `(120, N)` |

特征值通过 `log2(1 + scale * eigenvalue)` 转为速率。默认模型输入使用当前信息构造的预测；真实 5 ms 后状态只用于教师标签、辅助监督和评价。`--use_future_input` 是显式的 oracle 诊断模式。

## 运行实验

小规模合成数据流程检查：

```bash
python run_demo.py --data_mode synthetic --num_samples 300 --epochs 1 --batch_size 16 --models lstm,ssa --output_subdir quick_check --overwrite
```

正式实验可先在新目录生成官方格式仿真数据，再转换、训练和绘图：

```bash
python scripts/08_generate_official_like_mat.py --train_N 16000 --test_N 2000 --out_dir data/generated_official_like_rebuilt
python run_demo.py --train_mat data/generated_official_like_rebuilt/Training_Data.mat --test_mat data/generated_official_like_rebuilt/Test_Data.mat --official_out data/official_converted_rebuilt --epochs 50 --batch_size 16 --models lstm,ssa --output_subdir rebuilt --overwrite
```

`--official_out` 指定转换缓存，`--output_subdir` 将 checkpoint、结果和图片放到 `outputs/<run>/`。`--overwrite` 会覆盖所指定的数据缓存；一键生成参数 `--generate_official_like` 使用固定的 `data/generated_official_like/` 目录。需要保留已有 MAT 时，使用上面的独立生成命令并指定新目录。

生成器、标签或模型结构发生变化后，应使用新转换目录并重新训练，避免混用不同版本的缓存和 checkpoint。早期 official-like 生成器曾按 chunk 改变隐含映射；仅重新转换旧 MAT 无法修复这类旧数据，正式结果应使用当前生成器重新生成。

已有模型时，只重新绘图：

```bash
python scripts/04_plot_figures.py --data_dir data/official_converted/test --ckpt_dir outputs/checkpoints --fig_dir outputs/figures --models lstm,ssa
```

## 当前 Fig.3–Fig.5

论文图的吞吐量使用固定总功率、按流分配功率、选中集合内空间泄漏得到的未来总速率。复 PMI 相关性为 `|h_i^H h_j| / (||h_i|| ||h_j||)`，SINR 泄漏使用其平方。

| 图 | 实验含义 | 图片 / 数值文件 |
| --- | --- | --- |
| Fig.3 | 调度规模 `S=4,8,12,16,20` 与未来调度总速率，单图 | `fig3_scheduled_ue_performance.png` / `fig3_scheduled_ue_performance_values.csv` |
| Fig.4 | 公平指数 alpha 与吞吐量、Jain 公平性，双纵轴 | `fig4_alpha_throughput_fairness.png` / `fig4_alpha_throughput_fairness_values.csv` |
| Fig.5 | UE 移动速度与未来调度总速率，单图 | `fig5_robustness.png` / `fig5_mobility_values.csv` |

图片和 CSV 默认保存在 `outputs/figures/`，指定运行名称时保存在 `outputs/<run>/figures/`。

- 主图比较 No Prediction、PF Prediction、Enhanced Greedy、LSTM 和 SSA-Net；Fig.3 另含使用真实未来 CSI 的 Greedy Oracle。
- Fig.3 的辅助 Jain 统计仍保存在 CSV。公平性在固定 UE 身份的持续 episode 内计算；不会把独立样本中的同一列当成长期用户。
- Fig.4 固定 eta，在相同持续 episode 上用 Q-guided Enhanced Greedy 扫描 alpha。每个 alpha 重新调度并更新历史速率；该图不扫描训练好的 SSA-Net。
- Fig.5 的速度变化保持边际空间相关分布一致。置信区间、选集相关性等辅助统计保存在 CSV，主图不绘制阴影置信带。

独立的预测 NMSE 诊断保留在 `src.plot_results.evaluate_prediction_error_impact`；它不属于默认三张主图。常规评价输出 `results/metrics_test.csv` 和逐样本 CSV，其字段与主图 episode CSV 应分别解读。
## SSA-Net 与训练

Q prior 融合当前速率、预测速率和历史速率的公平性权重。SSA-Net 在此基础上使用预测速率校准头、自适应 eta gate、跨 UE 注意力和从零初始化的排序残差。相关性作为学习特征，推理阶段的集合解码器只读取预测 CSI。

训练包含 BCE、ListNet、Top-K pairwise margin、未来速率 Smooth-L1 辅助监督，以及多个调度规模下的可微集合效用损失。checkpoint 选择结合物理未来吞吐量和 Q。

在完整运行命令后追加以下参数可启用扩展：

- `--run_stress`：运行额外压力实验。
- `--run_emergency_workshop`：运行应急业务实验；`--emergency_quick` 缩短应急实验，只用于检查流程。

多随机种子实验与均值、标准差、95% CI 汇总：

```bash
python scripts/09_run_multiseed.py --train_mat data/generated_official_like_rebuilt/Training_Data.mat --test_mat data/generated_official_like_rebuilt/Test_Data.mat --seeds 42,43,44 --epochs 50 --models lstm,ssa
```

每个 seed 使用独立的转换目录和输出目录；汇总读取当前的 `fig5_mobility_values.csv`。论文结果建议至少报告 3 个随机种子的统计量。

## 应急通信扩展

应急优先级通过 `Q_emg = omega * Q` 进入 SSA、PF、No Prediction 和 Enhanced Greedy 的调度效用。默认权重为 mission-critical 1.5、rescue coordination 1.2、regular 1.0；SSA 在 log-Q prior 中加入 log-omega，不增加网络层或 checkpoint 参数。

TRP outage 在同一个 episode 内保持失效 TRP 不变：将相应当前、预测和真实未来 PMI block 置零，并按保留信道能量衰减 SNR、重新计算速率。CSV 包含未来总速率、各类业务 QoS 满足率、关键用户调度比例、加权服务量与选集相关性。

时延实验使用 batch size 1，统计包含特征构造、前向传播和集合解码的在线端到端调度时延，输出 mean / median / P95。CPU 为共同基准；有 CUDA 时额外报告神经模型的 GPU 时延。quick 模式计时只用于流程检查。

已有模型时单独运行：

```bash
python scripts/10_run_emergency_workshop.py --data_dir data/official_converted/test --ckpt_dir outputs/checkpoints --out_dir outputs/emergency_workshop --models lstm,ssa
```

输出 `emergency_priority_sweep.csv`、`emergency_trp_outage.csv`、`emergency_latency.csv` 及对应图片。应急实验作用于调度和物理评价，不改变三张主图的定义。

## 分阶段脚本

| 入口 | 用途 |
| --- | --- |
| `scripts/00_inspect_official_mat.py` | 查看 MAT 字段和形状 |
| `scripts/01_convert_official_mat.py` | 转换官方格式 MAT 为统一数据布局 |
| `scripts/01_generate_data.py` | 直接生成合成数据；与 MAT 转换是两种数据准备方式 |
| `scripts/02_train_all.py` | 训练指定模型 |
| `scripts/03_evaluate_all.py` | 评价指定模型和规则基线 |
| `scripts/04_plot_figures.py` | 生成当前三张主图和数值 CSV |
| `scripts/05_predict_submission.py` | 导出调度分数、用户索引和 mask |
| `scripts/06_score_official.py` | 按官方示例评分逻辑评分 |
| `scripts/07_run_stress.py` | 运行预测误差压力实验与推理计时 |
| `scripts/08_generate_official_like_mat.py` | 生成扩展官方格式 MAT |
| `scripts/09_run_multiseed.py` | 多种子运行及统计汇总 |
| `scripts/10_run_emergency_workshop.py` | 单独运行应急通信实验 |

各脚本的参数以 `python <脚本路径> --help` 为准。默认目录见 `config.py`；单独训练/评价脚本默认使用合成数据目录，官方数据运行时应显式指定 `--data_dir`。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

从项目根目录运行。测试使用小规模数组，不需要重新生成正式数据或训练正式模型。
