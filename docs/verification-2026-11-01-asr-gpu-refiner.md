# ASR GPU 迁移与离线精修验证

2026-11-01，按“更精准转写且不明显牺牲速度”的目标完成两项变更并验证：转写推理迁移到 CUDA，段末精修从流式重解码改为离线 SeACo-Paraformer 加标点模型。本页记录实测数据；转写质量提升仍需课堂录音标注对比后才能声称（见文末未验证项）。

## 变更内容

- `.venv` 内 torch 从 `2.14.0+cpu` 换为 `torch 2.14.1+cu130` 与 `torchaudio 2.11.0+cu130`（RTX 50 系 Blackwell 需要 cu128 及以上的 CUDA 构建；cu128 渠道最高只有 torch 2.11，故选 cu130）。`pyproject.toml` 的 asr 依赖下限提升到 `torch>=2.7`。
- 新增 `src/classbot/asr/offline_refiner.py`：`paraformer-zh` 别名即离线 SeACo-Paraformer（热词内建），组合 `ct-punc` 标点模型；缓存校验复用流式模型的 `_cached_model_path` 规则，损坏缓存仍按“加载失败、不换权重”处理。
- `load()` 内置一次静音预热推理：实测首次 CUDA 前向需约 7.5 秒 kernel 编译，预热把它吸收进后台加载，避免首个语音段精修阻塞 worker 触发队列过载。
- `FunAsrStreamingRecognizer` 接受可选 `refiner`：会话开始后台预加载；精修时已加载则走离线解码，未加载或推理失败则回退原有流式整段重解码，行为与现有 `final_refinement` 开关兼容。
- 装配在 `main.py._make_recognizer`：`CLASSBOT_ASR_REFINER=seaco`（默认）启用离线精修，`stream` 保留旧行为，`off` 关闭；精修输出经过课程纠错表，且因标点模型而带完整标点。
- 用户 `.env` 已切换 `CLASSBOT_ASR_DEVICE=cuda`；`.env.example` 记录 cu130 安装命令。
- 冒烟脚本：`scripts/verify_asr_gpu.py`（流式加载、分块推理、离线加载加预热、整段精修、显存峰值）。

## 实测（RTX 5070 Laptop GPU，torch 2.14.1+cu130，模型本地缓存命中）

| 项目 | 实测值 |
|---|---|
| CUDA 可用性 | capability (12, 0)，真实 matmul kernel 通过 |
| 流式模型加载 | 19.7 秒（发生在开麦之前，不占监听时间） |
| 流式首个分块推理 | 1451 毫秒（一次性 CUDA 预热；后续分块即时） |
| 离线精修加载 + 预热 | 48.8 秒（后台预加载，跨会话复用不重复加载） |
| 3 秒音频离线精修（预热后） | 268 毫秒（RTF ≈ 0.09；预热前为 7607 毫秒） |
| 峰值显存 | 2966 MiB（8 GB 显存内） |
| 精修输出标点 | 有（“嗯，有有有有有，有没有没有证据？”） |

首次全量下载（SeACo 约 1.13 GB + ct-punc 约 0.46 GB）约 202 秒，仅发生一次。

## 测试

- `tests/test_offline_refiner.py`：缓存校验、别名首下载、损坏权重不换、热词参数、无流式参数、预热成功/失败（失败保持未加载以便回退）、加载幂等。
- `tests/test_funasr_streaming.py` 新增：已加载走离线精修（不触发流式整段重解码、纠错表仍生效）、未加载回退流式、离线失败回退流式、会话启动后台预热、`final_refinement=false` 不预热、stop 取消后台加载。
- 全量：`python -m pytest -q` 共 **1147 passed, 4 skipped**；`ruff check src tests scripts` 通过；`git diff --check` 无空白错误。

## 未验证项（下一步）

- 转写质量提升幅度：需要一段约 10 分钟带人工标注的课堂录音，按“流式基线 / +流式精修 / +SeACo 离线精修”三组算 CER 与术语命中率；合成音频只证明链路可用，不证明准确率。
- `corrections.txt` 裁剪：SeACo 热词生效后部分纠错条目可能冗余，需实测后逐条移除，避免替换互相干扰。

## 2026-11-02 追加：跨会话热缓存与 lookback 实验

- 流式与精修模型增加进程内按设备热缓存（`build_model` / `build_refiner_model`），模型对象无会话状态，可跨监听复用：第二次“开始监听”不再重付约 20 秒流式加载与后台精修加载；流式构建后吸收一次 600ms 静音预热（约 1.5 秒 CUDA kernel 编译）。测试以 autouse fixture 清缓存隔离。
- 真实录音质量评测（三变体全量、1238 段）见 `docs/eval-2026-11-02-asr-quality.md`：离线精修 RTF 0.036（旧精修 0.191），热词密度 +16%，空段 -35%；无人工参考，不声称准确率。
- `encoder_chunk_look_back` 4→12 实验（600 秒流式输出对离线参照）：相似度 0.653→0.639、热词 66→57、RTF 0.080→0.136，**全面更差**，参数维持 4。评测脚本新增 `--lookback` 与 `--variants` 便于复测。
