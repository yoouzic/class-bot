# Scripts

所有命令从仓库根目录执行，并设置 `PYTHONPATH=src`。

## ASR

- `asr/eval_asr_variants.py`：同一录音的流式、整段重解码、离线精修三变体对比。
- `asr/analyze_asr_eval.py`：汇总变体一致率、错误词形和样本。
- `asr/verify_asr_gpu.py`：验证流式模型与离线精修的设备、权重和推理路径。

## 答辩与音频

- `defense/verify_live_defense.py`：Windows 回放 → 生产答辩运行时 → 真实 ASR 的端到端验证。
- `defense/verify_audio_ingress.py`：观察回放、音频循环、ASR 和问题检测链路。

## 索引

- `index/build_knowledge_index.py`：在独立进程中构建场景本地索引。

## 评测

- `eval/eval_question_detector.py`：课堂/答辩问题识别评测。
