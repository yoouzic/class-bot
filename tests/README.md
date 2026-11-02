# Tests

- `asr/`：流式识别、离线精修、实时标点
- `audio/`：采集、回放、VAD 分段
- `detection/`：课堂/答辩问题检测、口误归一化
- `knowledge/`：索引、缓存、向量、检索、教材覆盖
- `llm/`：DeepSeek 客户端、Search/Fetch、证据核验、网络补充
- `defense/`：答辩管线、核验契约、实时答辩验证、SSE 所有权
- `pipeline/`：实时管线与流所有权
- `sources/`：来源引用与标签审计
- `ui/`：主窗口、卡片、来源渲染、对话框
- `workspace/`：场景、工作区、缓存管理
- `core/`：配置、课程、领域模型、启动入口、运行时

从仓库根目录运行：

```bash
QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -m pytest -q
```
