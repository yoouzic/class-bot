# 课堂实时问答助手设计

## 1. 目标与边界

本项目构建一个运行在 Windows 上的课堂辅助桌面应用：在用户明确开启监听后，从笔记本麦克风接收教室声音，实时转写中文语音，自动判断其中是否出现完整的提问，再把疑似问题以文本形式发送给 DeepSeek V4.1 Flash，并把流式简短回答按问题分卡片显示。

第一版的目标是让“马克思的生辰是什么”这类历史课事实问题在老师说完核心问题后尽快出现答案。目标体验是在收音清楚、网络稳定时，答案首个文字约 1～2 秒内开始显示，短答案约 2～4 秒内完成。该数值是工程目标，不是云服务的 SLA。

第一版不做隐蔽录音、原始音频持久化、语音播报、移动端、离线大模型、可靠的教师身份识别或考试自动作答。界面必须持续显示麦克风工作状态，并提供暂停/继续控制；使用前应取得课堂参与者和授课方的适当同意。

## 2. 方案选择

采用“本地实时音频处理 + 文本问答 API”的混合架构。

- 本地完成麦克风采集、语音活动检测、临时转写缓存、问题候选判断和 UI 展示，避免把连续环境声音全部上传。
- 语音识别使用可替换的实时中文 ASR 接口；第一版默认使用 FunASR 的 `paraformer-zh-streaming`，按官方示例以约 600 ms 音频块和约 300 ms look-ahead 做本地流式识别，避免把 DeepSeek 当成音频识别服务。
- 问答使用 DeepSeek V4.1 Flash。官方文档当前将该版本的 API 模型名写作 `deepseek-flash`，Chat Completions 支持 `stream: true` 的 SSE 增量输出；官方请求结构是文本/图像内容，因此麦克风音频必须先经过 ASR 转成文本。
- UI 使用 PySide6，作为 Windows 桌面悬浮窗运行。第一版只输出文字，不向教室播放声音，避免打扰课堂或形成声学反馈。

该方案比“录完整段 WAV 后上传”更快，也保留了替换 ASR、问答模型和 UI 的边界。DeepSeek API 参考：

- https://api-docs.deepseek.com/updates/
- https://api-docs.deepseek.com/api/create-chat-completion/

## 3. 用户流程

1. 用户启动应用，填写或读取本机保存的 DeepSeek API Key，选择学科上下文，例如“高中历史”。
2. 用户点击“开始监听”；窗口显示绿色或明确的“正在监听”状态。
3. 音频以 16 kHz、单声道 PCM 小块进入本地环形缓冲区。VAD 过滤长时间静音，ASR 持续产生 partial/final 文本。
4. 问题检测器读取 partial 文本，按疑问词、句式、文本稳定性和短暂停顿计算候选分数。达到阈值且文本已有足够语义内容时，提前创建问题卡片并开始请求回答；不满足阈值的普通课堂陈述只保留在短期转写区，不上传。
5. DeepSeek 请求使用短答案系统提示词和当前学科上下文，开启流式输出。每个 SSE 增量立即追加到对应问题卡片。
6. 如果后续 ASR final 文本与已提交的问题有实质变化，关闭当前流并重新提交修正后的问题；如果只是标点或空格变化，不重复请求。
7. 问题完成后卡片保留在最近问题列表中，默认只保留文本和状态，不保留原始音频。

## 4. 组件与职责

### 4.1 `AudioCapture`

- 使用 `sounddevice` 打开默认输入设备，输出固定采样率的 PCM 帧。
- 将帧放入有界异步队列；队列满时丢弃最旧的未处理帧，不能阻塞 UI 线程。
- 暴露 `start()`、`stop()`、`frames()` 和设备错误状态。
- 不写入音频文件。

### 4.2 `VoiceActivityDetector`

- 对 PCM 帧做语音/静音判定，默认使用 WebRTC VAD 或同等本地实现。
- 通过 200～400 ms 的连续静音结束一个短语音段，并保留前后少量 padding，减少首字和尾字截断。
- 仅向 ASR 发送语音帧及必要的短上下文。

### 4.3 `SpeechRecognizer`

定义与具体 ASR 无关的接口：

```text
start() -> Awaitable[None]
push_audio(frame: bytes) -> None
finish_segment() -> None
events() -> AsyncIterator[TranscriptEvent]
stop() -> Awaitable[None]
```

`TranscriptEvent` 至少包含：`text`、`is_final`、`start_ms`、`end_ms`、`confidence` 和可选的 `segment_id`。真实 FunASR 事件始终提供递增 `segment_id`；时间字段只表示段内时序元数据。第一版提供基于 FunASR `paraformer-zh-streaming` 的本地实时中文 ASR 实现和 fake 实现；`finish_segment()` 将当前缓存标记为 final 并清理流式 cache。真实 ASR 的模型加载失败必须在 UI 中显示可诊断错误，而不是静默假装监听成功。实现参考：https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/README.md

### 4.4 `QuestionDetector`

问题检测器只处理文本，不依赖模型 API。它维护一个小状态机：`LISTENING`、`SPEAKING`、`CANDIDATE`、`SUBMITTED`、`COOLDOWN`，并以有界的 per-segment active 状态保存异步到达的 partial/final 修订。

候选分数由以下信号组成：

- 中文疑问词或疑问句式，例如“什么、谁、哪年、为什么、如何、是否、多少、吗、呢、请问”；
- 文本是否包含足够的主题实体和谓词，而不是只有“什么”；
- partial 文本在多个 ASR 事件中是否稳定；
- 语音段末尾是否出现约 300～700 ms 的停顿；
- 与最近已提交问题的归一化文本相似度，避免重复提交。

当候选达到阈值时，检测器发出 `QuestionCandidate`：

```text
question_id: str
text: str
subject: str
confidence: float
source_span: tuple[int, int]
```

第一版不声称能够识别说话人。所有检测到的提问都显示来源状态；“只识别老师”属于后续的说话人分离/声纹功能。

### 4.5 `DeepSeekClient`

- 使用 `POST https://api.deepseek.com/chat/completions`。
- 默认 `model` 为 `deepseek-flash`，`stream` 为 `true`。
- API Key 从 Windows Credential Manager 或环境变量读取，不写入源码、日志和问题卡片。
- 每次请求只发送已确认的问题文本、学科上下文和必要的回答格式要求，不发送原始音频。
- 暴露 `answer(question, subject, cancel_token) -> AsyncIterator[AnswerDelta]`。
- 客户端能在问题被 ASR 修正时主动关闭 SSE 连接，并让旧问题卡片进入 `cancelled` 状态。

回答提示词要求：

```text
你是课堂简答助手。请根据给定学科回答问题。
只给出直接、准确、简短的答案，优先使用 1～3 句中文。
如果事实不确定，请明确说“不确定”，不要编造。
不要输出推理过程、寒暄、标题或 Markdown 代码块。
```

### 4.6 `QuestionStore`

- 管理问题卡片及其状态：`detected`、`answering`、`answered`、`cancelled`、`error`。
- 按 `question_id` 保存 partial answer，保证多个问题不会互相串流。
- 仅保存在内存中；应用退出时清空。

### 4.7 `MainWindow`

窗口保持小尺寸、置顶但可隐藏。必须展示：

- 监听状态和暂停/继续按钮；
- 当前临时转写；
- 问题卡片列表；
- 每张卡片的问题文本、回答文本、回答状态和错误提示；
- 麦克风、ASR、网络/API 的独立状态。

UI 线程不能直接执行音频读取、ASR 推理或网络请求；所有外部工作通过 asyncio worker 与 Qt 信号/槽或等价桥接传回 UI。

## 5. 延迟策略

目标链路如下：

```text
音频帧 20～40 ms
→ VAD/ASR 持续 partial
→ 稳定文本 + 疑问句式达到阈值
→ 立即提交 DeepSeek
→ SSE 增量显示
```

为了“提前”回答，允许在 final 文本之前提交强候选，但必须满足两个条件：候选至少包含主题实体和问题谓词；ASR 后续修正会触发取消和重试。不能仅凭单个“什么”或极短 partial 文本触发。

第一版记录三个本地计时点但不记录音频：`candidate_detected_at`、`request_started_at`、`first_answer_delta_at`。日志只保存耗时、状态和错误类型，默认不保存完整课堂文本。

## 6. 错误处理

- 麦克风权限或设备打开失败：停止监听，显示设备名称和可操作的错误信息。
- ASR 模型加载失败：状态显示“ASR 不可用”，禁止显示“正在监听”的假成功状态。
- API Key 缺失或无效：保留转写功能，回答卡片显示配置错误，不循环重试。
- 网络超时或 SSE 中断：保留已经收到的部分答案，标记为错误，并只自动重试一次。
- API 限流：暂停新请求一小段时间，继续收音，不阻塞 UI；提示用户降低触发频率。
- 重复 partial/final 事件：通过文本归一化和问题 ID 去重。
- 模型回答为空或超过长度：显示“未得到有效简答”，不把异常堆栈暴露在课堂主界面。

## 7. 测试与验收

### 自动化测试

- `QuestionDetector` 对“马克思的生辰是什么”只生成一个候选，普通陈述“马克思出生于德国”不生成候选。
- partial 文本逐字增长时，检测器不会在每个事件重复创建问题。
- 同一 segment 的 final 文本修正沿用原 `question_id` 并更新候选，不会出现两个相同卡片。
- fake DeepSeek SSE 分片 `“马”`、`“克思”`、`“出生于”`、`“1818 年。”` 能按顺序显示在同一张卡片中。
- 网络错误、空回答和 API Key 错误会进入预期状态，音频采集任务不会因此崩溃。
- 应用退出后，临时问题和回答不再存在；测试目录中不得生成音频文件。

### 手工验收

1. 使用获得授权的课堂录音或现场测试，开启监听后窗口明确显示监听状态。
2. 连续提出两个问题，两个问题卡片分别显示，回答内容不串卡。
3. 在回答流式显示时点击暂停，后续音频不会创建新请求。
4. 拔掉麦克风或断开网络，窗口能给出可读状态，恢复后可继续使用。
5. 记录 10 次问题的候选到首个回答增量耗时；稳定网络下中位数目标不超过 2 秒，异常请求必须可解释。

## 8. 后续扩展

在第一版可稳定识别、回答和排版后，再考虑：

- 教师/学生说话人区分；
- 历史课教材或用户资料的本地检索增强；
- 可选的答案可信度和来源提示；
- 本地 ASR 与云端 ASR 的切换；
- 课堂结束后一键导出经用户确认的问答笔记。

