# Classbot

Classbot 是 Windows 线上答辩助手，适用于组会、竞赛和项目答辩。
从电脑播放设备读取钉钉、腾讯会议等软件的声音，本地实时转写并提取问题，
结合所选场景的代码和文档生成带来源的 DeepSeek 回答。原有课堂模式仍可选择。

## 仓库结构

- `src/classbot/`：应用代码（`asr/`、`audio/`、`knowledge/`、`llm/`、`ui/`）
- `tests/`：pytest 回归与离线评测用例
- `scripts/`：索引构建、ASR 评测、答辩/音频验证脚本，见 [scripts/README.md](scripts/README.md)
- `courses/`：课程热词、纠错与本地教材参考
- `docs/`：设计与验证文档，索引见 [docs/README.md](docs/README.md)
- `.cache/`、`.venv/`、`tmp/`、音频和 PDF 属于本地文件，不提交

## 线上答辩使用

1. 安装下面的项目依赖，然后双击 `start_classbot.bat`。
2. 点击“新建”，填写场景名（如某竞赛、某课题组会），选择资料文件夹。
3. 点击“刷新资料索引”。首次会下载约 100 MB 的本地中文向量模型；
   解析、向量计算和关键词检索都在本机进行。失败会显示原因，不会静默退回纯关键词检索。
   新场景默认“全量”；编辑场景可选择“均衡”。旧配置保留原来的均衡策略。
4. 选择钉钉/腾讯会议**实际播放声音的耳机或扬声器**，然后开始监听。
   电脑在该设备上播放的全部声音都会进入转写，请关闭其他音源。
5. 答辩回答逐条列出带来源的资料事实；依据不足时明确提示并给补证建议。卡片正文用
   中文短标签标出来源：`（教）`、`（PPT）`、`（xxx.py）`、`（文）`、`（PDF）`、`（网）`、`（通识）`；
   来源区列出对应的文件、代码行、Word 段落/表格、PDF页序或幻灯片位置。代码来源正文只
   显示文件名，来源区显示绝对路径和行号；来源区颜色：绿色=已引用本地资料，橙色=已引用
   网络，灰色=通识/待核/未引用候选。
   每条结论须绑定本次命中片段中的精确引文，完整校验后才显示；失败不会展示半成品。
   引文确实存在仍不等于结论已获语义证明，重要结论请结合原文核对。
6. 状态行旁的“对屏幕共享隐藏”默认开启：本窗口不会进入屏幕共享、录制和截图
   （Windows 10 2004 及以上系统支持；环境不支持或系统拒绝时勾选框自动停用并提示）。
   该设置只在本次运行有效，可随时切换，不影响监听。
7. 换场景时先暂停，再选择另一个场景并加载/刷新索引；历史卡片保留原场景标签。

支持常见代码、Markdown/TXT/JSON/YAML/TOML/LaTeX 等文本、DOCX 正文和表格、
文字型 PDF、PPTX 文本和表格。旧 `.doc` 请另存为 `.docx`；扫描 PDF 和图片需先 OCR。
解析失败、无可提取文字、超限或不支持的文件会显示在索引诊断中。
资料按文件逐一读取，不把整个文件夹的原始内容同时放进内存。精确相同格式的副本会去重，
优先纳入文档、代码和 PDF，之后处理 JSON/CSV 等批量数据。

| 策略 | 资料范围与容量 | 达到容量边界时 |
| --- | --- | --- |
| 全量 | 保留嵌套 `tmp`/`temp` 中的可解析资料；单文件 128 MiB、去重内容 16 GiB、400,000 文件、2,000,000 片段 | 明确失败并保留上一快照，不发布部分结果 |
| 均衡 | 跳过嵌套 `tmp`/`temp`；单文件 32 MiB、去重内容 2 GiB、5,000 文件、50,000 片段 | 整份跳过放不下的文件，显示“部分资料已索引” |

两种策略都排除隐藏、凭据、依赖目录、链接及不支持的格式；全量不包含图片 OCR，
也不表示普通文档中的所有嵌入对象都能提取。通过“索引详情”查看具体排除、解析错误和来源。
大型资料库首次计算向量需要时间。全量模式每 64 个片段保存进度，关闭后再次刷新可续建；
来源、解析结果、模型或编码算法改变时不会混用旧进度。完整快照原子发布，失败时保留上一可用版本。
全量向量采用只读内存映射，避免加载时再复制整份矩阵；同一资料索引禁止多进程同时构建。
相同文本的向量按模型和编码版本缓存在本机；不同文件的来源位置仍完整保留。

场景配置、文档片段和向量保存在 Git 忽略的 `.cache/classbot/`；不修改源文件，
不运行资料里的代码/宏，排除隐藏、凭据、依赖目录及链接文件；全量模式保留
`build`/`dist`/`out` 等目录内的可解析实验材料。
检索采用本地 BGE 语义向量加中文/代码标识符关键词。外发范围：问题会发送给 DeepSeek
Chat 生成回答，联网时同时发送给 Search/Fetch 检索服务；本地/网络候选片段发送给
DeepSeek Chat。资料变化后需要刷新；开始监听时会检查快照是否过期。
课堂与答辩都支持本地材料加网络检索：课堂在本地教材覆盖不足或问题含时效/外部信息信号
（最新/今年/现在/版本/哪一年/谁/多少人等）时联网；答辩仅在本地核验为 partial/insufficient
时补充网络一般背景。答辩的项目事实仍只由本地资料证明，网络补充不进入 `[S1]` 证据体系，
也不冒充项目事实。关闭 `CLASSBOT_WEB_SEARCH` 即恢复纯本地回答。
`CLASSBOT_WEB_FETCH=auto` 在摘要过薄（总长 <240 或单条 <80）或问题含版本/价格/参数/
步骤/代码/对比等细节信号时抓取 1–2 页；`off` 只用摘要；`always` 每次联网都抓取 1–2 页。
无 DeepSeek API key 时只用 Bing/DDG 摘要，不调用 Fetch。两个超时配置
`CLASSBOT_WEB_SEARCH_TIMEOUT`（默认 1.8 秒）和 `CLASSBOT_WEB_FETCH_TIMEOUT`
（默认 1.0 秒）限制对应阶段耗时。
多个场景共享同一身份的本地模型，内存最多保留最近三个资料索引；取消旧问题后，真实检索仍占有
许可直到计算结束，避免修订问题造成线程堆积。同时最多处理两个回答。

答辩模式只用系统回放，失败不会改用麦克风。一次采集一个播放设备；切换耳机或会议
输出设备后应暂停并重新选择。自己的麦克风发言通常不会进入系统回放，因此追问上下文
主要来自会议播放内容；屏幕/图片中的未口述内容也需要先放入可解析资料。
答辩分段等待约 600 毫秒静音，容纳问题中的短暂停顿，连续语音仍以 60 秒为上限。
专业词可能误识别；只有当前资料能唯一支持时，回答才明确说明按哪个资料术语理解，
原始问题保持不变。资料事实应按卡片定位核对，尤其注意版本、实验配置与完成状态。

默认 `CLASSBOT_MODE=defense`；设为 `classroom` 可启动到原课堂模式。
可用 `CLASSBOT_EMBEDDING_MODEL` 指定本地模型文件夹。模型和资料缓存位置可用
`CLASSBOT_CACHE_DIR` 指定。首次模型下载失败可在网络可用后重试。

## 资料库缓存管理

停止监听后点击“缓存管理”，可查看每个资料库的索引大小、文件/片段数量、来源路径，
以及共享向量、模型和其他缓存的占用。同一文件夹、模型和策略的不同场景共用一份磁盘索引。
内存中的三个资料库限制不会自动删除磁盘索引。

- 总上限默认 **10 GiB**，可在界面保存；统计包含缓存目录中所有文件的逻辑大小，
  也包含新旧快照同时存在和建库临时文件。Python/ASR 环境与缓存目录之外的原始资料不计入。
- 选择不用的索引并确认清理，或单独清理用于加速重建的共享向量缓存。
  模型、场景配置、未知文件和原始资料不属于清理目标，不会自动淘汰其他资料库。
- 正在构建或使用的索引受跨进程锁保护，不能清理。当前选中的已加载索引也受保护；
  如需清理它，先停止监听并切换到课堂模式，再打开缓存管理。
- 清理后须重新建立该资料库索引才能使用。容量不足时停止新增写入，保留旧完整快照和
  已保存的续建进度；提高上限或清理其他缓存后重试，不会切换到均衡策略或减少资料覆盖。
- 空间紧张时共享向量缓存可以只读复用或停用；仍为全部片段计算向量，改变的是重建速度。
  首次模型下载也按固定版本清单核对大小和摘要，并受同一配额约束。

资料库不能放在程序缓存目录内；程序缓存可以位于项目资料库下，扫描时会排除该缓存子目录。
配额设置在缓存目录的 `.cache-policy`，逐库登记为 `index-*.cache-info`，均为本地文件。

## 原课堂模式说明

## Install on Windows

Use Python 3.11 and a project virtual environment in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,asr]"
```

Create `.env` from `.env.example` and set at least:

```dotenv
DEEPSEEK_API_KEY=your-api-key
CLASSBOT_SUBJECT=马克思主义基本原理
CLASSBOT_COURSE=marxism_basic_principles_2023
CLASSBOT_THINKING=high
CLASSBOT_DEFENSE_THINKING=none
```

Start the application with:

```powershell
python -m classbot
```

For a one-click Windows start, double-click `start_classbot.bat`. It checks
the virtual environment, local `.env`, and ASR dependencies before opening the
microphone listener. To validate setup without launching the UI, run
`powershell -ExecutionPolicy Bypass -File .\start_classbot.ps1 -CheckOnly`.

The floating window keeps each new speech segment as its own question card,
shows the current question with a session-relative `HH:MM:SS` timestamp, and
retains the recent question history. Explicit classroom vocatives are removed
from answer prompts. The detector only sends stable textbook knowledge
questions and choice questions to the API; names, classroom-management speech,
isolated confirmations, and lecture scaffolding remain in the transcript but do
not create answer cards. A partial question may wait for a complete ending
before it is submitted. Repeated questions from different speech segments are
intentionally kept.

Live candidates are shown before submission. Non-final questions wait for
700 ms of stable question text before an answer request; a complete final
transcript can submit immediately. Rapid revisions replace the waiting task,
while trailing fillers and context changes on self-contained questions do not
restart an existing answer. A final transcript that rejects a provisional
question cancels its task and clears the unconfirmed answer. Bare confirmation
tails such as `是不是` wait for a complete proposition. Textbook retrieval
ignores generic question scaffolding so phrases like `什么样的情况下` alone
do not count as evidence for a page match.

Course-specific profiles live under `courses/`. Each profile can provide
`hotwords.txt`, `corrections.txt` for recognition fixes, and an optional local
`reference.txt` containing textbook excerpts. `CLASSBOT_COURSE` selects the
profile, so different courses can use different terminology and textbook
context without changing the source code.
`CLASSBOT_THINKING=high` enables a higher-effort DeepSeek reasoning pass in classroom
mode. Online defense uses the independent `CLASSBOT_DEFENSE_THINKING` setting,
which defaults to `none` for lower latency; it does not inherit classroom thinking.
Both settings accept `none`, `low`, `high`, or `max`; `disabled` is an alias for
`none`, and invalid values fall back to `none`. When a
question matches the local textbook index, the relevant excerpts are sent with
the question so the answer prioritizes the book's terminology and conclusions.
Answers also append the matching PDF page sequence, such as `PDF第37页`; the
page sequence is taken from the provided PDF and is not guessed by the model.
The PDF and generated reference index are ignored by Git and stay local.

Defense requests use the same independent thinking setting but return structured,
evidence-bound claims. Each request has one attempt; invalid JSON, missing or
fabricated quotations, and truncated responses show an error with no answer text.
Classroom responses keep their existing streaming behavior.

The first startup downloads the `paraformer-zh-streaming` ASR model and can
take a while. Obtain appropriate classroom consent before using the app.

Speech segments have a 60-second upper bound. The recognizer buffers short CPU
inference spikes; an exhausted ASR queue stops with an error instead of silently
discarding accepted audio or segment markers. The defense path avoids retaining
full-segment PCM when final refinement is disabled.

Classbot does not write raw audio to disk and does not play sound. Network,
DeepSeek API, and ASR failures are surfaced in the window as visible status
states, including `API 错误` and `ASR 不可用`.
