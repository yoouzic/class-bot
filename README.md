# Classbot

Classbot is a Windows desktop classroom listening assistant. It transcribes
speech locally, detects likely questions, and streams short answers from
DeepSeek.

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
CLASSBOT_SUBJECT=历史
```

Start the application with:

```powershell
python -m classbot
```

The floating window keeps each new speech segment as its own question card,
shows the current question with a session-relative `HH:MM:SS` timestamp, and
retains the recent question history. Explicit classroom vocatives are removed
from answer prompts, while long lecture scaffolding without a question ending
is filtered before an API request. Repeated questions from different speech
segments are intentionally kept.

The first startup downloads the `paraformer-zh-streaming` ASR model and can
take a while. Obtain appropriate classroom consent before using the app.

Classbot does not write raw audio to disk and does not play sound. Network,
DeepSeek API, and ASR failures are surfaced in the window as visible status
states, including `API 错误` and `ASR 不可用`.
