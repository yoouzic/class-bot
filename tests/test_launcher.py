from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_windows_launcher_has_checkable_start_contract():
    powershell = PROJECT_ROOT / "start_classbot.ps1"
    batch = PROJECT_ROOT / "start_classbot.bat"

    assert powershell.is_file()
    assert batch.is_file()

    script = powershell.read_text(encoding="utf-8")
    assert "CheckOnly" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "DEEPSEEK_API_KEY" in script
    assert "-m classbot" in script

    batch_text = batch.read_text(encoding="utf-8-sig")
    assert "start_classbot.ps1" in batch_text
