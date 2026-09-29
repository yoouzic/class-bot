from pathlib import Path

import pytest

import classbot.course as course_module
from classbot.course import load_course_profile


def test_load_course_profile_reads_hotwords_and_corrections(tmp_path, monkeypatch):
    courses_root = tmp_path / "courses"
    profile_root = courses_root / "demo"
    profile_root.mkdir(parents=True)
    (profile_root / "hotwords.txt").write_text(
        "# comment\n马克思主义\n\n辩证唯物主义\n",
        encoding="utf-8",
    )
    (profile_root / "corrections.txt").write_text(
        "违心=>唯心\n# comment\n",
        encoding="utf-8",
    )
    (profile_root / "reference.txt").write_text(
        "[第1页] 商品的价值由社会必要劳动时间决定。\n"
        "[第2页] 物质决定意识，意识对物质具有能动的反作用。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(course_module, "COURSES_ROOT", courses_root)

    profile = load_course_profile("demo")

    assert profile.hotwords == ("马克思主义", "辩证唯物主义")
    assert profile.corrections == {"违心": "唯心"}
    assert profile.find_reference("物质和意识的关系") == (
        "[第2页] 物质决定意识，意识对物质具有能动的反作用。",
    )


def test_load_course_profile_without_course_returns_empty_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(course_module, "COURSES_ROOT", Path(tmp_path) / "courses")

    profile = load_course_profile("")

    assert profile.hotwords == ()
    assert profile.corrections == {}


def test_load_course_profile_rejects_unknown_course(tmp_path, monkeypatch):
    monkeypatch.setattr(course_module, "COURSES_ROOT", Path(tmp_path) / "courses")

    with pytest.raises(FileNotFoundError, match="unknown"):
        load_course_profile("unknown")


def test_marxism_profile_covers_observed_asr_errors():
    profile = load_course_profile("marxism_basic_principles_2023")

    assert profile.corrections == {
        **profile.corrections,
        "马自主义": "马克思主义",
        "马思思主": "马克思主义",
        "唯特性": "唯一特性",
        "随心主意": "唯心主义",
        "近带形儿上学": "近代形而上学",
        "可助论": "唯物论",
        "意味性": "一维性",
        "三维星": "三维性",
    }
