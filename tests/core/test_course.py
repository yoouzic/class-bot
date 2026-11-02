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


def test_reference_reader_compacts_pdf_cjk_spacing(tmp_path, monkeypatch):
    courses_root = tmp_path / "courses"
    profile_root = courses_root / "demo"
    profile_root.mkdir(parents=True)
    (profile_root / "reference.txt").write_text(
        "[第1页] 马 克 思 主 义 基 本 原 理。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(course_module, "COURSES_ROOT", courses_root)

    profile = load_course_profile("demo")

    assert profile.reference_chunks == ("[第1页] 马克思主义基本原理。",)


def test_find_reference_excerpts_returns_hit_window_with_page_marker():
    profile = course_module.CourseProfile(
        course_id="demo",
        reference_chunks=(
            "[第7页] "
            + "导入内容。" * 120
            + "哲学的基本问题是思维和存在的关系问题。"
            + "后续内容。" * 120,
        ),
    )

    excerpts = profile.find_reference_excerpts(
        "哲学的基本问题是什么", limit=1, max_chars=180
    )

    assert len(excerpts) == 1
    assert excerpts[0].startswith("[第7页]")
    assert "哲学的基本问题是思维和存在的关系问题" in excerpts[0]
    assert len(excerpts[0]) <= 180


def test_generic_question_words_do_not_retrieve_unrelated_textbook_pages():
    profile = course_module.CourseProfile("demo", reference_chunks=(
        "[第335页] 在什么样的情况下采取措施取决于具体的历史环境。",
        "[第88页] 认识的过程包括由感性认识到理性认识的飞跃。",
    ))
    assert profile.find_reference_excerpts("呃一个珍的的问题在什什么样的情况下呢") == ()
    assert profile.find_reference_excerpts("认识的过程有哪些") == (
        "[第88页] 认识的过程包括由感性认识到理性认识的飞跃。",
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
        "第一心原理": "第一性原理",
    }


def test_find_reference_records_returns_page_label():
    profile = course_module.CourseProfile(
        course_id="demo",
        reference_chunks=(
            "[第7页] 哲学的基本问题是思维和存在的关系问题。" * 20,
        ),
    )
    records = profile.find_reference_records("哲学的基本问题是什么", limit=1)
    assert len(records) == 1
    assert records[0].page_label == "第7页"
    assert "哲学的基本问题" in records[0].text


def test_reference_coverage_matches_and_misses():
    profile = course_module.CourseProfile(
        course_id="demo",
        reference_chunks=(
            "[第2页] 物质决定意识，意识对物质具有能动的反作用。",
        ),
    )
    assert profile.reference_coverage("物质和意识的关系") >= 0.55
    assert profile.reference_coverage("随机森林的泛化性") == 0.0
    assert profile.reference_coverage("是什么") == 1.0


def test_reference_coverage_without_references_is_zero():
    profile = course_module.CourseProfile(course_id="demo")
    assert profile.reference_coverage("任何一个问题") == 0.0
