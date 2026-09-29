import inspect

import classbot.main as main_module
from classbot.course import CourseProfile
from classbot.main import main


def test_console_entrypoint_is_importable():
    assert callable(main)
    assert inspect.signature(main).return_annotation is None


def test_default_recognizer_loads_selected_course_profile(monkeypatch):
    profile = CourseProfile(
        "demo",
        hotwords=("辩证唯物主义",),
        corrections={"违心": "唯心"},
    )
    monkeypatch.setenv("CLASSBOT_COURSE", "demo")
    monkeypatch.setattr(main_module, "load_course_profile", lambda _course: profile)

    recognizer = main_module._make_recognizer("cpu")

    assert recognizer.hotwords == ("辩证唯物主义",)
    assert recognizer.corrections == {"违心": "唯心"}
    assert recognizer.final_refinement is False


def test_default_detector_loads_selected_course_hotwords(monkeypatch):
    profile = CourseProfile("demo", hotwords=("物质", "意识"))
    monkeypatch.setenv("CLASSBOT_COURSE", "demo")
    monkeypatch.setattr(main_module, "load_course_profile", lambda _course: profile)

    detector = main_module._make_detector("哲学")

    assert detector.hotwords == ("物质", "意识")
