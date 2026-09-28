import inspect

from classbot.main import main


def test_console_entrypoint_is_importable():
    assert callable(main)
    assert inspect.signature(main).return_annotation is None
