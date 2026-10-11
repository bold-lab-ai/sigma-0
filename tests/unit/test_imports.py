import importlib


def test_core_packages_import() -> None:
    for module in (
        "environment",
        "environment.jaxen",
        "environment.jaxrl",
        "matching_engine",
        "matching_engine.jaxlobster",
        "matching_engine.jaxob",
        "matching_engine.utils",
        "base_model",
        "base_model.config",
        "lobpipeline",
    ):
        assert importlib.import_module(module).__name__ == module
