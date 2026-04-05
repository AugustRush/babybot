from __future__ import annotations

import asyncio
import inspect


def _has_asyncio_plugin(config) -> bool:  # type: ignore[no-untyped-def]
    pluginmanager = config.pluginmanager
    return pluginmanager.hasplugin("asyncio") or pluginmanager.hasplugin(
        "pytest_asyncio"
    )


def pytest_addoption(parser) -> None:  # type: ignore[no-untyped-def]
    parser.addini(
        "asyncio_mode",
        "pytest-asyncio compatibility option; ignored by the local fallback runner.",
        default="auto",
    )


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    config.addinivalue_line(
        "markers",
        "asyncio: run an async test function with the local asyncio fallback when pytest-asyncio is unavailable.",
    )


def pytest_pyfunc_call(pyfuncitem) -> bool | None:  # type: ignore[no-untyped-def]
    if _has_asyncio_plugin(pyfuncitem.config):
        return None

    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    if pyfuncitem.get_closest_marker("asyncio") is None:
        return None

    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
        if name in pyfuncitem.funcargs
    }
    asyncio.run(test_func(**kwargs))
    return True
