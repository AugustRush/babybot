from babybot.resource import (
    CallableTool,
    LoadedSkill,
    ResourceBrief,
    ResourceCatalog,
    ResourceManager,
    SkillRuntimeConfig,
    ToolGroup,
    WorkerRuntime,
)


def test_resource_module_exports_core_runtime_symbols() -> None:
    assert ResourceManager.__name__ == "ResourceManager"
    assert CallableTool.__name__ == "CallableTool"
    assert ResourceCatalog.__name__ == "ResourceCatalog"
    assert WorkerRuntime.__name__ == "WorkerRuntime"


def test_resource_module_exports_resource_model_symbols() -> None:
    assert ToolGroup.__name__ == "ToolGroup"
    assert ResourceBrief.__name__ == "ResourceBrief"
    assert LoadedSkill.__name__ == "LoadedSkill"
    assert SkillRuntimeConfig.__name__ == "SkillRuntimeConfig"
