import importlib
import sys
from pathlib import Path


SCRIPT_DIR = Path("skills/auto_skill_creator/scripts").resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

init_skill = importlib.import_module("init_skill")
quick_validate = importlib.import_module("quick_validate")


def test_init_skill_creates_workspace_skill_at_default_root(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "Demo Skill",
        target="workspace",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["scripts", "references"],
        include_examples=False,
    )

    assert skill_dir == tmp_path / "workspace" / "skills" / "demo-skill"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "scripts").is_dir()
    assert (skill_dir / "references").is_dir()
    assert not (skill_dir / "assets").exists()


def test_init_skill_creates_builtin_skill_at_default_root(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "builtin helper",
        target="builtin",
        workspace_skills_dir=tmp_path / "workspace" / "skills",
        builtin_skills_dir=tmp_path / "builtin" / "skills",
        resources=["assets"],
        include_examples=False,
    )

    assert skill_dir == tmp_path / "builtin" / "skills" / "builtin-helper"
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "assets").is_dir()


def test_parse_resources_rejects_unknown_resource_types() -> None:
    try:
        init_skill.parse_resources("scripts,unknown")
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected parse_resources to exit on invalid resource type")


def test_validate_skill_accepts_generated_skill(tmp_path: Path) -> None:
    skill_dir = init_skill.init_skill(
        "validator-skill",
        path=tmp_path,
        resources=[],
        include_examples=False,
    )
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n"
        "name: validator-skill\n"
        "description: Create and validate babybot skills.\n"
        "---\n\n"
        "# Validator Skill\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert valid, message


def test_validate_skill_rejects_todo_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: bad-skill\n"
        'description: "[TODO: fill me in]"\n'
        "---\n\n"
        "# Bad Skill\n",
        encoding="utf-8",
    )

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "TODO" in message


def test_validate_skill_rejects_unexpected_root_file(tmp_path: Path) -> None:
    skill_dir = tmp_path / "bad-root-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: bad-root-skill\n"
        "description: Valid description.\n"
        "---\n\n"
        "# Skill\n",
        encoding="utf-8",
    )
    (skill_dir / "README.md").write_text("extra\n", encoding="utf-8")

    valid, message = quick_validate.validate_skill(skill_dir)

    assert not valid
    assert "Unexpected file or directory" in message


def test_validate_skill_accepts_current_auto_skill_creator_folder_naming() -> None:
    valid, message = quick_validate.validate_skill(
        Path("skills/auto_skill_creator").resolve()
    )

    assert valid, message
