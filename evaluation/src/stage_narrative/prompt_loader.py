from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_VARIABLE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


@dataclass(frozen=True, slots=True)
class _YamlPromptSpec:
    prompt_id: str
    category: str
    template: str
    task_variables: tuple[str, ...]
    static_variables: tuple[str, ...]
    required_variables: tuple[str, ...]
    source_path: Path


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    system: str
    user: str
    variables: tuple[str, ...]
    source_paths: tuple[Path, ...]


class YamlPromptRegistry:
    """Task-spec prompt loader modeled after NarrativeKnowledgeWeaver."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Prompt directory does not exist: {self.root}")
        self._raw_by_id: dict[str, _YamlPromptSpec] = {}
        self._raw_by_relative_id: dict[str, _YamlPromptSpec] = {}
        self._all_source_paths: set[Path] = set()
        for path in sorted(self.root.rglob("*.yaml")):
            self._load_file(path)

    def get(self, prompt_id: str) -> PromptSpec:
        task = self._resolve(prompt_id)
        if task.source_path.name == "system_prompt.yaml":
            raise ValueError(f"System prompt cannot be used as a task prompt: {prompt_id}")
        system = self._raw_by_relative_id.get(
            f"{task.source_path.parent.relative_to(self.root).as_posix()}/system_prompt"
        )
        if system is None:
            raise FileNotFoundError(
                f"Missing sibling system_prompt.yaml for {task.source_path}"
            )
        variables = tuple(dict.fromkeys((*system.task_variables, *system.static_variables,
                                         *task.task_variables, *task.static_variables)))
        return PromptSpec(
            prompt_id=task.prompt_id,
            system=system.template,
            user=task.template,
            variables=variables,
            source_paths=(system.source_path, task.source_path),
        )

    def render(self, prompt_id: str, **values: Any) -> tuple[str, str]:
        spec = self.get(prompt_id)
        missing = set(spec.variables) - set(values)
        extra = set(values) - set(spec.variables)
        if missing or extra:
            raise ValueError(
                f"Prompt {prompt_id} variables mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return (
            self._render_text(spec.system, spec.variables, values),
            self._render_text(spec.user, spec.variables, values),
        )

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._all_source_paths))

    @staticmethod
    def _render_text(
        template: str, variables: tuple[str, ...], values: dict[str, Any]
    ) -> str:
        # Existing task templates retain doubled JSON braces for `.format` compatibility.
        rendered = template.replace("{{", "{").replace("}}", "}")
        allowed = set(variables)

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            return _stringify(values[name]) if name in allowed else match.group(0)

        return _VARIABLE.sub(replace, rendered)

    def _resolve(self, prompt_id: str) -> _YamlPromptSpec:
        normalized = str(prompt_id or "").strip().replace("\\", "/")
        if normalized.endswith(".yaml"):
            normalized = normalized[:-5]
        if "/" in normalized:
            try:
                return self._raw_by_relative_id[normalized]
            except KeyError as exc:
                raise KeyError(f"Unknown YAML prompt path: {prompt_id}") from exc
        try:
            return self._raw_by_id[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown or ambiguous YAML prompt id: {prompt_id}") from exc

    def _load_file(self, path: Path) -> None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Prompt YAML must contain a mapping: {path}")
        required_keys = {
            "id", "category", "name", "description", "task_variables",
            "static_variables", "template",
        }
        if set(payload) != required_keys:
            raise ValueError(
                f"Prompt YAML keys mismatch in {path}: "
                f"missing={sorted(required_keys - set(payload))}, "
                f"extra={sorted(set(payload) - required_keys)}"
            )
        prompt_id = str(payload["id"]).strip()
        category = str(payload["category"]).strip()
        template = str(payload["template"]).strip()
        if not prompt_id or not category or not template:
            raise ValueError(f"Prompt id, category, and template are required: {path}")
        task_vars, task_required = self._variables(payload["task_variables"], path)
        static_vars, static_required = self._variables(payload["static_variables"], path)
        declared = set(task_vars) | set(static_vars)
        used = set(_VARIABLE.findall(template))
        # Ignore doubled JSON examples; only declared placeholders are substituted.
        undeclared = {name for name in used if name not in declared}
        if undeclared:
            raise ValueError(f"Undeclared variables in {path}: {sorted(undeclared)}")
        relative_id = path.relative_to(self.root).with_suffix("").as_posix()
        spec = _YamlPromptSpec(
            prompt_id=prompt_id,
            category=category,
            template=template,
            task_variables=task_vars,
            static_variables=static_vars,
            required_variables=tuple(sorted(task_required | static_required)),
            source_path=path.resolve(),
        )
        if prompt_id in self._raw_by_id:
            # Bare ids must be unique, matching the reference loader's ambiguity rule.
            self._raw_by_id.pop(prompt_id, None)
        else:
            self._raw_by_id[prompt_id] = spec
        self._raw_by_relative_id[relative_id] = spec
        self._all_source_paths.add(path.resolve())

    @staticmethod
    def _variables(raw: Any, path: Path) -> tuple[tuple[str, ...], set[str]]:
        if not isinstance(raw, list):
            raise ValueError(f"Prompt variables must be a list: {path}")
        names: list[str] = []
        required: set[str] = set()
        for item in raw:
            if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                raise ValueError(f"Invalid variable declaration in {path}: {item!r}")
            name = str(item["name"]).strip()
            names.append(name)
            if item.get("required") is True:
                required.add(name)
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate variable declaration in {path}")
        return tuple(names), required


DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parents[2] / "configs" / "prompts"
PROMPTS = YamlPromptRegistry(DEFAULT_PROMPT_ROOT)
