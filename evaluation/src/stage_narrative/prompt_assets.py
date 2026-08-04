from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_TASK_SPEC_PATH = _ROOT / "configs" / "task_specs" / "zh" / "knowledge_extraction.yaml"
_ENTITY_SCHEMA_PATH = _ROOT / "configs" / "schemas" / "zh" / "entity_types.json"
_RELATION_SCHEMA_PATH = _ROOT / "configs" / "schemas" / "zh" / "relation_groups.json"


def _load_task_specs() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(_TASK_SPEC_PATH.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, dict):
        raise ValueError(f"Invalid knowledge-extraction task specs: {_TASK_SPEC_PATH}")
    return tasks


def _load_entity_schema() -> list[dict[str, Any]]:
    payload = json.loads(_ENTITY_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid entity type schema: {_ENTITY_SCHEMA_PATH}")
    return payload


_TASK_SPECS = _load_task_specs()
_ENTITY_SCHEMA = _load_entity_schema()
_RELATION_SCHEMA = json.loads(_RELATION_SCHEMA_PATH.read_text(encoding="utf-8"))
if not isinstance(_RELATION_SCHEMA, list) or not _RELATION_SCHEMA:
    raise ValueError(f"Invalid relation-group schema: {_RELATION_SCHEMA_PATH}")


def entity_type_names() -> tuple[str, ...]:
    return tuple(str(item["type"]).strip() for item in _ENTITY_SCHEMA)


def entity_type_definitions() -> str:
    return _entity_type_definitions()


def extraction_prompt_values(prompt_id: str, **task_values: Any) -> dict[str, Any]:
    task = _TASK_SPECS.get(prompt_id)
    values = dict(task_values)
    if values.get("language") == "Chinese":
        values["language"] = "中文"
    elif values.get("language") == "English":
        values["language"] = "英文"
    if task is not None:
        values.update(
            {
                "task_name": task["task_name"],
                "task_goal": task["task_goal"],
                "global_constraints": _bullet_block(task.get("global_constraints", [])),
            }
        )
    if prompt_id in {"narrative_extraction", "narrative_repair"}:
        values["output_schema_block"] = str(
            _TASK_SPECS["narrative_extraction"]["output_schema"]
        ).strip()
    elif prompt_id in {"entity_extraction", "entity_repair"}:
        values["type_definitions"] = _entity_type_definitions()
        values["output_schema_block"] = _entity_output_schema()
    elif prompt_id in {"relation_extraction", "relation_repair"}:
        values["relation_definitions"] = _relation_definitions()
        values["output_schema_block"] = _relation_output_schema()
    return values


def prompt_asset_paths() -> tuple[Path, ...]:
    return (
        _TASK_SPEC_PATH.resolve(),
        _ENTITY_SCHEMA_PATH.resolve(),
        _RELATION_SCHEMA_PATH.resolve(),
    )


def _entity_type_definitions() -> str:
    blocks: list[str] = []
    for item in _ENTITY_SCHEMA:
        rules = _bullet_block(item.get("naming_rules", []))
        block = f"{item['type']}: {item['description']}"
        if rules:
            block += f"\n命名原则：\n{rules}"
        scope_rules = item.get("scope_rules")
        if isinstance(scope_rules, dict):
            block += (
                "\nscope 判定：\n"
                f"- global: {scope_rules.get('global', '')}\n"
                f"- local: {scope_rules.get('local', '')}"
            )
        blocks.append(block)
    return "\n\n".join(blocks)


def _entity_output_schema() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": "",
                    "entity_type": "使用一个给定类型名称",
                    "scope": "global|local",
                    "description": "",
                    "aliases": [],
                    "evidence": "当前片段中的简短原文",
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _relation_output_schema() -> str:
    return json.dumps(
        {
            "relations": [
                {
                    "subject": "已锁定实体名称或别名",
                    "predicate": "简洁关系名称",
                    "object": "已锁定实体名称或别名",
                    "description": "",
                    "evidence": "当前片段中的简短原文",
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    )


def _relation_definitions() -> str:
    blocks: list[str] = []
    for item in _RELATION_SCHEMA:
        block = f"{item['group']}: {item['description']}"
        criteria = _bullet_block(item.get("criteria", []))
        if criteria:
            block += f"\n判定标准：\n{criteria}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _bullet_block(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return "\n".join(f"- {str(value).strip()}" for value in values if str(value).strip())


def _field_block(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return "\n".join(f'- "{str(value).strip()}"' for value in values if str(value).strip())
