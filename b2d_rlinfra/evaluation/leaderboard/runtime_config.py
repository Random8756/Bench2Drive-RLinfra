from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Union


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "leaderboard_eval"


def resolve_project_path(
    raw_path: Union[str, Path],
    *,
    base_dir: Path,
    project_root: Path = PROJECT_ROOT,
    allow_missing: bool = False,
) -> Path:
    path = Path(raw_path).expanduser()

    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(base_dir / path)
        candidates.append(project_root / path)

    seen = set()
    normalized = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    for candidate in normalized:
        if candidate.exists():
            return candidate

    if allow_missing and normalized:
        return normalized[0]

    raise FileNotFoundError(
        f"Unable to resolve existing path from {raw_path!r}; checked {normalized}"
    )


@dataclass(frozen=True)
class AgentConfig:
    rl_config_path: Path
    checkpoint_path: Path
    stochastic: bool = False
    output_root: Path = DEFAULT_OUTPUT_ROOT

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in ("1", "true", "yes", "on")


def load_agent_config(config_path: Union[str, Path]) -> AgentConfig:
    config_path = Path(config_path).expanduser().resolve()
    base_dir = config_path.parent

    checkpoint_raw = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_raw:
        raise ValueError("CHECKPOINT_PATH is required in the environment")

    output_root_raw = os.environ.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))

    rl_config_path = resolve_project_path(config_path, base_dir=base_dir)
    checkpoint_path = resolve_project_path(checkpoint_raw, base_dir=base_dir)
    output_root = resolve_project_path(
        output_root_raw,
        base_dir=base_dir,
        allow_missing=True,
    )

    return AgentConfig(
        rl_config_path=rl_config_path,
        checkpoint_path=checkpoint_path,
        stochastic=_parse_bool_env("STOCHASTIC", False),
        output_root=output_root,
    )
