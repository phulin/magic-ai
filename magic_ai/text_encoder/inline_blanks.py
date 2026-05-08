"""Shared inline-blank metadata enums."""

from __future__ import annotations

from typing import Final

BLANK_GROUP_PER_BLANK: Final = 0
BLANK_GROUP_CROSS_BLANK: Final = 1
BLANK_GROUP_CONSTRAINED: Final = 2

_BLANK_GROUP_KIND_BY_NAME: Final[dict[str, int]] = {
    "PER_BLANK": BLANK_GROUP_PER_BLANK,
    "CROSS_BLANK": BLANK_GROUP_CROSS_BLANK,
    "CONSTRAINED": BLANK_GROUP_CONSTRAINED,
}


def blank_group_kind_id(name: str | int) -> int:
    """Translate a renderer blank group-kind name into its stable int enum."""

    if isinstance(name, int):
        return int(name)
    try:
        return _BLANK_GROUP_KIND_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown blank group_kind {name!r}") from exc
