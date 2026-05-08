"""Struct-of-arrays container for parallel per-row tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class Field:
    name: str
    dtype: torch.dtype
    fill: float | int | bool = 0
    inner_shape: tuple[int, ...] = ()


class AggregateTensor:
    """Named parallel tensors sharing a leading dim. Fields exposed as attributes."""

    _length: int
    _fields: tuple[Field, ...]
    _device: torch.device
    _tensors: dict[str, Tensor]

    def __init__(
        self,
        *,
        length: int,
        fields: tuple[Field, ...],
        device: torch.device | str = "cpu",
    ) -> None:
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError("AggregateTensor field names must be unique")
        self._length = int(length)
        self._fields = fields
        self._device = torch.device(device)
        self._tensors = {
            f.name: torch.full(
                (int(length), *f.inner_shape),
                f.fill,
                dtype=f.dtype,
                device=self._device,
            )
            for f in fields
        }

    def __getattr__(self, name: str) -> Tensor:
        tensors = self.__dict__.get("_tensors")
        if tensors is not None and name in tensors:
            return tensors[name]
        raise AttributeError(name)

    def reset(self) -> None:
        for f in self._fields:
            self._tensors[f.name].fill_(f.fill)

    def write(self, rows: Tensor, **values: Any) -> None:
        unknown = set(values) - set(self._tensors)
        if unknown:
            raise KeyError(f"unknown AggregateTensor fields: {sorted(unknown)}")
        for name, value in values.items():
            self._tensors[name][rows] = value
