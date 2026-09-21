"""Canonical byte encodings, so that a hash says what the content is and nothing else.

Every value is written with a one-byte kind and explicit lengths, so that no two
different structures share an encoding:

    a string     b"S", its length in bytes as 8 bytes big-endian, its UTF-8 bytes
    an integer   b"I", 8 bytes big-endian, signed
    a double     b"F", 8 bytes little-endian IEEE 754
    a null       b"N"
    numbers      b"A", the number of dimensions as 8 bytes, each dimension as 8 bytes,
                 then the values as little-endian IEEE 754 doubles in row order
    a list       b"L", the number of items as 8 bytes, then each item

A hash built from these depends on values, not on a file format, a library version or
the order in which a dictionary happens to iterate. Floats are hashed bit for bit: -0.0
and 0.0 differ, and no rounding is applied. The encodings that use this module carry
their own version header (``docs/data_contract.md``).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

from process_transfer.cstr_variables import FloatArray

Scalar = str | int | float | None


def length(value: int) -> bytes:
    return int(value).to_bytes(8, "big")


def encode_string(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {value!r}")
    data = value.encode("utf-8")
    return b"S" + length(len(data)) + data


def encode_numbers(values: FloatArray) -> bytes:
    """Shape first, then the doubles: arrays of different shapes never share an encoding."""
    array = np.ascontiguousarray(values, dtype="<f8")
    shape = b"".join(length(size) for size in array.shape)
    return b"A" + length(array.ndim) + shape + array.tobytes()


def encode_scalar(value: Scalar) -> bytes:
    """A string, an integer, a finite double or a null. Booleans are refused: they are
    integers to Python, and a flag silently hashed as 0 or 1 is not what anyone meant."""
    if value is None:
        return b"N"
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("booleans have no canonical encoding here")
    if isinstance(value, str):
        return encode_string(value)
    if isinstance(value, (int, np.integer)):
        return b"I" + int(value).to_bytes(8, "big", signed=True)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError(f"a non-finite number has no place in stored content: {value!r}")
        return b"F" + struct.pack("<d", float(value))
    raise TypeError(f"no canonical encoding for {type(value).__name__}: {value!r}")


def encode_list(items: Sequence[bytes]) -> bytes:
    """A list of already encoded items."""
    return b"L" + length(len(items)) + b"".join(items)
