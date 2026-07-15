"""Windows DPAPI helpers for user-scoped local secrets."""

from __future__ import annotations

import base64
import binascii
import ctypes
import os
from ctypes import wintypes


_PREFIX = "dpapi:v1:"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _windows_libraries():
    if os.name != "nt":
        raise OSError("Windows DPAPI is unavailable on this platform")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def _blob(payload: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(payload)
    pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    return _DataBlob(len(payload), pointer), buffer


def _raise_last_windows_error(action: str) -> None:
    error = ctypes.get_last_error()
    raise OSError(error, f"{action} failed: {ctypes.FormatError(error)}")


def protect_secret(secret: str) -> str:
    """Encrypt a secret for the current Windows user with DPAPI."""

    if not secret:
        return ""
    crypt32, kernel32 = _windows_libraries()
    input_blob, input_buffer = _blob(secret.encode("utf-8"))
    output_blob = _DataBlob()
    # Keep the input buffer alive until CryptProtectData returns.
    _ = input_buffer
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "instant-translate API key",
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    ):
        _raise_last_windows_error("DPAPI encryption")
    try:
        encrypted = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return _PREFIX + base64.b64encode(encrypted).decode("ascii")
    finally:
        kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))


def unprotect_secret(value: str) -> str:
    """Decrypt a value produced by :func:`protect_secret`."""

    if not value:
        return ""
    if not value.startswith(_PREFIX):
        raise ValueError("unsupported protected-secret format")
    try:
        encrypted = base64.b64decode(value[len(_PREFIX):], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid protected-secret payload") from exc

    crypt32, kernel32 = _windows_libraries()
    input_blob, input_buffer = _blob(encrypted)
    output_blob = _DataBlob()
    description = ctypes.c_void_p()
    _ = input_buffer
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        ctypes.byref(description),
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    ):
        _raise_last_windows_error("DPAPI decryption")
    try:
        plaintext = ctypes.string_at(output_blob.pbData, output_blob.cbData)
        return plaintext.decode("utf-8")
    finally:
        if description.value:
            kernel32.LocalFree(description)
        kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))
