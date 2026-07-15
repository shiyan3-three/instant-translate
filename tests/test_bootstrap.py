"""Tests for top-level application bootstrap objects."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.app_context import ApplicationContext
from app.hotkeys import HotkeyMap
from app.main import (
    _acquire_instance_lock,
    _apply_hotkey_change,
    _run_ocr_smoke,
    build_application_context,
)
from app.runtime import ensure_supported_python, is_supported_python
from app.settings import AppSettings


class BuildApplicationContextTests(unittest.TestCase):
    """Verify the default desktop runtime state."""

    def test_build_application_context_uses_expected_defaults(self) -> None:
        context = build_application_context()

        self.assertIsInstance(context, ApplicationContext)
        self.assertEqual(context.hotkeys.create_selection, "Ctrl+Shift+Z")
        self.assertEqual(context.hotkeys.toggle_edit_mode, "Ctrl+Shift+X")
        self.assertEqual(context.active_group_count, 0)
        self.assertTrue(
            context.settings.prompt.compiled_prompt_path.endswith("compiled-prompt.md"),
            f"compiled_prompt_path should end with 'compiled-prompt.md', got {context.settings.prompt.compiled_prompt_path!r}",
        )

    def test_instance_lock_rejects_second_process_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "app.lock"
            first = _acquire_instance_lock(path)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(_acquire_instance_lock(path))
            finally:
                first.unlock()
            third = _acquire_instance_lock(path)
            self.assertIsNotNone(third)
            third.unlock()

    def test_settings_load_normalizes_prompt_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "ai": {},
                        "prompt": {
                            "compiled_prompt_path": "prompts/prompts/compiled-prompt.md"
                        },
                    }
                ),
                encoding="utf-8",
            )

            settings = AppSettings.load(settings_path)

        self.assertEqual(settings.prompt.compiled_prompt_path, "prompts/compiled-prompt.md")

    def test_settings_load_maps_legacy_model_to_fast_and_thinking_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "ai": {
                            "base_url": "https://api.example.test/v1",
                            "api_key": "key",
                            "model": "legacy-model",
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = AppSettings.load(settings_path)

        self.assertEqual(settings.ai.model, "legacy-model")
        self.assertEqual(settings.ai.fast_model_name, "legacy-model")
        self.assertEqual(settings.ai.thinking_model_name, "legacy-model")

    def test_settings_load_accepts_utf8_bom_without_losing_valid_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            payload = {
                "ai": {
                    "base_url": "https://api.example.test/v1",
                    "api_key": "legacy-key",
                    "fast_model": "fast-model",
                    "thinking_model": "thinking-model",
                },
                "prompt": {
                    "constraints_text": "Keep prompt structure.",
                    "knowledge_reference_paths": ["terms.md"],
                    "compiled_prompt_path": "prompts/compiled-prompt.md",
                },
                "default_source_language": "English",
                "default_target_language": "\u4e2d\u6587",
                "hotkey_create_selection": "Ctrl+Alt+1",
                "hotkey_toggle_edit_mode": "Ctrl+Alt+2",
            }
            settings_path.write_bytes(
                b"\xef\xbb\xbf" + json.dumps(payload, ensure_ascii=False).encode("utf-8")
            )

            loaded = AppSettings.load(settings_path)

        self.assertEqual(loaded.ai.base_url, "https://api.example.test/v1")
        self.assertEqual(loaded.ai.api_key, "legacy-key")
        self.assertEqual(loaded.ai.fast_model, "fast-model")
        self.assertEqual(loaded.ai.thinking_model, "thinking-model")
        self.assertEqual(loaded.prompt.constraints_text, "Keep prompt structure.")
        self.assertEqual(loaded.prompt.knowledge_reference_paths, ["terms.md"])
        self.assertEqual(loaded.default_source_language, "English")
        self.assertEqual(loaded.default_target_language, "\u4e2d\u6587")
        self.assertEqual(loaded.hotkey_create_selection, "Ctrl+Alt+1")
        self.assertEqual(loaded.hotkey_toggle_edit_mode, "Ctrl+Alt+2")

    def test_settings_save_and_load_dual_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            settings = AppSettings()
            settings.ai.base_url = "https://api.example.test/v1"
            settings.ai.api_key = "key"
            settings.ai.fast_model = "deepseek-v4-flash"
            settings.ai.thinking_model = "deepseek-v4-pro"

            settings.save(settings_path)
            raw_settings = settings_path.read_text(encoding="utf-8")
            loaded = AppSettings.load(settings_path)

        self.assertNotIn('"api_key"', raw_settings)
        self.assertNotIn('"key"', raw_settings)
        self.assertIn('"api_key_protected"', raw_settings)
        self.assertEqual(loaded.ai.api_key, "key")
        self.assertEqual(loaded.ai.fast_model_name, "deepseek-v4-flash")
        self.assertEqual(loaded.ai.thinking_model_name, "deepseek-v4-pro")

    def test_runtime_version_guard_accepts_declared_range_and_rejects_unsupported(self) -> None:
        self.assertTrue(is_supported_python((3, 10, 0)))
        self.assertTrue(is_supported_python((3, 12, 9)))
        self.assertFalse(is_supported_python((3, 9, 18)))
        self.assertFalse(is_supported_python((3, 13, 0)))
        with self.assertRaisesRegex(RuntimeError, "Python 3.10 through 3.12"):
            ensure_supported_python((3, 9, 18))

    def test_settings_save_is_atomic_and_keeps_old_file_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"sentinel": true}', encoding="utf-8")
            settings = AppSettings()
            settings.ai.api_key = "secret-key"
            with patch("app.settings.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    settings.save(path)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"sentinel": true}')
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_settings_load_recovers_from_valid_json_with_invalid_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            for payload in ([], {"ai": []}, {"prompt": "invalid"}):
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    loaded = AppSettings.load(path)
                    self.assertEqual(loaded.ai.api_key, "")
                    self.assertEqual(loaded.default_source_language, "English")

    def test_settings_load_discards_invalid_field_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps({
                    "ai": {"api_key": 123, "base_url": "https://example.test"},
                    "prompt": {"knowledge_reference_paths": ["valid.md", 42]},
                    "default_source_language": False,
                }),
                encoding="utf-8",
            )
            loaded = AppSettings.load(path)

        self.assertEqual(loaded.ai.api_key, "")
        self.assertEqual(loaded.ai.base_url, "https://example.test")
        self.assertEqual(loaded.prompt.knowledge_reference_paths, ["valid.md"])
        self.assertEqual(loaded.default_source_language, "English")

    def test_ocr_smoke_requires_a_warmed_paddle_backend(self) -> None:
        engine = Mock()
        engine._backend = "paddle"
        engine._warmed_languages = {"ch"}
        with patch("app.ocr.engine.OcrEngine", return_value=engine):
            self.assertEqual(_run_ocr_smoke(), 0)

        engine._backend = "tesseract"
        engine._warmed_languages = set()
        with patch("app.ocr.engine.OcrEngine", return_value=engine):
            self.assertEqual(_run_ocr_smoke(), 2)

    def test_hotkey_save_failure_rolls_back_runtime_and_memory(self) -> None:
        context = ApplicationContext()
        old_map = context.hotkeys
        context.settings.save = Mock(side_effect=OSError("disk full"))
        hotkeys = Mock()
        hotkeys.re_register.side_effect = [(True, ""), (True, "")]
        window = Mock()
        new_map = HotkeyMap("Ctrl+Shift+A", "Ctrl+Shift+B")

        _apply_hotkey_change(context, hotkeys, window, new_map)

        self.assertEqual(context.hotkeys, old_map)
        self.assertEqual(context.settings.hotkey_create_selection, old_map.create_selection)
        self.assertEqual(context.settings.hotkey_toggle_edit_mode, old_map.toggle_edit_mode)
        self.assertEqual(hotkeys.re_register.call_args_list[1].args, (old_map,))
        window.show_hotkey_result.assert_called_once()
        self.assertFalse(window.show_hotkey_result.call_args.args[0])
