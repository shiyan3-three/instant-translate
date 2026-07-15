from __future__ import annotations

import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from app import logger


class LoggerPathTests(unittest.TestCase):
    def test_source_log_root_is_inside_project(self) -> None:
        project_root = Path(logger.__file__).resolve().parent.parent

        with (
            patch.object(logger.sys, "frozen", False, create=True),
            patch.dict(logger.os.environ, {}, clear=True),
        ):
            self.assertEqual(
                logger._application_root(),
                project_root,
            )
            self.assertEqual(
                logger._log_root(),
                project_root / ".tmp" / "runtime" / "logs",
            )

    def test_frozen_log_root_uses_local_appdata_not_executable_directory(self) -> None:
        executable = Path("F:/portable/instant-translate/instant-translate.exe")

        with (
            patch.object(logger.sys, "frozen", True, create=True),
            patch.object(logger.sys, "executable", str(executable)),
            patch.dict(logger.os.environ, {"LOCALAPPDATA": "F:/Users/test/AppData/Local"}, clear=True),
        ):
            self.assertEqual(
                logger._application_root(),
                executable.parent,
            )
            self.assertEqual(
                logger._log_root(),
                Path("F:/Users/test/AppData/Local") / "instant-translate" / "logs",
            )

    def test_test_log_root_is_separate_from_runtime_and_retention_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            test_root = Path(tmp) / "test-logs"
            with patch.dict(
                logger.os.environ,
                {"INSTANT_TRANSLATE_TEST_LOG_DIR": str(test_root)},
                clear=True,
            ):
                self.assertEqual(logger._log_root(), test_root)
                for index in range(31):
                    path = test_root / f"old-{index}.log"
                    path.write_text("x", encoding="utf-8")
                    os.utime(path, (index, index))
                logger._prune_old_logs(test_root, keep=4)
            self.assertEqual(len(list(test_root.glob("*.log"))), 4)


if __name__ == "__main__":
    unittest.main()
