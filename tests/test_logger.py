from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app import logger


class LoggerPathTests(unittest.TestCase):
    def test_source_log_root_is_inside_project(self) -> None:
        project_root = Path(logger.__file__).resolve().parent.parent

        with patch.object(logger.sys, "frozen", False, create=True):
            self.assertEqual(
                logger._application_root(),
                project_root,
            )
            self.assertEqual(
                logger._log_root(),
                project_root / ".tmp" / "runtime" / "logs",
            )

    def test_frozen_log_root_is_beside_executable(self) -> None:
        executable = Path("F:/portable/instant-translate/instant-translate.exe")

        with (
            patch.object(logger.sys, "frozen", True, create=True),
            patch.object(logger.sys, "executable", str(executable)),
        ):
            self.assertEqual(
                logger._application_root(),
                executable.parent,
            )


if __name__ == "__main__":
    unittest.main()
