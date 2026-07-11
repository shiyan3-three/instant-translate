"""Prompt storage helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.prompt.policy import ConstraintPolicy
from app.prompt.runtime_profile import RuntimeProfile
from app.reference_layer import (
    ReferencePackage,
    extract_ai_optimization_reference_candidates,
    format_reference_candidate_markdown,
)
from app.settings import AppSettings


DEFAULT_COMPILED_PROMPT_PATH = "prompts/compiled-prompt.md"
DEFAULT_REFERENCE_DIR = "prompts/references"


def _project_root() -> Path:
    """Return the project root, or the executable directory when frozen."""

    try:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        return Path(__file__).resolve().parent.parent.parent
    except Exception:
        return Path.cwd()


class PromptStorage:
    """Load and save prompt-layer files and the compiled prompt."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self._config_dir = config_dir or AppSettings.config_dir()
        self._legacy_project_root = _project_root() if config_dir is None else None

    def resolve_compiled_prompt_path(self, path: str | Path) -> Path:
        """Resolve a compiled prompt under the install/project directory."""

        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        if raw_path.is_absolute():
            return raw_path

        root = self._base_dir()
        parts = raw_path.parts
        if parts and parts[0].lower() == "prompts":
            return root / raw_path
        return root / "prompts" / raw_path

    def stored_compiled_prompt_path(self, path: str | Path) -> str:
        """Return the stable relative path stored in settings."""

        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        if raw_path.is_absolute():
            return DEFAULT_COMPILED_PROMPT_PATH
        parts = raw_path.parts
        if parts and parts[0].lower() == "prompts":
            return raw_path.as_posix()
        return (Path("prompts") / raw_path).as_posix()

    def reference_dir(self) -> Path:
        """Return the install/project-local directory for knowledge references."""

        return self._base_dir() / DEFAULT_REFERENCE_DIR

    def ensure_reference_dir(self) -> Path:
        """Create and return the knowledge-reference directory."""

        path = self.reference_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_ai_optimization_reference_candidates(
        self,
        path: str | Path = DEFAULT_COMPILED_PROMPT_PATH,
        *,
        filename: str = "ai-optimization-candidates.md",
    ) -> Path | None:
        """Write review-only glossary candidates extracted from a legacy prompt.

        The exported file is not added to ``knowledge_reference_paths``.  Users
        must review and enable it explicitly before those candidates become
        trusted local reference rules.
        """

        content = self.load_compiled_prompt(path)
        entries = extract_ai_optimization_reference_candidates(content)
        if not entries:
            return None
        safe_name = Path(filename).name or "ai-optimization-candidates.md"
        candidate_path = self.ensure_reference_dir() / safe_name
        candidate_path.write_text(
            format_reference_candidate_markdown(entries),
            encoding="utf-8",
        )
        return candidate_path

    def preview_reference_file(self, path: str | Path) -> ReferencePackage:
        """Parse one knowledge-reference markdown file for user preview."""

        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError:
            return ReferencePackage()
        return ReferencePackage.from_texts([content])

    def save_compiled_prompt(
        self,
        content: str,
        path: str | Path,
        policy: ConstraintPolicy | dict | None = None,
        reference_package: ReferencePackage | dict | None = None,
        runtime_profile: RuntimeProfile | dict | None = None,
        clear_runtime_profile: bool = False,
    ) -> Path:
        """Write the prompt and its optional declarative-policy sidecar."""

        if runtime_profile is not None and clear_runtime_profile:
            raise ValueError("runtime_profile and clear_runtime_profile cannot be used together")

        prompt_path = self.resolve_compiled_prompt_path(path)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        if policy is not None:
            parsed_policy = (
                policy if isinstance(policy, ConstraintPolicy) else ConstraintPolicy.from_dict(policy, strict=True)
            )
            policy_path = self.resolve_compiled_policy_path(path)
            policy_tmp = policy_path.with_suffix(policy_path.suffix + ".tmp")
            policy_tmp.write_text(
                json.dumps(parsed_policy.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            policy_tmp.replace(policy_path)
        if reference_package is not None:
            parsed_references = (
                reference_package
                if isinstance(reference_package, ReferencePackage)
                else ReferencePackage.from_dict(reference_package, strict=True)
            )
            references_path = self.resolve_compiled_references_path(path)
            references_tmp = references_path.with_suffix(references_path.suffix + ".tmp")
            references_tmp.write_text(
                json.dumps(parsed_references.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            references_tmp.replace(references_path)
        if runtime_profile is not None:
            runtime_profile_data = (
                runtime_profile.to_dict()
                if isinstance(runtime_profile, RuntimeProfile)
                else runtime_profile
            )
            parsed_runtime_profile = RuntimeProfile.from_dict(
                runtime_profile_data,
                strict=True,
            )
            runtime_profile_path = self.resolve_compiled_runtime_profile_path(path)
            runtime_profile_tmp = runtime_profile_path.with_suffix(runtime_profile_path.suffix + ".tmp")
            runtime_profile_tmp.write_text(
                json.dumps(parsed_runtime_profile.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            runtime_profile_tmp.replace(runtime_profile_path)
        elif clear_runtime_profile:
            runtime_profile_path = self.resolve_compiled_runtime_profile_path(path)
            try:
                runtime_profile_path.unlink()
            except FileNotFoundError:
                pass
        tmp = prompt_path.with_suffix(prompt_path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(prompt_path)
        return prompt_path

    def resolve_compiled_policy_path(self, path: str | Path) -> Path:
        prompt_path = self.resolve_compiled_prompt_path(path)
        return prompt_path.with_suffix(".policy.json")

    def resolve_compiled_references_path(self, path: str | Path) -> Path:
        prompt_path = self.resolve_compiled_prompt_path(path)
        return prompt_path.with_suffix(".references.json")

    def resolve_compiled_runtime_profile_path(self, path: str | Path) -> Path:
        prompt_path = self.resolve_compiled_prompt_path(path)
        return prompt_path.with_suffix(".runtime-profile.json")

    def load_compiled_policy(self, path: str | Path) -> ConstraintPolicy:
        """Load a validated policy sidecar, returning an empty policy when absent."""

        candidates = [self.resolve_compiled_policy_path(path)]
        if self._legacy_project_root is not None:
            raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
            candidates.append((self._legacy_project_root / raw_path).with_suffix(".policy.json"))
        for policy_path in candidates:
            policy = self._read_policy_sidecar(policy_path)
            if policy.rules:
                return policy
        return ConstraintPolicy()

    def load_compiled_references(self, path: str | Path) -> ReferencePackage:
        """Load the structured reference sidecar, returning empty package when absent."""

        candidates = [self.resolve_compiled_references_path(path)]
        try:
            existing_prompt = self.existing_compiled_prompt_path(path)
            existing_references = existing_prompt.with_suffix(".references.json")
            if existing_references not in candidates:
                candidates.append(existing_references)
        except Exception:
            pass
        if self._legacy_project_root is not None:
            raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
            candidates.append((self._legacy_project_root / raw_path).with_suffix(".references.json"))
        for references_path in candidates:
            package = self._read_references_sidecar(references_path)
            if not package.is_empty:
                return package
        return ReferencePackage()

    def load_compiled_runtime_profile(self, path: str | Path) -> RuntimeProfile:
        """Load a confirmed runtime profile sidecar, returning empty when absent."""

        candidates = [self.resolve_compiled_runtime_profile_path(path)]
        try:
            existing_prompt = self.existing_compiled_prompt_path(path)
            existing_runtime_profile = existing_prompt.with_suffix(".runtime-profile.json")
            if existing_runtime_profile not in candidates:
                candidates.append(existing_runtime_profile)
        except Exception:
            pass
        if self._legacy_project_root is not None:
            raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
            candidates.append((self._legacy_project_root / raw_path).with_suffix(".runtime-profile.json"))
        for runtime_profile_path in candidates:
            profile = self._read_runtime_profile_sidecar(runtime_profile_path)
            if profile.is_usable:
                return profile
        return RuntimeProfile()

    def load_compiled_prompt(self, path: str | Path) -> str:
        """Load a confirmed compiled prompt, returning empty text when missing."""

        prompt_path = self.existing_compiled_prompt_path(path)
        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
            primary_path = self.resolve_compiled_prompt_path(path)
            if content and prompt_path != primary_path:
                try:
                    policy = self._read_policy_sidecar(prompt_path.with_suffix(".policy.json"))
                    references = self._read_references_sidecar(prompt_path.with_suffix(".references.json"))
                    runtime_profile = self._read_runtime_profile_sidecar(
                        prompt_path.with_suffix(".runtime-profile.json")
                    )
                    self.save_compiled_prompt(
                        content,
                        path,
                        policy=policy if policy.rules else None,
                        reference_package=references if not references.is_empty else None,
                        runtime_profile=runtime_profile if runtime_profile.is_usable else None,
                    )
                except OSError:
                    pass
            return content
        except OSError as exc:
            from app.logger import get_debug_logger
            get_debug_logger().warning("Compiled prompt \u52a0\u8f7d\u5931\u8d25 (%s): %s", prompt_path, exc)
            return ""

    def existing_compiled_prompt_path(self, path: str | Path) -> Path:
        """Return the readable prompt path, including legacy fallback."""

        prompt_path = self.resolve_compiled_prompt_path(path)
        if prompt_path.exists():
            return prompt_path

        if self._legacy_project_root is not None:
            raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
            project_path = self._legacy_project_root / raw_path
            if project_path.exists():
                return project_path

        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        old_config_path = self._base_dir() / raw_path.name
        if old_config_path.exists():
            return old_config_path

        legacy_path = self._legacy_double_prompt_path(path)
        if legacy_path is not None and legacy_path.exists():
            return legacy_path
        return prompt_path

    def _base_dir(self) -> Path:
        return self._config_dir

    @staticmethod
    def _read_policy_sidecar(path: Path) -> ConstraintPolicy:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ConstraintPolicy()
        try:
            return ConstraintPolicy.from_dict(raw, strict=True)
        except ValueError:
            return ConstraintPolicy()

    @staticmethod
    def _read_references_sidecar(path: Path) -> ReferencePackage:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ReferencePackage()
        try:
            return ReferencePackage.from_dict(raw, strict=True)
        except ValueError:
            return ReferencePackage()

    @staticmethod
    def _read_runtime_profile_sidecar(path: Path) -> RuntimeProfile:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return RuntimeProfile()
        try:
            return RuntimeProfile.from_dict(raw, strict=True)
        except ValueError:
            return RuntimeProfile()

    def _legacy_double_prompt_path(self, path: str | Path) -> Path | None:
        raw_path = Path(path or DEFAULT_COMPILED_PROMPT_PATH)
        if raw_path.is_absolute():
            return None
        parts = raw_path.parts
        if parts and parts[0].lower() == "prompts":
            return self._base_dir() / "prompts" / raw_path
        return None
