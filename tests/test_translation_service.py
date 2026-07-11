"""Tests for translation-service prompt selection."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from app.agent.agent import TranslationAgent
from app.agent.session_store import AgentSessionMeta
from app.feedback.store import FeedbackStore
from app.prompt.base_template import DEFAULT_BASE_PROMPT
from app.prompt.policy import ConstraintPolicy
from app.prompt.storage import PromptStorage
from app.reference_layer import ReferenceEntry, ReferencePackage
from app.settings import AppSettings
from app.translation.client import TranslationError
from app.translation.service import AgentRuntimeMeta, TranslationRequest, TranslationService


class FailingClient:
    """Client test double that always fails."""

    def translate(self, system_prompt: str, source_text: str) -> str:
        raise TranslationError("old request timed out")


class FakeSessionStore:
    """In-memory session store test double."""

    def __init__(self, loaded_messages=None) -> None:
        self.loaded_messages = loaded_messages
        self.profile_messages = None
        self.load_calls = []
        self.save_calls = []
        self.delete_calls = []
        self.profile_load_calls = []
        self.profile_save_calls = []

    def load(self, meta: AgentSessionMeta):
        self.load_calls.append(meta)
        return self.loaded_messages

    def save(self, meta: AgentSessionMeta, messages: list[dict]) -> None:
        self.save_calls.append((meta, list(messages)))

    def delete(self, group_id: int) -> None:
        self.delete_calls.append(group_id)

    def load_profile(self, meta):
        self.profile_load_calls.append(meta)
        return self.profile_messages

    def save_profile(self, meta, messages: list[dict]) -> None:
        self.profile_save_calls.append((meta, list(messages)))


class FakeMemoryAgent:
    """Agent test double that records runtime memory hints."""

    def __init__(self) -> None:
        self.messages = []
        self.text = ""
        self.memory_hints = None
        self.reference_hints = None
        self.recorded = []

    def translate(self, text: str, memory_hints=None, *, reference_hints=None, record=True, cancellation_check=None) -> str:
        self.text = text
        self.memory_hints = memory_hints
        self.reference_hints = reference_hints
        return "the national college entrance exam starts soon"

    def record_translation(self, source_text: str, translated_text: str) -> None:
        self.recorded.append((source_text, translated_text))


class TranslationServicePromptTests(unittest.TestCase):
    """Verify runtime translation prompts prefer confirmed compiled prompts."""

    def test_current_prompt_uses_compiled_prompt_file_when_available(self) -> None:
        compiled_content = f"{DEFAULT_BASE_PROMPT}\n\nCONFIRMED USER CONSTRAINT"
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(compiled_content, encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            service = TranslationService(settings)

            prompt = service._current_prompt("English", "中文")
            service.shutdown()

        self.assertIn("CONFIRMED USER CONSTRAINT", prompt)
        self.assertIn("Translate from English to 中文.", prompt)
        # Compact joins lines with spaces; DEFAULT_BASE_PROMPT content is present
        self.assertIn("instant translation assistant", prompt)

    def test_reference_entries_ignore_raw_compiled_prompt_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(
                "| Chinese Term | Output |\n"
                "|--------------|--------|\n"
                "| 接口 | [いんたあふぇえす] |\n",
                encoding="utf-8",
            )
            reference_path = Path(tmp) / "extra.md"
            reference_path.write_text("- se_server: 服务器 -> [さあばあ]", encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            settings.prompt.knowledge_reference_paths = [str(reference_path)]
            service = TranslationService(settings)

            entries = service._current_reference_entries()
            definition = service._resolve_agent_definition("中文", "日本語")
            service.shutdown()

        by_source = {entry.source: entry.target for entry in entries}
        self.assertNotIn("接口", by_source)
        self.assertEqual(by_source["服务器"], "[さあばあ]")
        self.assertEqual(definition.reference_entries, entries)
        self.assertTrue(definition.runtime_meta.prompt_hash)

    def test_reference_entries_prefer_compiled_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            package = ReferencePackage.from_texts(
                ["- se_sidecar: 侧边栏 -> [さいどばあ]; risk: medium"]
            )
            storage.save_compiled_prompt(
                "compiled prompt without raw glossary",
                "prompts/compiled-prompt.md",
                reference_package=package,
            )
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            service = TranslationService(settings)
            service._settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"

            with patch("app.translation.service.PromptStorage", lambda: storage):
                entries = service._current_reference_entries()
                hints = service._current_reference_package().runtime_hints("侧边栏没有刷新")
            service.shutdown()

        self.assertEqual(entries[0].source, "侧边栏")
        self.assertTrue(any("风险=medium" in hint for hint in hints))

    def test_reference_context_reuses_protected_plan_matches_for_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            package = ReferencePackage(
                entries=(
                    ReferenceEntry("request", "请求", "[りくえすと]", risk="medium", note="generic request"),
                    ReferenceEntry("async_request", "异步请求", "[あしんくろなすりくえすと]", risk="medium", note="async request"),
                )
            )
            storage.save_compiled_prompt(
                "compiled prompt",
                "prompts/compiled-prompt.md",
                reference_package=package,
            )
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            service = TranslationService(settings)

            with patch("app.translation.service.PromptStorage", lambda: storage):
                context = service._reference_context_for_request(
                    TranslationRequest(group_id=1, request_id=1, ocr_text="请改成异步请求")
                )
            service.shutdown()

        self.assertEqual([entry.id for entry in context.plan.matched_entries], ["async_request"])
        self.assertTrue(any("async request" in hint for hint in context.hints))
        self.assertFalse(any("generic request" in hint for hint in context.hints))

    def test_current_prompt_falls_back_to_fixed_template_without_compiled_file(self) -> None:
        settings = AppSettings()
        settings.prompt.compiled_prompt_path = ""
        service = TranslationService(settings)

        prompt = service._current_prompt("日本語", "English")
        service.shutdown()

        self.assertIn(DEFAULT_BASE_PROMPT, prompt)
        self.assertIn("Translate from 日本語 to English.", prompt)

    def test_current_prompt_ignores_japanese_compiled_prompt_for_english_target(self) -> None:
        compiled_content = """
# Instant Translate Compiled Prompt

## User Constraint Layer
Chinese to Japanese output must be hiragana only.

## AI Optimization Layer
Use natural Japanese examples.
""".strip()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(compiled_content, encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            service = TranslationService(settings)

            prompt = service._current_prompt("中文", "English")
            service.shutdown()

        self.assertIn(DEFAULT_BASE_PROMPT, prompt)
        self.assertNotIn("hiragana only", prompt)
        self.assertNotIn("natural Japanese", prompt)
        self.assertIn("Translate from 中文 to English.", prompt)

    def test_current_prompt_strips_legacy_ai_optimization_glossary(self) -> None:
        compiled_content = """
# Instant Translate Compiled Prompt

## User Constraint Layer
如果中文翻译为日语时只能由平假名构成，软件工程术语使用[]包裹。

## AI Optimization Layer
Supplemental Rules
- First produce a natural Japanese translation.
- Wrap every software engineering term in brackets.

Glossary Entries (Software Engineering Terms)
Input the Chinese term; output the bracketed hiragana equivalent. These are mandatory replacements if the source contains them.

| Chinese Term | Output (hiragana in brackets) |
|--------------|-------------------------------|
| 软件 | [そふとうぇあ] |
| 数据库 | [でーたべーす] |

For any other software engineering term not listed, generate its hiragana reading and enclose in brackets.

Style Guidance
- Keep Japanese word order natural.

## Runtime Direction
For every request, follow the source and target languages appended by the app.
""".strip()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(compiled_content, encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            service = TranslationService(settings)

            prompt = service._current_prompt("中文", "日本語")
            service.shutdown()

        self.assertIn("First produce a natural Japanese translation", prompt)
        self.assertIn("Keep Japanese word order natural", prompt)
        self.assertNotIn("Wrap every software engineering term", prompt)
        self.assertNotIn("Glossary Entries", prompt)
        self.assertNotIn("mandatory replacements", prompt)
        self.assertNotIn("| 软件 |", prompt)
        self.assertNotIn("[そふとうぇあ]", prompt)
        self.assertNotIn("generate its hiragana reading", prompt)

    def test_current_prompt_clarifies_hiragana_as_semantic_translation(self) -> None:
        compiled_content = """
# Instant Translate Compiled Prompt

## User Constraint Layer
如果中文翻译为日语时，翻译的日语句子只能由平假名构成

## AI Optimization Layer
All kanji must be converted to their corresponding hiragana readings without exception.
- Convert all Chinese characters to hiragana equivalents.
- Maintain original meaning while adhering to the hiragana-only constraint.
**Examples**:
- 中国語 (ちゅうごくご) → ちゅうごくご

## Runtime Direction
For every request, follow the source and target languages appended by the app.
""".strip()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "compiled.md"
            prompt_path.write_text(compiled_content, encoding="utf-8")
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = str(prompt_path)
            service = TranslationService(settings)

            prompt = service._current_prompt("中文", "日本語")
            service.shutdown()

        self.assertIn("先把源文本按语义翻译成自然日语", prompt)
        self.assertIn("不能仅因字符是“高”“考”就输出“こうこう”", prompt)
        self.assertIn("自然日语质量规则", prompt)
        self.assertIn("不要逐词硬译中文量词", prompt)
        self.assertIn("不要译成“いちにんのおんな”", prompt)
        self.assertIn("单元测试用例", prompt)
        self.assertIn("たんたいてすとけーす", prompt)
        self.assertIn("不要机械翻译成“いえ”", prompt)
        self.assertNotIn("Convert all Chinese characters to hiragana equivalents", prompt)
        self.assertNotIn("corresponding hiragana readings", prompt)
        self.assertNotIn("中国語 (ちゅうごくご)", prompt)

    def test_stale_translation_error_does_not_notify_result_callback(self) -> None:
        service = TranslationService(AppSettings())
        ctx = service._ensure_group(1)
        ctx.current_request_id = 2
        service._build_client = lambda: FailingClient()
        results = []

        service._execute(
            TranslationRequest(
                group_id=1,
                request_id=1,
                ocr_text="old text",
                source_language="English",
                target_language="中文",
            ),
            results.append,
        )
        service.shutdown()

        self.assertEqual(results, [])

    def test_invalidate_group_requests_marks_existing_request_stale(self) -> None:
        service = TranslationService(AppSettings())
        request = TranslationRequest(
            group_id=1,
            request_id=1,
            ocr_text="old text",
            source_language="English",
            target_language="中文",
        )
        ctx = service._ensure_group(1)
        ctx.current_request_id = 1

        service.invalidate_group_requests(1, reason="ocr changed")

        self.assertTrue(service._is_stale_request(request, ctx))
        service.shutdown()

    def test_invalidated_text_can_be_submitted_again(self) -> None:
        service = TranslationService(AppSettings())
        pending = Future()
        with patch.object(service._executor, "submit", return_value=pending):
            first = service.request_translation(1, "same text", lambda result: None)
            service.invalidate_group_requests(1, reason="edit mode enabled")
            second = service.request_translation(1, "same text", lambda result: None)

        service.shutdown()

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertGreater(second.request_id, first.request_id)

    def test_pending_duplicate_is_skipped_but_committed_duplicate_is_also_skipped(self) -> None:
        service = TranslationService(AppSettings())
        pending = Future()
        with patch.object(service._executor, "submit", return_value=pending):
            first = service.request_translation(1, "same text", lambda result: None)
            duplicate_pending = service.request_translation(1, "same text", lambda result: None)
            ctx = service._ensure_group(1)
            ctx.pending_text = ""
            ctx.last_committed_text = "same text"
            duplicate_committed = service.request_translation(1, "same text", lambda result: None)

        service.shutdown()

        self.assertIsNotNone(first)
        self.assertIsNone(duplicate_pending)
        self.assertIsNone(duplicate_committed)

    def test_current_translation_error_clears_pending_text_for_retry(self) -> None:
        service = TranslationService(AppSettings())

        class FailingAgent:
            messages = []

            def translate(self, text, memory_hints=None, *, reference_hints=None, record=True, cancellation_check=None):
                raise TranslationError("network failed")

        service._ensure_agent = lambda group_id, source, target: FailingAgent()
        request = TranslationRequest(group_id=1, request_id=1, ocr_text="retry me")
        ctx = service._ensure_group(1)
        ctx.current_request_id = 1
        ctx.pending_text = "retry me"
        results = []

        service._execute(request, results.append)
        service.shutdown()

        self.assertEqual(ctx.pending_text, "")
        self.assertEqual(results[0].error, "network failed")

    def test_api_backoff_opens_after_repeated_transient_failures(self) -> None:
        service = TranslationService(AppSettings())
        service.API_BACKOFF_FAILURE_THRESHOLD = 2
        service.API_BACKOFF_COOLDOWN_SECONDS = 10.0
        now = [100.0]
        service._time_fn = lambda: now[0]

        service._record_api_outcome(False, "Translation request timed out after 30s.")
        service._raise_if_api_backoff_active()
        service._record_api_outcome(False, "API returned an empty translation.")

        with self.assertRaisesRegex(TranslationError, "retry in 10s"):
            service._raise_if_api_backoff_active()

        now[0] = 111.0
        service._raise_if_api_backoff_active()
        self.assertEqual(service._api_failure_count, 0)
        service.shutdown()

    def test_api_backoff_ignores_local_validation_failures(self) -> None:
        service = TranslationService(AppSettings())
        service.API_BACKOFF_FAILURE_THRESHOLD = 1
        service._record_api_outcome(False, "Translation failed local validation: allowed_characters")

        service._raise_if_api_backoff_active()
        self.assertEqual(service._api_failure_count, 0)
        service.shutdown()

    def test_stale_request_after_agent_prepare_never_calls_flash(self) -> None:
        service = TranslationService(AppSettings())
        fake_agent = FakeMemoryAgent()

        def prepare_and_invalidate(group_id, source, target):
            service.invalidate_group_requests(group_id, reason="screen changed during prepare")
            return fake_agent

        service._ensure_agent = prepare_and_invalidate
        request = TranslationRequest(group_id=1, request_id=1, ocr_text="old text")
        ctx = service._ensure_group(1)
        ctx.current_request_id = 1
        ctx.pending_text = "old text"

        service._execute(request, lambda result: None)
        service.shutdown()

        self.assertEqual(fake_agent.text, "")
        self.assertEqual(fake_agent.recorded, [])

    def test_execute_injects_confirmed_feedback_memory_when_trigger_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feedback_store = FeedbackStore(tmp)
            record = feedback_store.add_feedback(
                group_id=1,
                source_language="English",
                target_language="中文",
                ocr_text="The gaokao starts soon",
                translation_text="高中考试很快开始",
            )
            feedback_store.approve_feedback(
                record.id,
                trigger="gaokao",
                rule="gaokao 应译为中国高考/大学入学考试，不要译成普通高中考试。",
                preferred_translation="高考很快开始",
            )
            service = TranslationService(AppSettings(), feedback_store=feedback_store)
            fake_agent = FakeMemoryAgent()
            service._ensure_agent = lambda group_id, source, target: fake_agent
            request = TranslationRequest(
                group_id=1,
                request_id=1,
                ocr_text="The gaokao starts soon",
                source_language="English",
                target_language="中文",
            )
            service._ensure_group(1).current_request_id = 1
            results = []

            service._execute(request, results.append)
            service.shutdown()

            self.assertEqual(results[0].text, "the national college entrance exam starts soon")
            self.assertEqual(fake_agent.text, "The gaokao starts soon")
            self.assertIsNotNone(fake_agent.memory_hints)
            self.assertIn("gaokao 应译为", fake_agent.memory_hints[0])

    def test_execute_injects_compiled_reference_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = PromptStorage(config_dir=Path(tmp))
            package = ReferencePackage.from_texts(
                [
                    "- se_sidebar: 侧边栏 -> [さいどばあ]; risk: medium; note: UI navigation area\n"
                    "style: keep Japanese word order natural"
                ]
            )
            storage.save_compiled_prompt(
                "compiled prompt",
                "prompts/compiled-prompt.md",
                reference_package=package,
            )
            settings = AppSettings()
            settings.prompt.compiled_prompt_path = "prompts/compiled-prompt.md"
            service = TranslationService(settings)
            fake_agent = FakeMemoryAgent()
            service._ensure_agent = lambda group_id, source, target: fake_agent
            request = TranslationRequest(
                group_id=1,
                request_id=1,
                ocr_text="侧边栏没有刷新",
                source_language="中文",
                target_language="日本語",
            )
            service._ensure_group(1).current_request_id = 1
            results = []

            with patch("app.translation.service.PromptStorage", lambda: storage):
                service._execute(request, results.append)
            service.shutdown()

        self.assertEqual(results[0].text, "the national college entrance exam starts soon")
        self.assertIsNotNone(fake_agent.reference_hints)
        self.assertTrue(any("引用层风格" in hint for hint in fake_agent.reference_hints))
        self.assertTrue(any("侧边栏" in hint and "风险=medium" in hint for hint in fake_agent.reference_hints))


    def test_similarity_does_not_hide_meaning_changing_insertions(self) -> None:
        self.assertFalse(
            TranslationService._is_similar("测试没有通过", "测试通过")
        )
        self.assertFalse(
            TranslationService._is_similar(
                "请不要把窗口放在屏幕下方",
                "请把窗口放在屏幕下方",
            )
        )

    def test_stale_success_is_not_recorded_or_saved(self) -> None:
        store = FakeSessionStore()
        service = TranslationService(AppSettings(), session_store=store)
        fake_agent = FakeMemoryAgent()
        service._ensure_agent = lambda group_id, source, target: fake_agent
        request = TranslationRequest(
            group_id=1,
            request_id=1,
            ocr_text="old text",
            source_language="English",
            target_language="中文",
        )
        service._ensure_group(1).current_request_id = 2
        results = []

        service._execute(request, results.append)
        service.shutdown()

        self.assertEqual(results, [])
        self.assertEqual(fake_agent.recorded, [])
        self.assertEqual(store.save_calls, [])

    def test_stale_request_after_flash_skips_thinking_retry(self) -> None:
        """Flash returns invalid result, request goes stale, retry is skipped."""
        from app.agent.agent import StaleRequestAborted

        store = FakeSessionStore()
        service = TranslationService(AppSettings(), session_store=store)
        results = []

        class AgentThatGoesStaleAfterFlash:
            messages = []

            def __init__(self):
                self.recorded = []
                self.retry_called = False

            def translate(self, text, memory_hints=None, *, reference_hints=None, record=True, cancellation_check=None):
                # Simulate Flash already returned with a validation failure.
                # A new OCR request now supersedes this one.
                service.invalidate_group_requests(1, reason="new OCR")
                # Before thinking retry, the cancellation check runs.
                if cancellation_check is not None and cancellation_check():
                    raise StaleRequestAborted("stale after flash")
                self.retry_called = True
                return "translated"

            def record_translation(self, source_text, translated_text):
                self.recorded.append((source_text, translated_text))

        agent = AgentThatGoesStaleAfterFlash()
        service._ensure_agent = lambda group_id, source, target: agent
        request = TranslationRequest(
            group_id=1,
            request_id=1,
            ocr_text="old text",
            source_language="English",
            target_language="中文",
        )
        ctx = service._ensure_group(1)
        ctx.current_request_id = 1
        ctx.pending_text = "old text"

        service._execute(request, results.append)
        service.shutdown()

        self.assertEqual(results, [])
        self.assertEqual(agent.recorded, [])
        self.assertFalse(agent.retry_called)
        self.assertEqual(store.save_calls, [])

    def test_reset_group_clears_state_and_increments_request_id(self) -> None:
        """reset_group clears dedup state and increments request id (no ABA)."""
        service = TranslationService(AppSettings())
        ctx = service._ensure_group(1)
        ctx.current_request_id = 5
        ctx.last_committed_text = "committed"
        ctx.last_translation = "訳文"
        ctx.pending_text = "pending"

        service.reset_group(1)

        ctx_after = service._ensure_group(1)
        self.assertEqual(ctx_after.last_committed_text, "")
        self.assertEqual(ctx_after.last_translation, "")
        self.assertEqual(ctx_after.pending_text, "")
        self.assertEqual(ctx_after.current_request_id, 6)

        # Old request is now stale
        old_request = TranslationRequest(group_id=1, request_id=5, ocr_text="old")
        self.assertTrue(service._is_stale_request(old_request, ctx_after))

        # New request gets ID strictly greater than old
        with patch.object(service._executor, "submit", return_value=Future()):
            new_request = service.request_translation(1, "new text", lambda r: None)
        service.shutdown()

        self.assertIsNotNone(new_request)
        self.assertGreater(new_request.request_id, 5)

    def test_reset_group_prevents_aba_request_id_reuse(self) -> None:
        """ABA: reset_group must not let a stale worker reuse a request id."""
        store = FakeSessionStore()
        service = TranslationService(AppSettings(), session_store=store)
        results = []

        class SimpleAgent:
            messages = []

            def __init__(self):
                self.recorded = []

            def translate(self, text, memory_hints=None, *, reference_hints=None, record=True, cancellation_check=None):
                return f"translated: {text}"

            def record_translation(self, source_text, translated_text):
                self.recorded.append((source_text, translated_text))

        agent = SimpleAgent()
        service._ensure_agent = lambda group_id, source, target: agent

        # Step 1: Submit old request (request_id=1), prevent execution
        with patch.object(service._executor, "submit", return_value=Future()):
            old_request = service.request_translation(1, "old text", results.append)
        self.assertIsNotNone(old_request)
        self.assertEqual(old_request.request_id, 1)

        # Step 2: reset_group clears state and increments request_id
        service.reset_group(1)

        # Step 3: Submit new request
        with patch.object(service._executor, "submit", return_value=Future()):
            new_request = service.request_translation(1, "new text", results.append)
        self.assertIsNotNone(new_request)

        # Critical ABA assertion: new request ID must be strictly greater
        self.assertGreater(new_request.request_id, old_request.request_id)

        # Step 4: Old request worker now completes via _execute.
        # In the old pop-based implementation, reset_group would have destroyed
        # the GroupContext, current_request_id would restart from 0, and
        # request_translation would assign request_id=1 again — making
        # old_request.request_id == new_request.request_id (ABA).
        service._execute(old_request, results.append)

        # Step 5: Old request must not produce callback, record, or save
        self.assertEqual(results, [])
        self.assertEqual(agent.recorded, [])
        self.assertEqual(store.save_calls, [])

        service.shutdown()


class TranslationServiceAgentPerGroupTests(unittest.TestCase):
    """Verify each group gets its own independent Agent instance."""

    def test_different_groups_create_different_agents(self) -> None:
        """Two groups should have separate Agent instances."""
        service = TranslationService(AppSettings())
        
        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None):
            agent1 = service._ensure_agent(1, "English", "中文")
            agent2 = service._ensure_agent(2, "日本語", "English")
        
        service.shutdown()
        
        self.assertIsNot(agent1, agent2, "Different groups should have different agents")
        self.assertEqual(len(service._agents), 2)
        self.assertIn(1, service._agents)
        self.assertIn(2, service._agents)

    def test_same_group_reuses_same_agent(self) -> None:
        """Multiple calls for same group should reuse the same Agent."""
        service = TranslationService(AppSettings())
        
        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None):
            agent1 = service._ensure_agent(1, "English", "中文")
            agent2 = service._ensure_agent(1, "English", "中文")
        
        service.shutdown()
        
        self.assertIs(agent1, agent2, "Same group should reuse same agent")
        self.assertEqual(len(service._agents), 1)

    def test_reset_agent_only_clears_specified_group(self) -> None:
        """reset_agent(group_id) should only clear that group's agent."""
        service = TranslationService(AppSettings())
        
        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None):
            agent1 = service._ensure_agent(1, "English", "中文")
            agent2 = service._ensure_agent(2, "日本語", "English")
        
        service.reset_agent(1)
        
        self.assertNotIn(1, service._agents, "Group 1 agent should be cleared")
        self.assertIn(2, service._agents, "Group 2 agent should remain")
        self.assertIs(service._agents[2], agent2, "Group 2 agent should be unchanged")
        
        service.shutdown()

    def test_reset_agent_can_remove_persisted_group_session(self) -> None:
        store = FakeSessionStore()
        service = TranslationService(AppSettings(), session_store=store)

        service.reset_agent(3, delete_persisted=True)

        self.assertEqual(store.delete_calls, [3])
        service.shutdown()

    def test_agent_is_rebuilt_when_model_configuration_changes(self) -> None:
        settings = AppSettings()
        settings.ai.fast_model = "fast-v1"
        settings.ai.thinking_model = "pro-v1"
        service = TranslationService(settings, session_store=FakeSessionStore())

        with patch.object(TranslationAgent, "digest_rules"), patch.object(
            TranslationAgent, "__init__", return_value=None
        ):
            first = service._ensure_agent(1, "English", "中文")
            settings.ai.fast_model = "fast-v2"
            second = service._ensure_agent(1, "English", "中文")

        self.assertIsNot(first, second)
        self.assertEqual(service._agent_meta[1].fast_model, "fast-v2")
        service.shutdown()

    def test_background_prepare_uses_saved_profile_without_pro_call(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        store = FakeSessionStore()
        store.profile_messages = messages
        service = TranslationService(settings, session_store=store)

        with patch.object(TranslationAgent, "digest_rules") as digest_mock:
            future = service.prepare_agent_profile("中文", "日本語")
            self.assertIsNotNone(future)
            self.assertEqual(future.result(timeout=2), messages)

        service.shutdown()
        digest_mock.assert_not_called()
        self.assertEqual(len(store.profile_load_calls), 1)

    def test_group_session_is_seeded_from_prepared_profile(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        store = FakeSessionStore()
        store.profile_messages = messages
        service = TranslationService(settings, session_store=store)
        future = service.prepare_agent_profile("中文", "日本語")
        assert future is not None
        future.result(timeout=2)

        with patch.object(TranslationAgent, "digest_rules") as digest_mock:
            agent = service._ensure_agent(1, "中文", "日本語")

        service.shutdown()
        digest_mock.assert_not_called()
        self.assertEqual(agent.messages, messages)
        self.assertEqual(store.save_calls[0][1], messages)

    def test_ensure_agent_passes_fast_and_thinking_model_configs(self) -> None:
        """Agent should receive separate fast and thinking model configs."""
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        service = TranslationService(settings)

        with patch.object(TranslationAgent, 'digest_rules'), \
             patch.object(TranslationAgent, '__init__', return_value=None) as init_mock:
            service._ensure_agent(1, "中文", "日本語")

        service.shutdown()

        fast_config, thinking_config = init_mock.call_args.args
        self.assertEqual(fast_config.model, "deepseek-v4-flash")
        self.assertEqual(thinking_config.model, "deepseek-v4-pro")
        self.assertEqual(fast_config.base_url, "https://api.example.test/v1")
        self.assertEqual(thinking_config.api_key, "key")

    def test_ensure_agent_restores_saved_session_without_digest(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        store = FakeSessionStore(loaded_messages=messages)
        service = TranslationService(settings, session_store=store)

        with patch.object(TranslationAgent, 'digest_rules') as digest_mock, \
             patch.object(TranslationAgent, 'restore_messages') as restore_mock:
            service._ensure_agent(1, "中文", "日本語")

        service.shutdown()

        digest_mock.assert_not_called()
        restore_mock.assert_called_once_with(messages, policy=ConstraintPolicy())
        self.assertEqual(store.load_calls[0].fast_model, "deepseek-v4-flash")
        self.assertEqual(store.load_calls[0].thinking_model, "deepseek-v4-pro")

    def test_ensure_agent_saves_session_after_new_digest(self) -> None:
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "confirmed"},
        ]
        settings = AppSettings()
        settings.ai.base_url = "https://api.example.test/v1"
        settings.ai.api_key = "key"
        settings.ai.fast_model = "deepseek-v4-flash"
        settings.ai.thinking_model = "deepseek-v4-pro"
        store = FakeSessionStore()
        service = TranslationService(settings, session_store=store)

        def fake_digest(agent, prompt, policy=None):
            agent.messages = messages

        with patch.object(TranslationAgent, 'digest_rules', autospec=True, side_effect=fake_digest):
            service._ensure_agent(1, "中文", "日本語")

        service.shutdown()

        self.assertEqual(len(store.save_calls), 1)
        self.assertEqual(store.save_calls[0][1], messages)

    def test_save_agent_session_persists_only_safe_profile_context(self) -> None:
        store = FakeSessionStore()
        service = TranslationService(AppSettings(), session_store=store)

        class AgentWithUntrustedHistory:
            messages = [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "confirm"},
                {"role": "assistant", "content": "confirmed"},
                {"role": "user", "content": "客户说发票多开了26.5元"},
                {"role": "assistant", "content": "おきゃくさまが  いんしん"},
            ]

            def persistable_messages(self):
                return [dict(message) for message in self.messages[:3]]

        service._agent_meta[1] = AgentRuntimeMeta(
            source_language="中文",
            target_language="日本語",
            prompt_hash="hash",
            fast_model="deepseek-v4-flash",
            thinking_model="deepseek-v4-pro",
        )

        service._save_agent_session(1, AgentWithUntrustedHistory())
        service.shutdown()

        self.assertEqual(len(store.save_calls), 1)
        saved_messages = store.save_calls[0][1]
        self.assertEqual(len(saved_messages), 3)
        serialized = "\n".join(message["content"] for message in saved_messages)
        self.assertNotIn("发票", serialized)
        self.assertNotIn("いんしん", serialized)

    def test_scoped_policy_does_not_apply_to_another_language_pair(self) -> None:
        settings = AppSettings()
        settings.prompt.constraints_text = "如果中文翻译为日语时只能由平假名构成。"
        service = TranslationService(settings, session_store=FakeSessionStore())

        policy = service._current_policy("English", "中文")
        service.shutdown()

        self.assertEqual(policy.rules, ())

    def test_different_groups_translate_concurrently_with_api_limit(self) -> None:
        service = TranslationService(
            AppSettings(),
            max_workers=3,
            max_concurrent_api_calls=2,
        )
        gate = threading.Event()
        two_active = threading.Event()
        all_done = threading.Event()
        counter_lock = threading.Lock()
        state = {"active": 0, "max_active": 0}
        results = []

        class BlockingAgent:
            messages = []

            def translate(self, text: str, memory_hints=None, *, reference_hints=None, record=True, cancellation_check=None) -> str:
                with counter_lock:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    if state["active"] >= 2:
                        two_active.set()
                try:
                    if not gate.wait(timeout=2.0):
                        raise TranslationError("test gate timed out")
                    return f"translated: {text}"
                finally:
                    with counter_lock:
                        state["active"] -= 1

            def record_translation(self, source_text: str, translated_text: str) -> None:
                return None

        fake_agent = BlockingAgent()
        service._ensure_agent = lambda group_id, source, target: fake_agent

        def on_result(result) -> None:
            with counter_lock:
                results.append(result)
                if len(results) == 3:
                    all_done.set()

        try:
            for group_id in (1, 2, 3):
                service.request_translation(
                    group_id,
                    f"text {group_id}",
                    on_result,
                )

            self.assertTrue(two_active.wait(timeout=1.0), "different groups did not overlap")
            self.assertEqual(state["max_active"], 2)
            gate.set()
            self.assertTrue(all_done.wait(timeout=2.0))
            self.assertEqual(len(results), 3)
            self.assertTrue(all(result.error is None for result in results))
        finally:
            gate.set()
            service.shutdown()


if __name__ == "__main__":
    unittest.main()
