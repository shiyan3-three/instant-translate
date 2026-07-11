"""Tests for production reference/glossary injection."""

from __future__ import annotations

import unittest

from app.reference_layer import (
    ReferenceEntry,
    ReferencePackage,
    ReferenceStore,
    extract_ai_optimization_reference_candidates,
    format_reference_candidate_markdown,
    parse_reference_entries,
    parse_reference_guidance,
    parse_reference_markdown,
)


class ReferenceLayerTests(unittest.TestCase):
    def test_parses_markdown_table_entries(self) -> None:
        entries = parse_reference_entries(
            "| Chinese Term | Output |\n"
            "|--------------|--------|\n"
            "| 接口 | [いんたあふぇえす] |\n"
            "| 服务器 | [さあばあ] |\n"
        )

        by_source = {entry.source: entry.target for entry in entries}
        self.assertEqual(by_source["接口"], "[いんたあふぇえす]")
        self.assertEqual(by_source["服务器"], "[さあばあ]")

    def test_parses_arrow_entries_with_aliases(self) -> None:
        entries = parse_reference_entries(
            "- se_interface: 接口 -> [いんたあふぇえす]; aliases: API接口, 网关接口; risk: medium"
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "se_interface")
        self.assertEqual(entries[0].source, "接口")
        self.assertEqual(entries[0].target, "[いんたあふぇえす]")
        self.assertIn("API接口", entries[0].aliases)
        self.assertEqual(entries[0].risk, "medium")

    def test_parses_style_and_risk_guidance(self) -> None:
        style, risk = parse_reference_guidance(
            "## Style Guide\n"
            "- keep Japanese word order natural\n"
            "risk: 对象有软件术语和普通名词两种含义，按上下文判断\n"
        )

        self.assertIn("keep Japanese word order natural", style)
        self.assertIn("对象有软件术语和普通名词两种含义，按上下文判断", risk)

    def test_section_aware_parser_ignores_arrows_outside_reference_sections(self) -> None:
        package = parse_reference_markdown(
            "## Glossary\n"
            "接口 -> [いんたあふぇえす]\n\n"
            "## Style\n"
            "- Keep A -> B notation readable when it is part of prose.\n\n"
            "## Risk\n"
            "- 对象 -> 这里是说明文字，不是术语映射\n\n"
            "## Examples\n"
            "按钮 -> [ぼたん]\n"
        )

        by_source = {entry.source: entry.target for entry in package.entries}
        self.assertEqual(by_source, {"接口": "[いんたあふぇえす]"})
        self.assertTrue(any("A -> B" in item for item in package.style_guidance))
        self.assertTrue(any("对象 ->" in item for item in package.risk_notes))

    def test_reference_package_round_trips_and_emits_runtime_hints(self) -> None:
        package = ReferencePackage.from_texts(
            [
                "- se_object: 对象 -> [おぶじぇくと]; risk: high; note: only when it means software object\n"
                "style: keep Japanese word order natural\n"
                "risk: 对象有软件术语和普通名词两种含义，按上下文判断"
            ]
        )

        loaded = ReferencePackage.from_dict(package.to_dict(), strict=True)
        hints = loaded.runtime_hints("这个对象为空")

        self.assertEqual(loaded.entries[0].source, "对象")
        self.assertIn("keep Japanese word order natural", loaded.style_guidance)
        self.assertTrue(any("引用层风格" in hint for hint in hints))
        self.assertTrue(any("对象" in hint and "风险=high" in hint for hint in hints))
        self.assertTrue(any("按上下文判断" in hint for hint in hints))

    def test_runtime_hints_can_reuse_protected_reference_plan_matches(self) -> None:
        package = ReferencePackage(
            entries=(
                ReferenceEntry("request", "请求", "[りくえすと]", risk="medium", note="generic request"),
                ReferenceEntry("async_request", "异步请求", "[あしんくろなすりくえすと]", risk="medium", note="async request"),
            )
        )
        plan = ReferenceStore(package.entries).protect("请改成异步请求")

        hints = package.runtime_hints(
            "请改成异步请求",
            matched_entries=plan.matched_entries,
        )

        self.assertEqual([entry.id for entry in plan.matched_entries], ["async_request"])
        self.assertTrue(any("async request" in hint for hint in hints))
        self.assertFalse(any("generic request" in hint for hint in hints))

    def test_risk_note_does_not_match_unrelated_two_character_overlap(self) -> None:
        package = ReferencePackage.from_texts(
            ["risk: 对象有软件术语和普通名词两种含义，按上下文判断"]
        )

        matched = package.runtime_hints("这个对象为空")
        unrelated = package.runtime_hints("这个软件已经启动")

        self.assertTrue(any("按上下文判断" in hint for hint in matched))
        self.assertFalse(any("按上下文判断" in hint for hint in unrelated))

    def test_risk_note_supports_explicit_triggers(self) -> None:
        package = ReferencePackage.from_texts(
            ["risk: triggers: 侧边栏, sidebar; UI navigation area may be confused with page content"]
        )

        hints = package.runtime_hints("侧边栏没有刷新")

        self.assertTrue(any("UI navigation" in hint for hint in hints))

    def test_extracts_ai_optimization_reference_candidates_without_trusting_them(self) -> None:
        entries = extract_ai_optimization_reference_candidates(
            "# Instant Translate Compiled Prompt\n\n"
            "## User Constraint Layer\n"
            "软件 -> [そふとうぇあ]\n\n"
            "## AI Optimization Layer\n"
            "| Source | Target |\n"
            "|--------|--------|\n"
            "| 数据库 | [でえたべえす] |\n\n"
            "## Knowledge Reference Layer\n"
            "No knowledge references.\n"
        )
        markdown = format_reference_candidate_markdown(entries)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].source, "数据库")
        self.assertIn("Review them manually", markdown)
        self.assertIn("| 数据库 | [でえたべえす] |", markdown)

    def test_protects_longest_non_overlapping_exact_matches(self) -> None:
        store = ReferenceStore(
            [
                ReferenceEntry("request", "请求", "[りくえすと]"),
                ReferenceEntry("async_request", "异步请求", "[あしんくろなすりくえすと]"),
                ReferenceEntry("script", "脚本", "[すくりぷと]"),
            ]
        )

        plan = store.protect("请把脚本改成异步请求")

        self.assertEqual(plan.source, "请把⟦REF_0⟧改成⟦REF_1⟧")
        self.assertEqual(plan.replacements["⟦REF_0⟧"], "[すくりぷと]")
        self.assertEqual(plan.replacements["⟦REF_1⟧"], "[あしんくろなすりくえすと]")
        self.assertNotIn("[りくえすと]", plan.replacements.values())

    def test_ambiguous_surface_is_not_locally_protected(self) -> None:
        store = ReferenceStore(
            [
                ReferenceEntry("object_general", "对象", "[たいしょう]"),
                ReferenceEntry("object_code", "对象", "[おぶじぇくと]"),
            ]
        )

        plan = store.protect("这个对象为空")

        self.assertEqual(plan.source, "这个对象为空")
        self.assertFalse(plan.has_placeholders)

    def test_validates_placeholder_counts_and_restores_targets(self) -> None:
        plan = ReferenceStore([ReferenceEntry("server", "服务器", "[さあばあ]")]).protect(
            "服务器没有响应"
        )

        self.assertEqual(plan.validate_raw_output("⟦REF_0⟧  が  おうとう  しない"), "")
        self.assertIn("got 0", plan.validate_raw_output("さあばあ  が  おうとう  しない"))
        self.assertEqual(
            plan.restore("⟦REF_0⟧  が  おうとう  しない"),
            "[さあばあ]  が  おうとう  しない",
        )


if __name__ == "__main__":
    unittest.main()
