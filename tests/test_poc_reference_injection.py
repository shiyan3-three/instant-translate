"""Offline tests for the third reference-injection POC harness."""

from __future__ import annotations

import unittest

from poc.poc_reference_injection import (
    GlossaryEntry,
    PocCase,
    PocDataset,
    build_request,
    evaluate_response,
    protect_glossary_terms,
    restore_placeholders,
    validate_dataset,
)


def _dataset() -> PocDataset:
    entries = (
        GlossaryEntry("software", "软件", "[そふとうぇあ]"),
        GlossaryEntry("develop", "开发", "[かいはつ]"),
    )
    return PocDataset(
        version=1,
        status="draft",
        name="test",
        base_system_prompt="Translate all source text.",
        global_rules=("Return only the translation.",),
        style_rules=(),
        forbidden_outputs=("翻訳：",),
        examples=({"source": "软件", "target": "[そふとうぇあ]"},),
        glossary_sets={"20": entries, "200": entries},
        cases=(),
    )


class PlaceholderTests(unittest.TestCase):
    def test_longest_non_overlapping_term_is_protected_and_restored(self) -> None:
        entries = (
            GlossaryEntry("database", "数据库", "[でえたべえす]"),
            GlossaryEntry("data", "数据", "[でえた]"),
        )

        plan = protect_glossary_terms("备份数据库数据", entries)

        self.assertEqual(plan.source, "备份⟦REF_0⟧⟦REF_1⟧")
        self.assertEqual(
            restore_placeholders(plan.source, plan.replacements),
            "备份[でえたべえす][でえた]",
        )

    def test_ambiguous_surface_is_not_hidden_by_a_placeholder(self) -> None:
        entries = (
            GlossaryEntry("session_web", "Session", "会话"),
            GlossaryEntry("session_db", "Session", "事务会话"),
        )

        plan = protect_glossary_terms("检查 Session", entries)

        self.assertEqual(plan.source, "检查 Session")
        self.assertEqual(plan.replacements, {})


class RequestTests(unittest.TestCase):
    def test_variants_place_reference_in_the_expected_message_role(self) -> None:
        dataset = _dataset()
        case = PocCase("c1", "terminology", "开发软件", ("develop", "software"))
        entries = dataset.glossary_sets["20"]

        request_a = build_request(dataset, case, entries, "A")
        request_b2 = build_request(dataset, case, entries, "B2")
        request_b3 = build_request(dataset, case, entries, "B3")

        self.assertIn("software: 软件", request_a["messages"][0]["content"])
        self.assertIn("MANDATORY_REFERENCE", request_b2["messages"][1]["content"])
        self.assertNotIn("MANDATORY_REFERENCE", request_b2["messages"][0]["content"])
        self.assertIn("Dynamic mandatory reference", request_b3["messages"][0]["content"])
        self.assertIn("⟦REF_", request_b2["protected_source"])
        self.assertIn("⟦REF_", request_b3["protected_source"])
        self.assertEqual(case.source, request_a["protected_source"])

    def test_machine_evaluation_restores_terms_and_checks_generic_rules(self) -> None:
        dataset = _dataset()
        case = PocCase(
            "c1",
            "terminology",
            "打开软件",
            ("software",),
            checks={"allowed_pattern": r"^[あ-ん\[\] ]+$", "no_edge_whitespace": True},
        )
        request = build_request(dataset, case, dataset.glossary_sets["20"], "B2")
        placeholder = next(iter(request["placeholders"]))

        result = evaluate_response(
            dataset,
            case,
            dataset.glossary_sets["20"],
            f"{placeholder}を  ひらく",
            request["placeholders"],
        )

        self.assertTrue(result["machine_ok"], result)
        self.assertEqual(result["restored_response"], "[そふとうぇあ]を  ひらく")

    def test_machine_evaluation_rejects_missing_placeholder(self) -> None:
        dataset = _dataset()
        case = PocCase("c1", "terminology", "软件", ("software",))
        request = build_request(dataset, case, dataset.glossary_sets["20"], "B3")

        result = evaluate_response(
            dataset,
            case,
            dataset.glossary_sets["20"],
            "そふとうぇあ",
            request["placeholders"],
        )

        self.assertFalse(result["placeholder_ok"])
        self.assertFalse(result["machine_ok"])


class DatasetValidationTests(unittest.TestCase):
    def test_strict_validation_rejects_draft_scale_and_case_shortcuts(self) -> None:
        with self.assertRaisesRegex(ValueError, "status must be 'approved'"):
            validate_dataset(_dataset(), strict=True)

    def test_non_strict_validation_allows_harness_fixtures(self) -> None:
        validate_dataset(_dataset(), strict=False)


if __name__ == "__main__":
    unittest.main()
