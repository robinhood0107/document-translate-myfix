from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_gemma_sampler_stability as sampler  # noqa: E402


def _case(case_id: str, *, language_pair: str, family: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "language_pair": language_pair,
        "family": family,
        "category": "meaning-control",
        "source_text": "Number 25 stays here." if language_pair == "en-ko" else "これは文です。",
        "canonical_ko": "25번은 여기에 남습니다." if language_pair == "en-ko" else "이것은 문장입니다.",
        "context_before": ["앞 문장"],
        "context_after": ["뒤 문장"],
        "required_meaning": ["preserve the proposition"],
        "forbidden_changes": ["number_change"],
        "allowed_style": ["ordinary synonym"],
        "evidence": {"kind": "test", "source_sha256": "a" * 64},
    }


def _manifest() -> dict[str, object]:
    cases = [
        _case(f"ja-router-{index:02d}", language_pair="ja-ko", family="router_mismatch")
        for index in range(14)
    ]
    cases.extend(
        _case(f"ja-control-{index:02d}", language_pair="ja-ko", family="meaning_control")
        for index in range(4)
    )
    cases.extend(
        _case(f"en-control-{index:02d}", language_pair="en-ko", family="meaning_control")
        for index in range(4)
    )
    return {
        "schema_version": "gemma-sampler-corpus-v1",
        "protocol_version": "gemma-sampler-stability-v1",
        "cases": cases,
    }


class GemmaSamplerStabilityTests(unittest.TestCase):
    def test_protocol_is_sanitized_and_pins_990_response_ceiling(self) -> None:
        payload = sampler.load_protocol()
        self.assertEqual(payload["corpus"]["total_cases"], 22)
        self.assertEqual(payload["maximum_total_responses"], 990)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("canonical_ko", serialized)
        self.assertNotIn("source_text", serialized)

    def test_manifest_validation_enforces_the_18_plus_4_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(_manifest(), ensure_ascii=False), encoding="utf-8")
            payload, digest = sampler.load_corpus_manifest(path)
        self.assertEqual(len(payload["cases"]), 22)
        self.assertEqual(len(digest), 64)

    def test_arm_reuse_uses_sampler_identity_not_phase_name(self) -> None:
        all_arms = sampler.build_arms(
            "all",
            selected_temperature=0.7,
            selected_top_p=0.95,
        )
        self.assertEqual(len(all_arms), 15)
        self.assertEqual(
            sum(arm.arm_id == "sampler-t0.7-p0.95-k64-m0" for arm in all_arms),
            1,
        )
        self.assertEqual(len(sampler.build_arms("temperature")), 11)
        self.assertEqual(len(sampler.build_arms("top-p", selected_temperature=0.7)), 3)
        self.assertEqual(
            len(sampler.build_arms("top-k", selected_temperature=0.7, selected_top_p=0.95)),
            3,
        )

    def test_seed_orders_are_forward_reverse_and_center_out(self) -> None:
        self.assertEqual(sampler.seed_case_order(0, 5), [0, 1, 2, 3, 4])
        self.assertEqual(sampler.seed_case_order(1, 5), [4, 3, 2, 1, 0])
        self.assertEqual(sampler.seed_case_order(2, 5), [2, 3, 1, 4, 0])

    def test_request_changes_only_sampler_values_and_seed(self) -> None:
        case = _case("ja-router-00", language_pair="ja-ko", family="router_mismatch")
        first_arm = sampler.SamplerArm("temperature", 0.0, 0.95, 64)
        second_arm = sampler.SamplerArm("temperature", 1.0, 0.95, 64)
        first, first_keys = sampler.build_case_payload(case, first_arm, 20260801)
        second, second_keys = sampler.build_case_payload(case, second_arm, 20260801)
        self.assertEqual(first_keys, ["translation"])
        self.assertEqual(second_keys, first_keys)
        self.assertEqual(sampler.request_contract_hash(first), sampler.request_contract_hash(second))
        self.assertNotEqual(first["temperature"], second["temperature"])
        self.assertEqual(first["seed"], second["seed"])
        first_user = first["messages"][1]["content"][0]["text"]
        self.assertIn("target_text", first_user)
        self.assertIn("これは文です。", first_user)

    def test_response_validator_rejects_schema_order_and_channel_tokens(self) -> None:
        valid = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"translation":"좋습니다."}'},
                }
            ],
            "usage": {"completion_tokens": 3},
        }
        self.assertEqual(
            sampler.validate_response(valid, expected_keys=["translation"])["structural_status"],
            "pass",
        )
        malformed = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '<|channel>thought\\n<channel|>{"extra":"x","translation":"value"}'
                    },
                }
            ]
        }
        result = sampler.validate_response(malformed, expected_keys=["translation"])
        self.assertEqual(result["structural_status"], "fail")
        self.assertIn("finish_reason:length", result["structural_failures"])
        self.assertIn("schema_order_or_count", result["structural_failures"])
        self.assertTrue(result["channel_token_sanitized"])

    def test_quality_gate_keeps_noncanonical_unique_outputs_for_manual_review(self) -> None:
        case = _case("en-control-00", language_pair="en-ko", family="meaning_control")
        exact = {"structural_status": "pass", "translation": case["canonical_ko"], "translation_hash": "a"}
        self.assertEqual(sampler.automatic_quality(case, exact)["status"], "pass")
        unique = {"structural_status": "pass", "translation": "다른 표현입니다. 25.", "translation_hash": "b"}
        self.assertEqual(sampler.automatic_quality(case, unique)["status"], "review_required")
        number_changed = {"structural_status": "pass", "translation": "번호가 다릅니다.", "translation_hash": "c"}
        self.assertEqual(sampler.automatic_quality(case, number_changed)["status"], "fail")
        self.assertIn("number_change", sampler.automatic_quality(case, number_changed)["hard_failures"])

    def test_atomic_response_files_are_resume_eligible_only_when_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            sampler.atomic_json_write(
                path,
                {
                    "run_status": "complete",
                    "response_id": "response-id",
                    "contract_hash": "contract",
                },
            )
            self.assertIsNotNone(
                sampler.read_completed_response(
                    path,
                    expected_contract_hash="contract",
                    expected_response_id="response-id",
                )
            )
            path.write_text('{"run_status":"incomplete"}', encoding="utf-8")
            self.assertIsNone(
                sampler.read_completed_response(
                    path,
                    expected_contract_hash="contract",
                    expected_response_id="response-id",
                )
            )

    def test_summary_reapplies_manual_judgments_and_counts_channel_sanitization(self) -> None:
        manifest = _manifest()
        arm = sampler.SamplerArm("temperature", 0.0, 0.95, 64)
        manifest_sha256 = "a" * 64
        cases = [case for case in manifest["cases"] if isinstance(case, dict)]
        first_case = cases[0]
        alternate_translation = "자연스러운 대체 번역입니다."
        alternate_hash = sampler.text_sha256(alternate_translation)
        judgments = {
            (str(first_case["case_id"]), arm.arm_id, alternate_hash): {
                "status": "pass",
                "naturalness": 4,
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            responses = Path(directory) / "responses"
            responses.mkdir()
            for seed_index, seed in enumerate(sampler.DEFAULT_SEEDS):
                for case_index, case in enumerate(cases):
                    translation = (
                        alternate_translation
                        if case is first_case
                        else str(case["canonical_ko"])
                    )
                    translation_hash = sampler.text_sha256(translation)
                    sampler.atomic_json_write(
                        responses / f"{seed_index:02d}-{case_index:02d}.json",
                        {
                            "run_status": "complete",
                            "manifest_sha256": manifest_sha256,
                            "case_id": case["case_id"],
                            "arm_id": arm.arm_id,
                            "validation": {
                                "structural_status": "pass",
                                "translation": translation,
                                "translation_hash": translation_hash,
                                "channel_token_sanitized": True,
                            },
                        },
                    )

            summary = sampler.summarize_run(
                output_dir=Path(directory),
                protocol={"protocol_version": "gemma-sampler-stability-v1"},
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                arms=[arm],
                judgments=judgments,
            )

        arm_summary = summary["arm_summaries"][0]
        self.assertTrue(arm_summary["clean_candidate"])
        self.assertEqual(arm_summary["channel_token_sanitized_responses"], 66)
        self.assertEqual(summary["clean_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
