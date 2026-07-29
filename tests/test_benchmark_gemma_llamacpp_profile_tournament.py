from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_gemma_llamacpp_profile_tournament as tournament  # noqa: E402


class GemmaLlamaCppProfileTournamentTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _manifest_payload(self, root: Path) -> dict[str, object]:
        artifacts = root / "artifacts"
        artifacts.mkdir(exist_ok=True)
        paths = {
            name: artifacts / f"{name}.gguf"
            for name in ("baseline", "candidate", "draft-a", "draft-b")
        }
        for path in paths.values():
            path.write_bytes(b"neutral")
        return {
            "protocol_version": 1,
            "image": {
                "reference": "example.invalid/llama:locked",
                "expected_id": "sha256:" + ("1" * 64),
            },
            "helper_image": "alpine:3.22",
            "volumes": [
                {
                    "id": "product",
                    "name": "neutral-product-models",
                    "managed": False,
                },
                {
                    "id": "lab",
                    "name": "neutral-lab-models",
                    "managed": True,
                },
            ],
            "targets": [
                {
                    "id": "baseline",
                    "filename": "baseline.gguf",
                    "local_path": str(paths["baseline"]),
                    "volume_id": "product",
                    "baseline": True,
                    "mtp_draft_id": "draft-a",
                    "initial_ngl": 23,
                },
                {
                    "id": "candidate",
                    "filename": "candidate.gguf",
                    "local_path": str(paths["candidate"]),
                    "volume_id": "lab",
                    "baseline": False,
                    "mtp_draft_id": "draft-b",
                    "initial_ngl": 23,
                },
            ],
            "drafts": [
                {
                    "id": "draft-a",
                    "filename": "draft-a.gguf",
                    "local_path": str(paths["draft-a"]),
                    "volume_id": "product",
                    "allowed_target_ids": ["baseline"],
                },
                {
                    "id": "draft-b",
                    "filename": "draft-b.gguf",
                    "volume_filename": "draft-b-volume.gguf",
                    "local_path": str(paths["draft-b"]),
                    "volume_id": "lab",
                    "allowed_target_ids": ["candidate"],
                },
            ],
            "cache_contract": {
                "persistent_translation_cache": False,
                "exact_translation_memory": False,
                "project_checkpoint": False,
                "llama_prompt_cache_ram_mib": 0,
            },
            "preflight": {
                "max_idle_gpu_used_mb": 2048,
                "max_swap_growth_mb": 128,
            },
        }

    def _load_manifest(
        self,
        root: Path,
        mutate: object | None = None,
    ) -> tournament.BenchmarkManifest:
        payload = self._manifest_payload(root)
        if mutate is not None:
            mutate(payload)
        path = root / "model-manifest.json"
        self._write_json(path, payload)
        return tournament.load_manifest(path)

    def _fake_inventory(
        self,
        manifest: tournament.BenchmarkManifest,
    ) -> dict[str, object]:
        artifacts: dict[str, object] = {}
        for artifact in [*manifest.targets.values(), *manifest.drafts.values()]:
            artifacts[artifact.id] = {
                "filename": artifact.filename,
                "volume_filename": artifact.volume_filename,
                "volume_id": artifact.volume_id,
                "size": 1024,
                "sha256": hashlib.sha256(
                    artifact.id.encode("utf-8")
                ).hexdigest(),
                "gguf": {
                    "tokenizer_fingerprint": "2" * 64,
                },
            }
        return {
            "lock_sha256": "3" * 64,
            "artifacts": artifacts,
            "mtp_compatibility": {
                "baseline": {
                    "draft_id": "draft-a",
                    "manifest_pair_allowed": True,
                    "metadata_compatible": True,
                    "coverage": "full",
                    "load_generation_smoke_required": True,
                },
                "candidate": {
                    "draft_id": "draft-b",
                    "manifest_pair_allowed": True,
                    "metadata_compatible": True,
                    "coverage": "full",
                    "load_generation_smoke_required": True,
                },
            },
        }

    def test_manifest_and_profile_matrix_lock_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load_manifest(Path(tmp))

        profiles = tournament.enumerate_profiles(manifest)

        self.assertEqual(manifest.baseline.id, "baseline")
        self.assertEqual(len(profiles), 14)
        self.assertEqual(
            {profile.speculation for profile in profiles},
            {"none", "ngram", "mtp"},
        )
        self.assertTrue(
            all(
                profile.draft_id == "draft-a"
                for profile in profiles
                if profile.target_id == "baseline"
                and profile.speculation == "mtp"
            )
        )
        self.assertTrue(
            all(
                profile.draft_id == "draft-b"
                for profile in profiles
                if profile.target_id == "candidate"
                and profile.speculation == "mtp"
            )
        )

    def test_manifest_rejects_cross_directory_mtp_pairing(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["targets"][0]["mtp_draft_id"] = "draft-b"  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                tournament.ProtocolError,
                "Invalid MTP pairing",
            ):
                self._load_manifest(Path(tmp), mutate)

    def test_manifest_rejects_duplicate_volume_destination(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["drafts"][0]["volume_id"] = "lab"  # type: ignore[index]
            payload["drafts"][0]["volume_filename"] = (  # type: ignore[index]
                "draft-b-volume.gguf"
            )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                tournament.ProtocolError,
                "Duplicate volume destination",
            ):
                self._load_manifest(Path(tmp), mutate)

    def test_manifest_rejects_any_cache_contamination(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["cache_contract"]["project_checkpoint"] = True  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                tournament.ProtocolError,
                "cache_contract",
            ):
                self._load_manifest(Path(tmp), mutate)

    def test_manifest_never_manages_product_baseline_volume(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["volumes"][0]["managed"] = True  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                tournament.ProtocolError,
                "product baseline volume",
            ):
                self._load_manifest(Path(tmp), mutate)

    def test_manifest_and_raw_output_must_live_outside_git(self) -> None:
        with self.assertRaises(tournament.ProtocolError):
            tournament.load_manifest(ROOT / "neutral-model-manifest.json")
        with self.assertRaises(tournament.ProtocolError):
            tournament._require_external_path(
                ROOT / "neutral-result",
                label="result output",
            )

    def test_external_json_write_is_atomic_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "immutable.json"
            tournament._atomic_write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                tournament._atomic_write_json(path, {"value": 2})
            payload = json.loads(path.read_text(encoding="utf-8"))
            leftovers = sorted(
                item.name
                for item in path.parent.iterdir()
                if item.name != path.name
            )

        self.assertEqual(payload, {"value": 1})
        self.assertEqual(leftovers, [])

    def test_manifest_rejects_relative_model_path(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["targets"][0]["local_path"] = "relative-model.gguf"  # type: ignore[index]

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                tournament.ProtocolError,
                "local_path must be absolute",
            ):
                self._load_manifest(Path(tmp), mutate)

    def test_server_commands_use_current_mtp_and_ngram_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load_manifest(Path(tmp))
            inventory = self._fake_inventory(manifest)

        no_spec = tournament.find_profile(manifest, "baseline__none")
        ngram = tournament.find_profile(manifest, "baseline__ngram-8")
        mtp = tournament.find_profile(manifest, "baseline__mtp-4")
        no_spec_command = tournament.build_server_command(
            manifest,
            inventory,
            no_spec,
        )
        ngram_command = tournament.build_server_command(
            manifest,
            inventory,
            ngram,
        )
        mtp_command = tournament.build_server_command(
            manifest,
            inventory,
            mtp,
        )

        self.assertNotIn("--spec-type", no_spec_command)
        self.assertIn("ngram-mod", ngram_command)
        self.assertIn("--spec-ngram-mod-n-min", ngram_command)
        self.assertIn("--spec-ngram-mod-n-max", ngram_command)
        self.assertEqual(
            ngram_command[ngram_command.index("--spec-ngram-mod-n-min") + 1],
            "8",
        )
        self.assertEqual(
            ngram_command[ngram_command.index("--spec-ngram-mod-n-max") + 1],
            "8",
        )
        self.assertNotIn("--spec-draft-n-max", ngram_command)
        self.assertIn("draft-mtp", mtp_command)
        self.assertIn("-md", mtp_command)
        self.assertIn(
            "/volumes/lab/draft-b-volume.gguf",
            tournament.build_server_command(
                manifest,
                inventory,
                tournament.find_profile(manifest, "candidate__mtp-4"),
            ),
        )
        self.assertIn("--spec-draft-n-max", mtp_command)
        self.assertIn("--spec-draft-ngl", mtp_command)
        self.assertIn("-ctkd", mtp_command)
        self.assertIn("-ctvd", mtp_command)
        self.assertNotIn("-cd", mtp_command)
        self.assertNotIn("--draft", mtp_command)
        self.assertEqual(mtp_command[mtp_command.index("-ctk") + 1], "f16")
        self.assertEqual(mtp_command[mtp_command.index("-ctv") + 1], "f16")
        self.assertEqual(mtp_command[mtp_command.index("-ctkd") + 1], "f16")
        self.assertEqual(mtp_command[mtp_command.index("-ctvd") + 1], "f16")
        self.assertEqual(mtp_command[mtp_command.index("-c") + 1], "4096")
        self.assertEqual(mtp_command[mtp_command.index("-np") + 1], "1")
        self.assertEqual(mtp_command[mtp_command.index("-t") + 1], "10")
        self.assertEqual(mtp_command[mtp_command.index("--cache-ram") + 1], "0")

    def test_runtime_fingerprint_covers_ngl_target_and_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load_manifest(Path(tmp))
            inventory = self._fake_inventory(manifest)

        original = tournament.find_profile(manifest, "candidate__mtp-4")
        changed_ngl = tournament.find_profile(
            manifest,
            "candidate__mtp-4",
            target_ngl=22,
        )
        original_fingerprint = tournament.runtime_fingerprint(
            manifest,
            inventory,
            original,
        )
        changed_fingerprint = tournament.runtime_fingerprint(
            manifest,
            inventory,
            changed_ngl,
        )
        changed_inventory = copy.deepcopy(inventory)
        changed_inventory["artifacts"]["draft-b"]["sha256"] = "4" * 64  # type: ignore[index]
        draft_fingerprint = tournament.runtime_fingerprint(
            manifest,
            changed_inventory,
            original,
        )

        self.assertNotEqual(original_fingerprint, changed_fingerprint)
        self.assertNotEqual(original_fingerprint, draft_fingerprint)

    def test_volume_verification_rejects_unknown_artifact_before_docker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load_manifest(Path(tmp))
            inventory = self._fake_inventory(manifest)

        with self.assertRaisesRegex(
            tournament.ProtocolError,
            "Unknown volume artifact IDs",
        ):
            tournament.verify_volume_artifacts(
                manifest,
                inventory,
                full_hash=False,
                artifact_ids=["not-in-manifest"],
            )

    def test_container_contract_checks_port_gpu_and_read_only_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load_manifest(Path(tmp))
            inventory = self._fake_inventory(manifest)
        profile = tournament.find_profile(manifest, "candidate__mtp-4")
        contract = tournament._container_contract(
            manifest,
            inventory,
            profile,
        )
        inspected = {
            "Image": manifest.expected_image_id,
            "Config": {
                "Cmd": contract["command"],
                "Entrypoint": ["/app/llama-server"],
                "Labels": {
                    "comic-translate.runtime": "gemma-profile-tournament",
                    "comic-translate.config-fingerprint": contract[
                        "fingerprint"
                    ],
                },
            },
            "HostConfig": {
                "Privileged": False,
                "AutoRemove": False,
                "RestartPolicy": {"Name": "no"},
                "DeviceRequests": [{"Driver": ""}],
                "NetworkMode": "default",
                "PortBindings": {
                    "8080/tcp": [
                        {
                            "HostIp": "127.0.0.1",
                            "HostPort": "18080",
                        }
                    ]
                },
            },
            "Mounts": [
                {
                    "Destination": "/volumes/lab",
                    "Name": "neutral-lab-models",
                    "Type": "volume",
                    "RW": False,
                }
            ],
        }

        self.assertEqual(
            tournament.container_contract_errors(
                inspected=inspected,
                manifest=manifest,
                profile=profile,
                contract=contract,
            ),
            [],
        )
        inspected["Mounts"][0]["RW"] = True
        inspected["HostConfig"]["PortBindings"]["8080/tcp"][0][  # type: ignore[index]
            "HostIp"
        ] = "0.0.0.0"
        errors = tournament.container_contract_errors(
            inspected=inspected,
            manifest=manifest,
            profile=profile,
            contract=contract,
        )
        self.assertIn("loopback port binding", errors)
        self.assertIn("read-only volume mount /volumes/lab", errors)

    def test_mtp_command_rejects_tokenizer_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._load_manifest(Path(tmp))
            inventory = self._fake_inventory(manifest)
        inventory["mtp_compatibility"]["baseline"][  # type: ignore[index]
            "metadata_compatible"
        ] = False

        with self.assertRaisesRegex(
            tournament.ProtocolError,
            "tokenizer/vocabulary metadata",
        ):
            tournament.build_server_command(
                manifest,
                inventory,
                tournament.find_profile(manifest, "baseline__mtp-4"),
            )

    def _write_fake_gguf(
        self,
        path: Path,
        *,
        tokens: list[str],
    ) -> None:
        metadata: list[tuple[str, int, object]] = [
            ("general.architecture", 8, "neutral-moe"),
            ("tokenizer.ggml.model", 8, "neutral-tokenizer"),
            ("tokenizer.ggml.pre", 8, "neutral-pre"),
            ("tokenizer.ggml.tokens", 9, tokens),
        ]
        with path.open("wb") as stream:
            stream.write(b"GGUF")
            stream.write(struct.pack("<I", 3))
            stream.write(struct.pack("<Q", 0))
            stream.write(struct.pack("<Q", len(metadata)))
            for key, value_type, value in metadata:
                key_bytes = key.encode("utf-8")
                stream.write(struct.pack("<Q", len(key_bytes)))
                stream.write(key_bytes)
                stream.write(struct.pack("<I", value_type))
                if value_type == 8:
                    raw = str(value).encode("utf-8")
                    stream.write(struct.pack("<Q", len(raw)))
                    stream.write(raw)
                else:
                    stream.write(struct.pack("<I", 8))
                    stream.write(struct.pack("<Q", len(value)))  # type: ignore[arg-type]
                    for item in value:  # type: ignore[union-attr]
                        raw = str(item).encode("utf-8")
                        stream.write(struct.pack("<Q", len(raw)))
                        stream.write(raw)

    def test_gguf_tokenizer_fingerprint_detects_incompatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.gguf"
            second = root / "second.gguf"
            third = root / "third.gguf"
            self._write_fake_gguf(first, tokens=["a", "b"])
            self._write_fake_gguf(second, tokens=["a", "b"])
            self._write_fake_gguf(third, tokens=["a", "c"])

            first_contract = tournament.gguf_metadata_contract(first)
            second_contract = tournament.gguf_metadata_contract(second)
            third_contract = tournament.gguf_metadata_contract(third)

        self.assertEqual(
            first_contract["tokenizer_fingerprint"],
            second_contract["tokenizer_fingerprint"],
        )
        self.assertNotEqual(
            first_contract["tokenizer_fingerprint"],
            third_contract["tokenizer_fingerprint"],
        )
        self.assertEqual(first_contract["architecture"], "neutral-moe")

    def test_mtp_metadata_allows_missing_draft_vocab_but_not_conflict(self) -> None:
        matching_digest = "a" * 64
        target = {
            "selected_value_sha256": {
                "tokenizer.ggml.model": matching_digest,
                "tokenizer.ggml.tokens": "b" * 64,
            }
        }
        partial_draft = {
            "selected_value_sha256": {
                "tokenizer.ggml.model": matching_digest,
            }
        }
        conflicting_draft = {
            "selected_value_sha256": {
                "tokenizer.ggml.model": matching_digest,
                "tokenizer.ggml.tokens": "c" * 64,
            }
        }
        ancillary_difference = {
            "selected_value_sha256": {
                "tokenizer.ggml.model": matching_digest,
                "tokenizer.ggml.tokens": "b" * 64,
                "tokenizer.ggml.token_type": "d" * 64,
            }
        }
        target_with_type = {
            "selected_value_sha256": {
                **target["selected_value_sha256"],
                "tokenizer.ggml.token_type": "e" * 64,
            }
        }

        partial = tournament.mtp_metadata_compatibility(
            target,
            partial_draft,
        )
        conflict = tournament.mtp_metadata_compatibility(
            target,
            conflicting_draft,
        )
        ancillary = tournament.mtp_metadata_compatibility(
            target_with_type,
            ancillary_difference,
        )

        self.assertTrue(partial["metadata_compatible"])
        self.assertEqual(partial["coverage"], "partial")
        self.assertIn(
            "tokenizer.ggml.tokens",
            partial["missing_core_in_draft"],
        )
        self.assertFalse(conflict["metadata_compatible"])
        self.assertEqual(
            conflict["core_mismatched_keys"],
            ["tokenizer.ggml.tokens"],
        )
        self.assertTrue(ancillary["metadata_compatible"])
        self.assertEqual(
            ancillary["ancillary_mismatched_keys"],
            ["tokenizer.ggml.token_type"],
        )

    def _corpus_payload(self) -> dict[str, object]:
        sensitive_groups = []
        contiguous_groups = []
        item_number = 0
        for language in ("Japanese", "Chinese", "English"):
            sensitive_items = []
            for _ in range(5):
                item_number += 1
                sensitive_items.append(
                    {
                        "id": f"sensitive-{item_number:02d}",
                        "text": f"neutral sensitive text {item_number}",
                        "reference": f"neutral reference {item_number}",
                        "review_focus": "명시 의미와 화자 보존",
                    }
                )
            sensitive_groups.append(
                {
                    "id": f"sensitive-{language.casefold()}",
                    "source_language": language,
                    "target_language": "Korean",
                    "items": sensitive_items,
                }
            )
            for group_index in range(3):
                items = []
                for block_index in range(6):
                    item_number += 1
                    items.append(
                        {
                            "id": (
                                f"contiguous-{language.casefold()}-"
                                f"{group_index}-{block_index}"
                            ),
                            "text": f"neutral context text {item_number}",
                            "reference": f"neutral reference {item_number}",
                            "review_focus": "문맥 의미 보존",
                        }
                    )
                contiguous_groups.append(
                    {
                        "id": f"{language.casefold()}-{group_index}",
                        "source_language": language,
                        "target_language": "Korean",
                        "items": items,
                    }
                )
        return {
            "protocol_version": 1,
            "sensitive_groups": sensitive_groups,
            "contiguous_groups": contiguous_groups,
        }

    def test_corpus_locks_counts_and_stage_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            self._write_json(path, self._corpus_payload())
            corpus = tournament.load_corpus(path)

        self.assertEqual(corpus["counts"]["sensitive"], 15)
        self.assertEqual(corpus["counts"]["contiguous"], 54)
        self.assertEqual(
            sum(
                len(group["items"])
                for group in tournament._stage_groups(corpus, "screen18")
            ),
            18,
        )
        self.assertEqual(
            sum(
                len(group["items"])
                for group in tournament._stage_groups(corpus, "final54")
            ),
            54,
        )
        self.assertEqual(
            sum(
                len(group["items"])
                for group in tournament._stage_groups(corpus, "breakeven30")
            ),
            30,
        )
        self.assertEqual(
            {
                language: sum(
                    len(group["items"])
                    for group in tournament._stage_groups(
                        corpus,
                        "breakeven30",
                    )
                    if group["source_language"] == language
                )
                for language in ("Japanese", "Chinese", "English")
            },
            {"Japanese": 10, "Chinese": 10, "English": 10},
        )

    def test_corpus_rejects_missing_language_block(self) -> None:
        payload = self._corpus_payload()
        payload["contiguous_groups"][-1]["items"].pop()  # type: ignore[index]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            self._write_json(path, payload)
            with self.assertRaisesRegex(
                tournament.ProtocolError,
                "18 Japanese",
            ):
                tournament.load_corpus(path)

    def test_paired_bootstrap_has_no_minimum_gain_threshold(self) -> None:
        baseline = [10.0] * 24
        candidate = [9.99] * 24

        result = tournament.paired_speed_bootstrap(
            baseline,
            candidate,
            samples=5000,
        )

        self.assertGreater(result["mean_gain_percent"], 0.0)
        self.assertGreater(result["one_sided_95_lower_percent"], 0.0)
        self.assertTrue(result["proven_faster"])
        self.assertFalse(result["requires_third_round"] if "requires_third_round" in result else False)

    def test_paired_bootstrap_marks_noisy_gain_uncertain(self) -> None:
        result = tournament.paired_speed_bootstrap(
            [10.0] * 8,
            [9.0, 11.0] * 4,
            samples=5000,
        )

        self.assertTrue(result["uncertain"])
        self.assertFalse(result["proven_faster"])
        self.assertFalse(result["proven_slower"])

    def test_ngl_probe_order_starts_up_or_down_from_23(self) -> None:
        self.assertEqual(
            tournament.next_ngl_probe_values(
                initial_ngl=23,
                max_ngl=26,
                initial_passed=True,
            ),
            [24, 25, 26],
        )
        self.assertEqual(
            tournament.next_ngl_probe_values(
                initial_ngl=23,
                max_ngl=26,
                initial_passed=False,
            )[:3],
            [22, 21, 20],
        )
        self.assertEqual(
            tournament.next_ngl_probe_values(
                initial_ngl=23,
                max_ngl=26,
                initial_passed=False,
                initial_swap_only_failure=True,
            ),
            [24, 25, 26],
        )

    def test_ngl_tuning_searches_up_after_swap_only_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._load_manifest(root)
            inventory = self._fake_inventory(manifest)
            profile = tournament.find_profile(manifest, "baseline__none")
            attempted_ngls: list[int] = []

            def fake_probe(**kwargs: object) -> dict[str, object]:
                candidate = kwargs["profile"]
                ngl = candidate.target_ngl  # type: ignore[union-attr]
                attempted_ngls.append(ngl)
                passed = ngl == 25
                return {
                    "status": "passed" if passed else "failed",
                    "resource_gates": {
                        "swap_growth_ok": passed,
                        "shared_gpu_growth_ok": True,
                    },
                }

            with mock.patch.object(
                tournament,
                "probe_profile_load",
                side_effect=fake_probe,
            ):
                result = tournament.tune_profile_ngl(
                    manifest=manifest,
                    inventory=inventory,
                    profile=profile,
                    max_ngl=26,
                    output_path=root / "tuning.json",
                    start_timeout_sec=1,
                    request_timeout_sec=1,
                )

        self.assertEqual(attempted_ngls, [23, 24, 25, 26])
        self.assertEqual(result["safe_target_ngls"], [25])
        self.assertEqual(result["safe_max_target_ngl"], 25)
        self.assertEqual(result["first_failed_target_ngl"], 26)
        self.assertEqual(result["screen_comparison_target_ngls"], [25])

    def test_ngl_tuning_caps_search_at_model_output_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._load_manifest(root)
            inventory = self._fake_inventory(manifest)
            inventory["artifacts"]["baseline"]["gguf"]["selected"] = {  # type: ignore[index]
                "gemma4.block_count": 30,
            }
            profile = tournament.find_profile(manifest, "baseline__none")
            attempted_ngls: list[int] = []

            def fake_probe(**kwargs: object) -> dict[str, object]:
                candidate = kwargs["profile"]
                attempted_ngls.append(  # type: ignore[union-attr]
                    candidate.target_ngl  # type: ignore[union-attr]
                )
                return {
                    "status": "passed",
                    "resource_gates": {
                        "swap_growth_ok": True,
                        "shared_gpu_growth_ok": True,
                    },
                }

            with mock.patch.object(
                tournament,
                "probe_profile_load",
                side_effect=fake_probe,
            ):
                result = tournament.tune_profile_ngl(
                    manifest=manifest,
                    inventory=inventory,
                    profile=profile,
                    max_ngl=40,
                    output_path=root / "capped-tuning.json",
                    start_timeout_sec=1,
                    request_timeout_sec=1,
                )

        self.assertEqual(attempted_ngls, list(range(23, 32)))
        self.assertEqual(result["model_block_count"], 30)
        self.assertEqual(result["effective_max_target_ngl"], 31)
        self.assertEqual(result["safe_max_target_ngl"], 31)
        self.assertEqual(result["screen_comparison_target_ngls"], [30, 31])

    def test_ngl_tuning_probes_unseen_lower_comparison_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._load_manifest(root)
            inventory = self._fake_inventory(manifest)
            profile = tournament.find_profile(
                manifest,
                "baseline__none",
                target_ngl=25,
            )
            attempted_ngls: list[int] = []

            def fake_probe(**kwargs: object) -> dict[str, object]:
                candidate = kwargs["profile"]
                ngl = candidate.target_ngl  # type: ignore[union-attr]
                attempted_ngls.append(ngl)
                passed = ngl in {24, 25}
                return {
                    "status": "passed" if passed else "failed",
                    "resource_gates": {
                        "swap_growth_ok": passed,
                        "shared_gpu_growth_ok": True,
                    },
                }

            with mock.patch.object(
                tournament,
                "probe_profile_load",
                side_effect=fake_probe,
            ):
                result = tournament.tune_profile_ngl(
                    manifest=manifest,
                    inventory=inventory,
                    profile=profile,
                    max_ngl=26,
                    output_path=root / "lower-neighbor-tuning.json",
                    start_timeout_sec=1,
                    request_timeout_sec=1,
                )

        self.assertEqual(attempted_ngls, [25, 26, 24])
        self.assertEqual(result["safe_target_ngls"], [24, 25])
        self.assertEqual(result["screen_comparison_target_ngls"], [24, 25])

    def test_mtp_tuning_skips_ngl_sweep_after_draft_load_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._load_manifest(root)
            inventory = self._fake_inventory(manifest)
            profile = tournament.find_profile(manifest, "baseline__mtp-4")
            attempts: list[tuple[int, str]] = []

            def fake_probe(**kwargs: object) -> dict[str, object]:
                candidate = kwargs["profile"]
                attempts.append(  # type: ignore[union-attr]
                    (
                        candidate.target_ngl,  # type: ignore[union-attr]
                        candidate.draft_ngl,  # type: ignore[union-attr]
                    )
                )
                if candidate.draft_ngl != "0":  # type: ignore[union-attr]
                    return {
                        "status": "failed",
                        "failure_kind": "draft_model_load",
                    }
                return {
                    "status": "passed",
                    "resource_gates": {
                        "swap_growth_ok": True,
                        "shared_gpu_growth_ok": True,
                    },
                }

            with mock.patch.object(
                tournament,
                "probe_profile_load",
                side_effect=fake_probe,
            ):
                result = tournament.tune_profile_ngl(
                    manifest=manifest,
                    inventory=inventory,
                    profile=profile,
                    max_ngl=23,
                    output_path=root / "mtp-fallback-tuning.json",
                    start_timeout_sec=1,
                    request_timeout_sec=1,
                )

        self.assertEqual(
            attempts,
            [(23, "all"), (23, "0"), (22, "0")],
        )
        self.assertEqual(result["selected_draft_ngl"], "0")
        self.assertEqual(result["safe_max_target_ngl"], 23)
        self.assertEqual(result["screen_comparison_target_ngls"], [22, 23])

    def test_draft_acceptance_log_fallback_is_aggregated(self) -> None:
        parsed = tournament.parse_draft_acceptance_logs(
            "\n".join(
                [
                    "draft acceptance = 0.30000 ( 12 accepted / 40 generated), mean len = 2.20",
                    "draft acceptance = 0.09848 ( 13 accepted / 132 generated), mean len = 1.39",
                ]
            )
        )
        summary = tournament.summarize_speculation_metrics(
            {},
            {
                "gemma_completion_tokens": 50,
                "gemma_decode_ms": 1000.0,
                "gemma_http_attempt_count": 2,
            },
            parsed,
        )

        self.assertEqual(parsed["line_count"], 2)
        self.assertEqual(summary["draft_tokens"], 172)
        self.assertEqual(summary["accepted_tokens"], 25)
        self.assertAlmostEqual(summary["acceptance_rate"], 25 / 172)
        self.assertEqual(summary["telemetry_source"], "llama-log")

    def test_draft_load_failure_is_classified(self) -> None:
        result = tournament.classify_profile_failure(
            RuntimeError("server exited"),
            "failed to load draft model, '/models/mtp.gguf'",
        )

        self.assertEqual(result, "draft_model_load")

    def test_prometheus_parser_keeps_speculation_and_timing_metrics(self) -> None:
        parsed = tournament.parse_prometheus_metrics(
            "\n".join(
                [
                    "# HELP ignored ignored",
                    "llamacpp:tokens_predicted_total 20",
                    'llamacpp:tokens_drafted_total{model="neutral"} 12',
                    "llamacpp:tokens_drafted_accepted_total 8",
                    "unrelated_metric 99",
                ]
            )
        )

        self.assertEqual(parsed["llamacpp:tokens_predicted_total"], 20)
        self.assertEqual(
            parsed['llamacpp:tokens_drafted_total{model="neutral"}'],
            12,
        )
        self.assertEqual(
            parsed["llamacpp:tokens_drafted_accepted_total"],
            8,
        )
        self.assertNotIn("unrelated_metric", parsed)
        summary = tournament.summarize_speculation_metrics(
            parsed,
            {
                "gemma_completion_tokens": 10,
                "gemma_decode_ms": 500.0,
                "gemma_http_attempt_count": 2,
            },
        )
        self.assertEqual(summary["draft_tokens"], 12)
        self.assertEqual(summary["accepted_tokens"], 8)
        self.assertAlmostEqual(summary["acceptance_rate"], 8 / 12)
        self.assertEqual(summary["accepted_tokens_per_http_attempt"], 4)
        self.assertEqual(summary["tpot_ms"], 50)

    def test_memory_parser_normalizes_docker_units(self) -> None:
        self.assertEqual(tournament.parse_memory_mib("512MiB"), 512)
        self.assertEqual(tournament.parse_memory_mib("1.5GiB"), 1536)
        self.assertIsNone(tournament.parse_memory_mib("not-memory"))

    def test_container_swap_stats_reads_cgroup_v2_bytes(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stdout=(
                "memory.swap.current=1048576\n"
                "memory.swap.peak=268435456\n"
            ),
            stderr="",
        )
        with (
            mock.patch.object(
                tournament,
                "docker_executable",
                return_value="docker",
            ),
            mock.patch.object(
                tournament,
                "run_process",
                return_value=completed,
            ) as run_process,
        ):
            stats = tournament.query_container_swap_stats("neutral-container")

        self.assertTrue(stats["available"])
        self.assertEqual(stats["current_mb"], 1)
        self.assertEqual(stats["peak_mb"], 256)
        command = run_process.call_args.args[0]
        self.assertEqual(command[:3], ["docker", "exec", "neutral-container"])
        self.assertIn("memory.swap.peak", command[-1])

    def _comparison_result(
        self,
        *,
        profile_id: str,
        round_index: int,
    ) -> dict[str, object]:
        request_contract = {
            "request_mode": "contextual-single",
            "chunk_size": 6,
            "context_size": 4096,
            "threads": 10,
            "target_kv": "f16",
            "draft_kv": "f16",
            "max_completion_tokens": 512,
            "prompt_profile": tournament.DEFAULT_GEMMA_PROMPT_PROFILE,
            "response_format_mode": (
                tournament.DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
            ),
            "response_schema_mode": (
                tournament.DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
            ),
            "temperature": tournament.DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
            "top_k": tournament.DEFAULT_GEMMA_TRANSLATION_TOP_K,
            "top_p": tournament.DEFAULT_GEMMA_TRANSLATION_TOP_P,
            "min_p": tournament.DEFAULT_GEMMA_TRANSLATION_MIN_P,
            "persistent_translation_cache": False,
            "exact_translation_memory": False,
            "project_checkpoint": False,
            "llama_prompt_cache_ram_mib": 0,
        }
        outputs = [
            {
                "item_id": "item-1",
                "source_sha256": "a" * 64,
            },
            {
                "item_id": "item-2",
                "source_sha256": "b" * 64,
            },
        ]
        if round_index == 2:
            outputs.reverse()
        return {
            "status": "passed",
            "profile": {"id": profile_id},
            "stage": "screen18",
            "round": round_index,
            "corpus_sha256": "c" * 64,
            "inventory_lock_sha256": "d" * 64,
            "request_contract": request_contract,
            "outputs": outputs,
        }

    def test_comparison_contract_accepts_reverse_order_but_rejects_tampering(
        self,
    ) -> None:
        results = [
            self._comparison_result(profile_id="baseline__none", round_index=1),
            self._comparison_result(profile_id="baseline__none", round_index=2),
        ]

        contract = tournament._comparison_result_contract(
            results,
            label="baseline",
        )

        self.assertEqual(contract["rounds"], [1, 2])
        tampered_source = copy.deepcopy(results)
        tampered_source[1]["outputs"][0]["source_sha256"] = "e" * 64  # type: ignore[index]
        with self.assertRaisesRegex(
            tournament.ProtocolError,
            "output corpus differs",
        ):
            tournament._comparison_result_contract(
                tampered_source,
                label="baseline",
            )
        contaminated = copy.deepcopy(results)
        contaminated[0]["request_contract"][  # type: ignore[index]
            "persistent_translation_cache"
        ] = True
        contaminated[1]["request_contract"][  # type: ignore[index]
            "persistent_translation_cache"
        ] = True
        with self.assertRaisesRegex(
            tournament.ProtocolError,
            "persistent_translation_cache",
        ):
            tournament._comparison_result_contract(
                contaminated,
                label="baseline",
            )

    def test_result_gate_rejects_parser_fallback_and_swap_growth(self) -> None:
        gates = tournament._result_gates(
            result={
                "outputs": [
                    {"item_id": "item-1", "empty": False},
                ],
                "stats": {
                    "gemma_contextual_merge_fallback_count": 1,
                },
            },
            expected_item_ids=["item-1"],
            resource_summary={"wsl_swap_growth_mb": 256},
            max_swap_growth_mb=128,
            max_shared_gpu_growth_mb=512,
        )

        self.assertFalse(gates["hard_gate_passed"])
        self.assertEqual(gates["unresolved_fallback_count"], 1)
        self.assertFalse(gates["swap_growth_ok"])
        self.assertEqual(gates["swap_gate_source"], "global-wsl-fallback")
        self.assertEqual(gates["global_wsl_swap_growth_mb"], 256)

    def test_result_gate_prefers_container_swap_over_global_wsl_noise(
        self,
    ) -> None:
        gates = tournament._result_gates(
            result={
                "outputs": [
                    {"item_id": "item-1", "empty": False},
                ],
                "stats": {},
                "container_swap": {
                    "available": True,
                    "current_mb": 0,
                    "peak_mb": 0,
                },
            },
            expected_item_ids=["item-1"],
            resource_summary={
                "wsl_swap_growth_mb": 256,
                "shared_gpu_growth_mb": 0,
            },
            max_swap_growth_mb=128,
            max_shared_gpu_growth_mb=512,
        )

        self.assertTrue(gates["hard_gate_passed"])
        self.assertTrue(gates["swap_growth_ok"])
        self.assertEqual(gates["swap_gate_source"], "container-cgroup")
        self.assertEqual(gates["swap_growth_mb"], 0)
        self.assertEqual(gates["container_swap_peak_mb"], 0)
        self.assertEqual(gates["global_wsl_swap_growth_mb"], 256)


if __name__ == "__main__":
    unittest.main()
