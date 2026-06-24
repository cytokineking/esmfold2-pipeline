from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import esmfold2_pipeline.validation.msa as msa_module
from esmfold2_pipeline.validation import (
    binder_msa_cache_dir,
    MsaPair,
    ProtenixMsaConfig,
    ProtenixTaskInput,
    build_protenix_input_json,
    resolve_binder_msa_pair,
    resolve_binder_msa_pairs,
    resolve_target_msa_pairs,
    target_msa_cache_dir,
)


class ProtenixMsaTest(unittest.TestCase):
    def test_build_protenix_input_json_attaches_target_msa_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = ProtenixTaskInput(
                validation_id="val_one",
                candidate_id="cand_one",
                model_name="protenix-v2",
                selection_rank=1,
                designed_sequence="ACD",
                target_sequences=("GG",),
                target_labels=("B",),
                seed=101,
                binder_scaffold="miniprotein",
                framework=None,
            )
            msa = MsaPair(
                pairing=">query\nGG\n",
                non_pairing=">query\nGG\n>hit\nGA\n",
                source="provided",
            )

            input_json, _sample_names, _chain_maps = build_protenix_input_json(
                [task],
                root / "input",
                target_msas={"val_one": (msa,)},
            )

            payload = json.loads(input_json.read_text())
            target_chain = payload[0]["sequences"][1]["proteinChain"]
            self.assertEqual(target_chain["sequence"], "GG")
            self.assertIn("pairedMsaPath", target_chain)
            self.assertIn("unpairedMsaPath", target_chain)
            self.assertEqual(
                Path(target_chain["pairedMsaPath"]).read_text(),
                ">query\nGG\n",
            )
            self.assertEqual(
                Path(target_chain["unpairedMsaPath"]).read_text(),
                ">query\nGG\n>hit\nGA\n",
            )

    def test_resolve_target_msa_pairs_caches_provided_msa(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            msa_dir = Path(tmpdir) / "target_msa"
            msa_dir.mkdir(parents=True)
            (msa_dir / "non_pairing.a3m").write_text(">query\nGGGG\n>hit\nGGGA\n")
            (msa_dir / "pairing.a3m").write_text(">query\nGGGG\n")
            config = ProtenixMsaConfig(
                target_mode="provided",
                target_msa_dir=msa_dir,
            )

            pairs = resolve_target_msa_pairs(
                campaign_dir,
                target_sequences=("GGGG",),
                target_labels=("B",),
                target_name="target",
                config=config,
            )

            self.assertEqual(len(pairs), 1)
            assert pairs[0] is not None
            self.assertEqual(pairs[0].source, "provided")
            cache_dir = target_msa_cache_dir(
                campaign_dir,
                sequence="GGGG",
                config=config,
            )
            self.assertTrue((cache_dir / "non_pairing.a3m").exists())
            self.assertTrue((cache_dir / "pairing.a3m").exists())
            metadata = json.loads((cache_dir / "metadata.json").read_text())
            self.assertEqual(metadata["target_mode"], "provided")
            self.assertEqual(metadata["target_label"], "B")

            second = resolve_target_msa_pairs(
                campaign_dir,
                target_sequences=("GGGG",),
                target_labels=("B",),
                target_name="target",
                config=config,
            )
            assert second[0] is not None
            self.assertEqual(second[0].non_pairing, pairs[0].non_pairing)

    def test_resolve_target_msa_pairs_uses_server_cache_before_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            config = ProtenixMsaConfig(
                target_mode="server",
                server_url="https://api.colabfold.com",
                status_poll_interval_seconds=0,
            )
            calls = 0

            def fetcher(sequence: str, _config: ProtenixMsaConfig) -> MsaPair:
                nonlocal calls
                calls += 1
                return MsaPair(
                    pairing=f">query\n{sequence}\n",
                    non_pairing=f">query\n{sequence}\n>hit\nGGGA\n",
                    source="server",
                )

            first = resolve_target_msa_pairs(
                campaign_dir,
                target_sequences=("GGGG",),
                target_labels=("B",),
                target_name="target",
                config=config,
                fetcher=fetcher,
            )
            second = resolve_target_msa_pairs(
                campaign_dir,
                target_sequences=("GGGG",),
                target_labels=("B",),
                target_name="target",
                config=config,
                fetcher=fetcher,
            )

            self.assertEqual(calls, 1)
            assert first[0] is not None
            assert second[0] is not None
            self.assertEqual(second[0].non_pairing, first[0].non_pairing)
            self.assertEqual(second[0].cache_dir, first[0].cache_dir)

    def test_build_protenix_input_json_attaches_binder_msa_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = ProtenixTaskInput(
                validation_id="val_one",
                candidate_id="cand_one",
                model_name="protenix-v2",
                selection_rank=1,
                designed_sequence="ACD",
                target_sequences=("GG",),
                target_labels=("B",),
                seed=101,
                binder_scaffold="miniprotein",
                framework=None,
            )
            msa = MsaPair(
                pairing=">query\nACD\n",
                non_pairing=">query\nACD\n",
                source="single_sequence",
            )

            input_json, _sample_names, _chain_maps = build_protenix_input_json(
                [task],
                root / "input",
                binder_msas={"val_one": msa},
            )

            payload = json.loads(input_json.read_text())
            binder_chain = payload[0]["sequences"][0]["proteinChain"]
            self.assertEqual(binder_chain["sequence"], "ACD")
            self.assertEqual(
                Path(binder_chain["pairedMsaPath"]).read_text(),
                ">query\nACD\n",
            )
            self.assertEqual(
                Path(binder_chain["unpairedMsaPath"]).read_text(),
                ">query\nACD\n",
            )

    def test_miniprotein_binder_msa_is_single_sequence_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            config = ProtenixMsaConfig()

            pair = resolve_binder_msa_pair(
                campaign_dir,
                binder_sequence="ACDE",
                binder_scaffold="miniprotein",
                config=config,
            )
            second = resolve_binder_msa_pair(
                campaign_dir,
                binder_sequence="ACDE",
                binder_scaffold="miniprotein",
                config=config,
            )

            assert pair is not None
            assert second is not None
            self.assertEqual(pair.non_pairing, ">query\nACDE\n")
            self.assertEqual(pair.pairing, ">query\nACDE\n")
            self.assertEqual(second.cache_dir, pair.cache_dir)
            cache_dir = binder_msa_cache_dir(
                campaign_dir,
                sequence="ACDE",
                scaffold="miniprotein",
                config=config,
            )
            metadata = json.loads((cache_dir / "metadata.json").read_text())
            self.assertEqual(metadata["source"], "single_sequence")
            self.assertEqual(metadata["binder_scaffold"], "miniprotein")

    def test_vhh_auto_binder_msa_requires_server_or_cached_representative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                msa_module,
                "number_vhh_sequence",
                side_effect=_fake_vhh_segmentation,
            ):
                with self.assertRaisesRegex(ValueError, "MSA server"):
                    resolve_binder_msa_pair(
                        Path(tmpdir),
                        binder_sequence="ACDEFGHIKLMNPQ",
                        binder_scaffold="vhh",
                        config=ProtenixMsaConfig(),
                    )

    def test_vhh_binder_msas_share_representative_and_query_swap_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            campaign_dir = Path(tmpdir) / "campaign"
            config = ProtenixMsaConfig(server_url="https://msa.example")
            seq_a = "ACDEFGHIKLMNPQ"
            seq_b = "LMNPQRSTVWYACD"
            calls: list[str] = []

            def fetcher(sequence: str, _config: ProtenixMsaConfig) -> MsaPair:
                calls.append(sequence)
                return MsaPair(
                    pairing=f">query\n{sequence}\n",
                    non_pairing=f">query\n{sequence}\n>hit\n{sequence[::-1]}\n",
                    source="server",
                )

            with patch.object(
                msa_module,
                "number_vhh_sequence",
                side_effect=_fake_vhh_segmentation,
            ):
                pairs = resolve_binder_msa_pairs(
                    campaign_dir,
                    binders=((seq_a, "vhh"), (seq_b, "vhh")),
                    config=config,
                    fetcher=fetcher,
                )
                second = resolve_binder_msa_pairs(
                    campaign_dir,
                    binders=((seq_a, "vhh"), (seq_b, "vhh")),
                    config=config,
                    fetcher=fetcher,
                )

            self.assertEqual(calls, [seq_a])
            assert pairs[0] is not None
            assert pairs[1] is not None
            assert second[0] is not None
            assert second[1] is not None
            self.assertEqual(pairs[0].non_pairing.splitlines()[:2], [">query", seq_a])
            self.assertEqual(pairs[1].non_pairing.splitlines()[:2], [">query", seq_b])
            self.assertIn(f">hit\n{seq_a[::-1]}", pairs[1].non_pairing)
            self.assertEqual(second[1].cache_dir, pairs[1].cache_dir)
            self.assertEqual(pairs[1].metadata["source"], "vhh_template")
            self.assertTrue(
                (
                    campaign_dir
                    / "validation"
                    / "protenix_v2"
                    / "msa_cache"
                    / "binder"
                    / "vhh_template"
                ).exists()
            )

    def test_vhh_single_sequence_binder_msa_is_not_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(NotImplementedError, "grouped/template"):
                resolve_binder_msa_pair(
                    Path(tmpdir),
                    binder_sequence="QVQLQQSGA",
                    binder_scaffold="vhh",
                    config=ProtenixMsaConfig(binder_mode="single_sequence"),
                )


def _fake_vhh_segmentation(sequence: str, numbering_scheme: str = "imgt"):
    seq = msa_module.normalize_sequence(sequence)
    return msa_module.VhhSegmentation(
        sequence=seq,
        numbering_scheme=numbering_scheme,
        chain_class="vhh",
        chain_type="H",
        fr1=seq[0:2],
        cdr1=seq[2:4],
        fr2=seq[4:6],
        cdr2=seq[6:8],
        fr3=seq[8:10],
        cdr3=seq[10:12],
        fr4=seq[12:],
        cdr1_register="H27,H28",
        cdr2_register="H56,H57",
        cdr3_register="H105,H106",
        fr1_length=2,
        fr2_length=2,
        fr3_length=2,
        fr4_length=len(seq[12:]),
        cdr1_length=2,
        cdr2_length=2,
        cdr3_length=2,
        total_binder_length=len(seq),
        framework_hash=msa_module.framework_hash(
            seq[0:2],
            seq[4:6],
            seq[8:10],
            seq[12:],
        ),
    )


if __name__ == "__main__":
    unittest.main()
