from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile
import time
from typing import Any, Callable, Literal, Sequence

from esmfold2_pipeline.artifacts import (
    write_bytes_atomic,
    write_json_atomic,
    write_text_atomic,
)
from esmfold2_pipeline.artifact_layout import validator_msa_cache_dir

DEFAULT_MSA_CACHE_DIR = validator_msa_cache_dir("protenix-v2").as_posix()
DEFAULT_MSA_PAIRING_STRATEGY = "greedy"
DEFAULT_MSA_USER_AGENT = "esmfold2-pipeline/0.1"

MsaMode = Literal["none", "provided", "server"]
BinderMsaMode = Literal["auto", "none", "single_sequence"]
MsaPairingStrategy = Literal["greedy", "query_only", "copy_non_pairing"]
MsaFetcher = Callable[[str, "ProtenixMsaConfig"], "MsaPair"]
VHH_TEMPLATE_FRAMEWORK_MODE = "lengths_only"


@dataclass(frozen=True)
class MsaPair:
    pairing: str | None
    non_pairing: str
    source: str
    cache_dir: Path | None = None
    metadata: dict[str, Any] | None = None


class VhhNumberingError(RuntimeError):
    pass


@dataclass(frozen=True)
class VhhSegmentation:
    sequence: str
    numbering_scheme: str
    chain_class: str
    chain_type: str
    fr1: str
    cdr1: str
    fr2: str
    cdr2: str
    fr3: str
    cdr3: str
    fr4: str
    cdr1_register: str
    cdr2_register: str
    cdr3_register: str
    fr1_length: int
    fr2_length: int
    fr3_length: int
    fr4_length: int
    cdr1_length: int
    cdr2_length: int
    cdr3_length: int
    total_binder_length: int
    framework_hash: str


@dataclass(frozen=True)
class ProtenixMsaConfig:
    target_mode: MsaMode = "none"
    binder_mode: BinderMsaMode = "auto"
    target_msa_dir: Path | None = None
    target_msa_map_csv: Path | None = None
    server_url: str | None = None
    cache_root: Path | None = None
    pairing_strategy: MsaPairingStrategy = DEFAULT_MSA_PAIRING_STRATEGY
    user_agent: str = DEFAULT_MSA_USER_AGENT
    max_submit_retries: int = 6
    max_status_polls: int = 120
    status_poll_interval_seconds: float = 10.0
    request_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.target_mode not in {"none", "provided", "server"}:
            raise ValueError("target MSA mode must be one of: none, provided, server")
        if self.binder_mode not in {"auto", "none", "single_sequence"}:
            raise ValueError(
                "binder MSA mode must be one of: auto, none, single_sequence"
            )
        if self.pairing_strategy not in {"greedy", "query_only", "copy_non_pairing"}:
            raise ValueError(
                "MSA pairing strategy must be one of: greedy, query_only, copy_non_pairing"
            )
        if self.target_mode == "provided" and not (
            self.target_msa_dir or self.target_msa_map_csv
        ):
            raise ValueError(
                "provided target MSA mode requires target_msa_dir or target_msa_map_csv"
            )
        if self.target_mode == "server" and not self.server_url:
            raise ValueError("server target MSA mode requires server_url")
        if self.max_submit_retries <= 0:
            raise ValueError("max_submit_retries must be positive")
        if self.max_status_polls <= 0:
            raise ValueError("max_status_polls must be positive")
        if self.status_poll_interval_seconds < 0:
            raise ValueError("status_poll_interval_seconds must be non-negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")


def resolve_target_msa_pairs(
    campaign_dir: str | Path,
    *,
    target_sequences: tuple[str, ...],
    target_labels: tuple[str, ...],
    target_name: str | None,
    config: ProtenixMsaConfig,
    fetcher: MsaFetcher | None = None,
) -> tuple[MsaPair | None, ...]:
    """Resolve target-chain MSA pairs and populate the campaign-local cache."""

    if config.target_mode == "none":
        return tuple(None for _sequence in target_sequences)
    if len(target_labels) != len(target_sequences):
        raise ValueError("target_labels must match target_sequences")

    root = Path(campaign_dir)
    resolved: list[MsaPair | None] = []
    for sequence, label in zip(target_sequences, target_labels):
        normalized = normalize_sequence(sequence)
        if config.target_mode == "provided":
            pair = _provided_target_msa(
                normalized,
                label=label,
                target_name=target_name,
                config=config,
            )
            resolved.append(
                _write_cache_entry(
                    root,
                    sequence=normalized,
                    label=label,
                    config=config,
                    pair=pair,
                    source="provided",
                )
            )
            continue

        cache_dir = target_msa_cache_dir(root, sequence=normalized, config=config)
        cached = read_cached_msa_pair(cache_dir, sequence=normalized, config=config)
        if cached is not None:
            resolved.append(cached)
            continue

        fetched = (fetcher or fetch_msa_colabfold)(normalized, config)
        resolved.append(
            _write_cache_entry(
                root,
                sequence=normalized,
                label=label,
                config=config,
                pair=fetched,
                source="server",
            )
        )
    return tuple(resolved)


def resolve_binder_msa_pair(
    campaign_dir: str | Path,
    *,
    binder_sequence: str,
    binder_scaffold: str | None,
    config: ProtenixMsaConfig,
    fetcher: MsaFetcher | None = None,
) -> MsaPair | None:
    """Resolve the binder MSA for one validation task."""

    return resolve_binder_msa_pairs(
        campaign_dir,
        binders=((binder_sequence, binder_scaffold),),
        config=config,
        fetcher=fetcher,
    )[0]


def resolve_binder_msa_pairs(
    campaign_dir: str | Path,
    *,
    binders: Sequence[tuple[str, str | None]],
    config: ProtenixMsaConfig,
    fetcher: MsaFetcher | None = None,
) -> tuple[MsaPair | None, ...]:
    """Resolve binder MSAs as a batch so VHH template groups share one MSA."""

    root = Path(campaign_dir)
    resolved: list[MsaPair | None] = [None for _binder in binders]
    vhh_records: list[dict[str, Any]] = []

    for index, (binder_sequence, binder_scaffold) in enumerate(binders):
        pair = _resolve_non_vhh_binder_msa(
            root,
            binder_sequence=binder_sequence,
            binder_scaffold=binder_scaffold,
            config=config,
            vhh_records=vhh_records,
            index=index,
        )
        if pair is not _DEFERRED_VHH_MSA:
            resolved[index] = pair

    if vhh_records:
        for index, pair in _resolve_vhh_grouped_msa_pairs(
            root,
            records=vhh_records,
            config=config,
            fetcher=fetcher,
        ).items():
            resolved[index] = pair
    return tuple(resolved)


_DEFERRED_VHH_MSA = object()


def _resolve_non_vhh_binder_msa(
    campaign_dir: Path,
    *,
    binder_sequence: str,
    binder_scaffold: str | None,
    config: ProtenixMsaConfig,
    vhh_records: list[dict[str, Any]],
    index: int,
) -> MsaPair | None | object:
    mode = config.binder_mode
    scaffold = str(binder_scaffold or "").strip().lower()
    sequence = normalize_sequence(binder_sequence)
    if mode == "none":
        return None
    if mode == "auto":
        if scaffold in {"", "miniprotein", "minibinder"}:
            mode = "single_sequence"
        elif scaffold in {"vhh", "scfv"}:
            if scaffold == "scfv":
                raise NotImplementedError(
                    "scfv binder MSAs require VH/VL grouped/template MSA support"
                )
            vhh_records.append(
                {
                    "index": index,
                    "binder_sequence": sequence,
                    "analysis": analyze_vhh_sequence(sequence),
                }
            )
            return _DEFERRED_VHH_MSA
        else:
            raise NotImplementedError(
                f"automatic binder MSA mode does not support scaffold={scaffold or 'unknown'}"
            )
    if mode == "single_sequence":
        if scaffold in {"vhh", "scfv"}:
            raise NotImplementedError(
                f"{scaffold} binder MSAs require grouped/template MSA support"
            )
        return _single_sequence_binder_msa(
            campaign_dir,
            sequence=sequence,
            scaffold=scaffold or "miniprotein",
            config=config,
        )
    raise ValueError(f"unsupported binder MSA mode: {mode}")


def target_msa_cache_dir(
    campaign_dir: str | Path,
    *,
    sequence: str,
    config: ProtenixMsaConfig,
) -> Path:
    root = Path(campaign_dir)
    cache_root = config.cache_root or root / DEFAULT_MSA_CACHE_DIR
    return (
        cache_root
        / "target"
        / sha256_text(normalize_sequence(sequence))
        / msa_context_hash(config)
    )


def binder_msa_cache_dir(
    campaign_dir: str | Path,
    *,
    sequence: str,
    scaffold: str,
    config: ProtenixMsaConfig,
) -> Path:
    root = Path(campaign_dir)
    cache_root = config.cache_root or root / DEFAULT_MSA_CACHE_DIR
    scaffold_key = re.sub(r"[^a-z0-9_]+", "_", scaffold.strip().lower()).strip("_")
    return (
        cache_root
        / "binder"
        / f"{scaffold_key or 'binder'}_single_sequence"
        / sha256_text(normalize_sequence(sequence))
        / msa_context_hash(config)
    )


def vhh_template_group_cache_dir(
    campaign_dir: str | Path,
    *,
    template_key_hash: str,
    config: ProtenixMsaConfig,
) -> Path:
    root = Path(campaign_dir)
    cache_root = config.cache_root or root / DEFAULT_MSA_CACHE_DIR
    return (
        cache_root
        / "binder"
        / "vhh_template"
        / template_key_hash
        / msa_context_hash(config)
    )


def vhh_member_msa_cache_dir(
    campaign_dir: str | Path,
    *,
    sequence: str,
    template_key_hash: str,
    config: ProtenixMsaConfig,
) -> Path:
    return (
        vhh_template_group_cache_dir(
            campaign_dir,
            template_key_hash=template_key_hash,
            config=config,
        )
        / "members"
        / sha256_text(normalize_sequence(sequence))
    )


def read_cached_msa_pair(
    cache_dir: str | Path,
    *,
    sequence: str,
    config: ProtenixMsaConfig,
) -> MsaPair | None:
    root = Path(cache_dir)
    non_pairing_path = root / "non_pairing.a3m"
    if not non_pairing_path.exists() or non_pairing_path.stat().st_size <= 0:
        return None

    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    if metadata.get("sequence_sha256") != sha256_text(normalize_sequence(sequence)):
        return None
    if metadata.get("context_hash") != msa_context_hash(config):
        return None

    non_pairing = non_pairing_path.read_text()
    pairing_path = root / "pairing.a3m"
    pairing = pairing_path.read_text() if pairing_path.exists() else None
    if not pairing:
        query = extract_query_from_a3m(non_pairing)
        pairing = f">query\n{query}\n" if query else None
    return MsaPair(
        pairing=pairing,
        non_pairing=non_pairing,
        source=str(metadata.get("source") or metadata.get("target_mode") or "cache"),
        cache_dir=root,
        metadata=metadata,
    )


def single_sequence_msa_pair(sequence: str, *, source: str = "single_sequence") -> MsaPair:
    normalized = normalize_sequence(sequence)
    content = f">query\n{normalized}\n"
    return MsaPair(
        pairing=content,
        non_pairing=content,
        source=source,
        metadata={"sequence_length": len(normalized)},
    )


def fetch_msa_colabfold(sequence: str, config: ProtenixMsaConfig) -> MsaPair:
    """Fetch a target MSA from the ColabFold/MMseqs API and return A3M strings."""

    import requests

    host_url = str(config.server_url or "").rstrip("/")
    if not host_url:
        raise ValueError("MSA server URL is required")
    query_seq = normalize_sequence(sequence)
    query_fasta = f">query_0\n{query_seq}"
    headers = {"User-Agent": config.user_agent}

    ticket: dict[str, Any] | None = None
    for attempt in range(config.max_submit_retries):
        response = requests.post(
            f"{host_url}/ticket/msa",
            data={"q": query_fasta, "mode": "env"},
            headers=headers,
            timeout=config.request_timeout_seconds,
        )
        response.raise_for_status()
        ticket = response.json()
        status = str(ticket.get("status", "")).upper()
        if status not in {"UNKNOWN", "RATELIMIT"}:
            break
        time.sleep(min(60.0, 2.0**attempt + 3.0))

    if not ticket:
        raise RuntimeError("MSA submit failed without a response")
    status = str(ticket.get("status", "")).upper()
    if status in {"ERROR", "MAINTENANCE"}:
        raise RuntimeError(f"MSA submit failed with status={status}")
    ticket_id = ticket.get("id")
    if not ticket_id:
        raise RuntimeError(f"MSA submit did not return a ticket id: {ticket}")

    current = ticket
    for _poll in range(config.max_status_polls):
        status = str(current.get("status", "")).upper()
        if status == "COMPLETE":
            break
        if status in {"ERROR", "MAINTENANCE"}:
            raise RuntimeError(f"MSA job failed with status={status}")
        time.sleep(config.status_poll_interval_seconds)
        response = requests.get(
            f"{host_url}/ticket/{ticket_id}",
            headers=headers,
            timeout=config.request_timeout_seconds,
        )
        response.raise_for_status()
        current = response.json()
    else:
        raise TimeoutError(f"MSA polling timed out for ticket {ticket_id}")

    response = requests.get(
        f"{host_url}/result/download/{ticket_id}",
        headers=headers,
        timeout=max(config.request_timeout_seconds, 60.0),
    )
    response.raise_for_status()

    non_pairing = _msa_from_colabfold_tar(response.content, query_seq=query_seq)
    pairing = pairing_from_strategy(
        non_pairing,
        query_seq=query_seq,
        pairing_strategy=config.pairing_strategy,
    )
    return MsaPair(
        pairing=pairing,
        non_pairing=non_pairing,
        source="server",
        metadata={"ticket_id": ticket_id, "server_url": host_url},
    )


def write_msa_files_for_input(
    input_dir: str | Path,
    *,
    prefix: str,
    msa: MsaPair,
) -> dict[str, str]:
    root = Path(input_dir)
    paths: dict[str, str] = {}
    if msa.pairing:
        pairing_path = root / f"{prefix}_pairing.a3m"
        write_text_atomic(pairing_path, msa.pairing)
        paths["pairedMsaPath"] = str(pairing_path)
    if msa.non_pairing:
        non_pairing_path = root / f"{prefix}_non_pairing.a3m"
        write_text_atomic(non_pairing_path, msa.non_pairing)
        paths["unpairedMsaPath"] = str(non_pairing_path)
    return paths


def normalize_sequence(sequence: str) -> str:
    return re.sub(r"\s+", "", str(sequence or "")).upper()


def sequence_is_protein_like(sequence: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]+", normalize_sequence(sequence)))


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 16) -> str:
    return sha256_text(text)[:n]


def msa_context_hash(config: ProtenixMsaConfig) -> str:
    payload = {
        "target_mode": config.target_mode,
        "binder_mode": config.binder_mode,
        "server_url": (config.server_url or "").rstrip("/"),
        "pairing_strategy": config.pairing_strategy,
        "schema": 1,
    }
    return short_hash(json.dumps(payload, sort_keys=True), n=16)


def parse_fasta_string(content: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    header = ""
    for line in str(content or "").replace("\x00", "").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith(">"):
            header = text[1:].strip()
            parsed[header] = ""
        elif header:
            parsed[header] = parsed.get(header, "") + text
    return parsed


def rewrite_query_in_non_pairing_a3m(non_pairing: str, query_seq: str) -> str:
    """Swap the representative query for the member query in template-derived MSAs."""

    return normalize_a3m_non_pairing(non_pairing, query_seq=query_seq)


def extract_query_from_a3m(content: str) -> str | None:
    parsed = parse_fasta_string(content)
    for header, sequence in parsed.items():
        if header.lower().startswith("query"):
            return normalize_sequence(sequence)
    first = next(iter(parsed.values()), None)
    return normalize_sequence(first) if first else None


def normalize_a3m_non_pairing(content: str, *, query_seq: str) -> str:
    parsed = parse_fasta_string(content)
    query = normalize_sequence(query_seq)
    records: list[tuple[str, str]] = []
    for header, sequence in parsed.items():
        header_text = header.strip()
        seq_text = sequence.strip()
        if not header_text or not seq_text:
            continue
        if header_text.lower().startswith("query"):
            continue
        records.append((header_text, seq_text))
    records.sort(key=lambda item: item[0])

    lines = [">query", query]
    for header, sequence in records:
        lines.extend([f">{header}", sequence.rstrip()])
    return "\n".join(lines).strip() + "\n"


def pairing_from_strategy(
    non_pairing: str,
    *,
    query_seq: str,
    pairing_strategy: str,
) -> str:
    strategy = str(pairing_strategy or DEFAULT_MSA_PAIRING_STRATEGY).strip().lower()
    if strategy == "copy_non_pairing":
        return non_pairing
    if strategy not in {"greedy", "query_only"}:
        raise ValueError(
            "MSA pairing strategy must be one of: greedy, query_only, copy_non_pairing"
        )
    return f">query\n{normalize_sequence(query_seq)}\n"


def normalize_framework_mode(framework_mode: str = VHH_TEMPLATE_FRAMEWORK_MODE) -> str:
    mode = str(framework_mode or VHH_TEMPLATE_FRAMEWORK_MODE).strip().lower()
    if mode == "exact_frameworks":
        mode = "exact"
    if mode not in {"exact", "lengths_only"}:
        raise ValueError("framework_mode must be one of: exact, lengths_only")
    return mode


def framework_hash(fr1: str, fr2: str, fr3: str, fr4: str) -> str:
    return sha256_text("|".join([fr1, fr2, fr3, fr4]))


def _position_label(pos: Any) -> str:
    if hasattr(pos, "format"):
        try:
            return str(pos.format(chain_type=True, region=False))
        except TypeError:
            try:
                return str(pos.format())
            except Exception:
                pass
    return str(pos)


def _region_sequence_and_register(region_map: dict[Any, str]) -> tuple[str, str]:
    ordered = sorted(region_map.items(), key=lambda item: item[0])
    sequence = "".join(str(aa) for _pos, aa in ordered)
    register = ",".join(_position_label(pos) for pos, _aa in ordered)
    return sequence, register


def _extract_region_map(chain: Any, region_name: str) -> dict[Any, str]:
    regions = getattr(chain, "regions", None)
    if not isinstance(regions, dict):
        raise VhhNumberingError("Numbering backend did not expose chain.regions")
    region = regions.get(region_name)
    if not isinstance(region, dict) or not region:
        raise VhhNumberingError(
            f"Missing or empty region '{region_name}' in numbering output"
        )
    return region


def number_vhh_sequence(
    sequence: str,
    numbering_scheme: str = "imgt",
) -> VhhSegmentation:
    seq = normalize_sequence(sequence)
    if not seq:
        raise VhhNumberingError("Empty sequence")
    if not sequence_is_protein_like(seq):
        raise VhhNumberingError("Sequence is not protein-like")
    if str(numbering_scheme or "imgt").strip().lower() != "imgt":
        raise VhhNumberingError("Only IMGT numbering is supported")
    try:
        from abnumber import Chain  # type: ignore
    except Exception as exc:
        raise VhhNumberingError(
            "AbNumber is required for VHH numbering. Install abnumber, anarcii, and hmmer."
        ) from exc

    try:
        chain = Chain(
            seq,
            scheme="imgt",
            cdr_definition="imgt",
            allowed_species=None,
            use_anarcii=True,
        )
    except Exception as exc:
        raise VhhNumberingError(f"IMGT numbering failed: {exc}") from exc

    chain_type = str(getattr(chain, "chain_type", "") or "").strip().upper()
    if chain_type != "H":
        raise VhhNumberingError(
            f"Sequence numbered as non-heavy chain: chain_type={chain_type or 'unknown'}"
        )

    variable_seq = normalize_sequence(str(getattr(chain, "seq", "") or ""))
    tail_seq = normalize_sequence(str(getattr(chain, "tail", "") or ""))
    if tail_seq:
        raise VhhNumberingError("Sequence includes a non-empty constant/tail region")
    if variable_seq != seq:
        raise VhhNumberingError(
            "Numbered variable sequence does not exactly match input sequence"
        )

    fr1, _fr1_register = _region_sequence_and_register(_extract_region_map(chain, "FR1"))
    cdr1, cdr1_register = _region_sequence_and_register(
        _extract_region_map(chain, "CDR1")
    )
    fr2, _fr2_register = _region_sequence_and_register(_extract_region_map(chain, "FR2"))
    cdr2, cdr2_register = _region_sequence_and_register(
        _extract_region_map(chain, "CDR2")
    )
    fr3, _fr3_register = _region_sequence_and_register(_extract_region_map(chain, "FR3"))
    cdr3, cdr3_register = _region_sequence_and_register(
        _extract_region_map(chain, "CDR3")
    )
    fr4, _fr4_register = _region_sequence_and_register(_extract_region_map(chain, "FR4"))

    rebuilt = "".join([fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4])
    if rebuilt != seq:
        raise VhhNumberingError(
            "FR/CDR segmentation did not reconstruct the input sequence exactly"
        )

    return VhhSegmentation(
        sequence=seq,
        numbering_scheme="imgt",
        chain_class="vhh",
        chain_type=chain_type,
        fr1=fr1,
        cdr1=cdr1,
        fr2=fr2,
        cdr2=cdr2,
        fr3=fr3,
        cdr3=cdr3,
        fr4=fr4,
        cdr1_register=cdr1_register,
        cdr2_register=cdr2_register,
        cdr3_register=cdr3_register,
        fr1_length=len(fr1),
        fr2_length=len(fr2),
        fr3_length=len(fr3),
        fr4_length=len(fr4),
        cdr1_length=len(cdr1),
        cdr2_length=len(cdr2),
        cdr3_length=len(cdr3),
        total_binder_length=len(seq),
        framework_hash=framework_hash(fr1, fr2, fr3, fr4),
    )


def build_canonical_template_key(
    segmentation: VhhSegmentation,
    framework_mode: str = VHH_TEMPLATE_FRAMEWORK_MODE,
) -> dict[str, Any]:
    framework_mode_n = normalize_framework_mode(framework_mode)
    key: dict[str, Any] = {
        "numbering_scheme": segmentation.numbering_scheme,
        "chain_class": segmentation.chain_class,
        "cdr1_register": segmentation.cdr1_register,
        "cdr2_register": segmentation.cdr2_register,
        "cdr3_register": segmentation.cdr3_register,
    }
    if framework_mode_n == "lengths_only":
        key.update(
            {
                "fr1_length": segmentation.fr1_length,
                "fr2_length": segmentation.fr2_length,
                "fr3_length": segmentation.fr3_length,
                "fr4_length": segmentation.fr4_length,
            }
        )
    else:
        key.update(
            {
                "fr1": segmentation.fr1,
                "fr2": segmentation.fr2,
                "fr3": segmentation.fr3,
                "fr4": segmentation.fr4,
            }
        )
    return key


def canonical_template_key_json(
    segmentation: VhhSegmentation,
    framework_mode: str = VHH_TEMPLATE_FRAMEWORK_MODE,
) -> str:
    return json.dumps(
        build_canonical_template_key(segmentation, framework_mode=framework_mode),
        sort_keys=True,
        separators=(",", ":"),
    )


def analyze_vhh_sequence(
    sequence: str,
    numbering_scheme: str = "imgt",
) -> dict[str, Any]:
    seg = number_vhh_sequence(sequence, numbering_scheme=numbering_scheme)
    canonical_key_json = canonical_template_key_json(seg, framework_mode="exact")
    lengths_key_json = canonical_template_key_json(
        seg,
        framework_mode=VHH_TEMPLATE_FRAMEWORK_MODE,
    )
    return {
        "binder_sequence": seg.sequence,
        "binder_sequence_sha256": sha256_text(seg.sequence),
        "numbering_scheme": seg.numbering_scheme,
        "chain_class": seg.chain_class,
        "chain_type": seg.chain_type,
        "fr1": seg.fr1,
        "cdr1": seg.cdr1,
        "fr2": seg.fr2,
        "cdr2": seg.cdr2,
        "fr3": seg.fr3,
        "cdr3": seg.cdr3,
        "fr4": seg.fr4,
        "fr1_length": seg.fr1_length,
        "fr2_length": seg.fr2_length,
        "fr3_length": seg.fr3_length,
        "fr4_length": seg.fr4_length,
        "cdr1_length": seg.cdr1_length,
        "cdr2_length": seg.cdr2_length,
        "cdr3_length": seg.cdr3_length,
        "cdr1_register": seg.cdr1_register,
        "cdr2_register": seg.cdr2_register,
        "cdr3_register": seg.cdr3_register,
        "total_binder_length": seg.total_binder_length,
        "framework_hash": seg.framework_hash,
        "canonical_template_key_json": canonical_key_json,
        "canonical_template_key_hash": short_hash(canonical_key_json, n=16),
        "lengths_only_template_key_json": lengths_key_json,
        "lengths_only_template_key_hash": short_hash(lengths_key_json, n=16),
    }


def _provided_target_msa(
    sequence: str,
    *,
    label: str,
    target_name: str | None,
    config: ProtenixMsaConfig,
) -> MsaPair:
    if config.target_msa_dir is not None:
        if config.target_msa_map_csv is not None:
            raise ValueError("use target_msa_dir or target_msa_map_csv, not both")
        pair = _read_msa_pair(msa_dir=config.target_msa_dir)
    elif config.target_msa_map_csv is not None:
        mapping = _parse_target_msa_map_csv(config.target_msa_map_csv)
        pair = _target_msa_lookup(
            mapping,
            target_name=target_name,
            label=label,
            target_sequence=sequence,
        )
        if pair is None:
            raise FileNotFoundError(
                f"target MSA map has no entry for target={target_name or ''} "
                f"label={label} sequence_sha256={sha256_text(sequence)}"
            )
    else:
        raise ValueError("provided target MSA mode requires an MSA path")

    if not pair.non_pairing:
        raise ValueError("provided target MSA must include non_pairing.a3m")
    return MsaPair(
        pairing=pair.pairing,
        non_pairing=normalize_a3m_non_pairing(pair.non_pairing, query_seq=sequence),
        source="provided",
        metadata=pair.metadata,
    )


def _read_msa_pair(
    *,
    msa_dir: Path | None = None,
    pairing_path: Path | None = None,
    non_pairing_path: Path | None = None,
) -> MsaPair:
    if msa_dir is not None:
        pairing_path = msa_dir / "pairing.a3m"
        non_pairing_path = msa_dir / "non_pairing.a3m"

    pairing = pairing_path.read_text() if pairing_path and pairing_path.exists() else None
    non_pairing = (
        non_pairing_path.read_text()
        if non_pairing_path and non_pairing_path.exists()
        else None
    )
    if not non_pairing:
        raise FileNotFoundError("MSA directory or mapping must provide non_pairing.a3m")
    if not pairing:
        query = extract_query_from_a3m(non_pairing)
        pairing = f">query\n{query}\n" if query else None
    return MsaPair(
        pairing=pairing,
        non_pairing=non_pairing,
        source="provided",
        metadata={
            "pairing_path": str(pairing_path) if pairing_path else None,
            "non_pairing_path": str(non_pairing_path) if non_pairing_path else None,
        },
    )


def _parse_target_msa_map_csv(path: Path) -> dict[str, MsaPair]:
    mapping: dict[str, MsaPair] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {
                str(key).strip().lower(): str(value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            target_name = (
                normalized.get("target_name")
                or normalized.get("name")
                or normalized.get("target")
            )
            label = normalized.get("chain") or normalized.get("label")
            sequence = normalize_sequence(
                normalized.get("target_sequence") or normalized.get("sequence") or ""
            )
            msa_dir = _optional_path(
                normalized.get("msa_dir") or normalized.get("msa_path"),
                base=path.parent,
            )
            pairing_path = _optional_path(normalized.get("pairing_path"), base=path.parent)
            non_pairing_path = _optional_path(
                normalized.get("non_pairing_path")
                or normalized.get("unpaired_msa_path"),
                base=path.parent,
            )
            pair = _read_msa_pair(
                msa_dir=msa_dir,
                pairing_path=pairing_path,
                non_pairing_path=non_pairing_path,
            )
            if target_name and sequence:
                mapping[f"name_seq:{_name_key(target_name)}|{sequence}"] = pair
            if target_name:
                mapping[f"name:{_name_key(target_name)}"] = pair
            if label and sequence:
                mapping[f"label_seq:{_name_key(label)}|{sequence}"] = pair
            if label:
                mapping[f"label:{_name_key(label)}"] = pair
            if sequence:
                mapping[f"seq:{sequence}"] = pair
    return mapping


def _target_msa_lookup(
    mapping: dict[str, MsaPair],
    *,
    target_name: str | None,
    label: str,
    target_sequence: str,
) -> MsaPair | None:
    sequence = normalize_sequence(target_sequence)
    keys = [
        f"name_seq:{_name_key(target_name)}|{sequence}" if target_name else "",
        f"label_seq:{_name_key(label)}|{sequence}" if label else "",
        f"seq:{sequence}",
        f"name:{_name_key(target_name)}" if target_name else "",
        f"label:{_name_key(label)}" if label else "",
    ]
    for key in keys:
        if key and key in mapping:
            return mapping[key]
    return None


def _resolve_vhh_grouped_msa_pairs(
    campaign_dir: Path,
    *,
    records: Sequence[dict[str, Any]],
    config: ProtenixMsaConfig,
    fetcher: MsaFetcher | None,
) -> dict[int, MsaPair]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        analysis = dict(record["analysis"])
        key_json = str(analysis["lengths_only_template_key_json"])
        groups.setdefault(key_json, []).append(dict(record))

    resolved: dict[int, MsaPair] = {}
    for key_json, group_records in groups.items():
        group_records = sorted(
            group_records,
            key=lambda item: (int(item["index"]), str(item["binder_sequence"])),
        )
        representative = _vhh_representative_record(
            campaign_dir,
            template_key_hash=short_hash(key_json, n=16),
            config=config,
            fallback=group_records[0],
        )
        representative_pair = _vhh_representative_msa_pair(
            campaign_dir,
            representative=representative,
            template_key_json=key_json,
            config=config,
            fetcher=fetcher,
        )
        for record in group_records:
            sequence = str(record["binder_sequence"])
            analysis = dict(record["analysis"])
            template_key_hash = str(analysis["lengths_only_template_key_hash"])
            member_cache = vhh_member_msa_cache_dir(
                campaign_dir,
                sequence=sequence,
                template_key_hash=template_key_hash,
                config=config,
            )
            cached = read_cached_msa_pair(
                member_cache,
                sequence=sequence,
                config=config,
            )
            if cached is not None:
                resolved[int(record["index"])] = cached
                continue
            resolved[int(record["index"])] = _write_vhh_member_msa_pair(
                member_cache,
                sequence=sequence,
                config=config,
                representative_pair=representative_pair,
                representative=representative,
                analysis=analysis,
            )
    return resolved


def _vhh_representative_record(
    campaign_dir: Path,
    *,
    template_key_hash: str,
    config: ProtenixMsaConfig,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    metadata_path = (
        vhh_template_group_cache_dir(
            campaign_dir,
            template_key_hash=template_key_hash,
            config=config,
        )
        / "representative"
        / "metadata.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
    sequence = normalize_sequence(str(metadata.get("sequence") or ""))
    analysis = metadata.get("vhh_analysis")
    if sequence and isinstance(analysis, dict):
        return {
            "index": -1,
            "binder_sequence": sequence,
            "analysis": analysis,
        }
    return fallback


def _vhh_representative_msa_pair(
    campaign_dir: Path,
    *,
    representative: dict[str, Any],
    template_key_json: str,
    config: ProtenixMsaConfig,
    fetcher: MsaFetcher | None,
) -> MsaPair:
    sequence = str(representative["binder_sequence"])
    analysis = dict(representative["analysis"])
    template_key_hash = str(analysis["lengths_only_template_key_hash"])
    representative_cache = (
        vhh_template_group_cache_dir(
            campaign_dir,
            template_key_hash=template_key_hash,
            config=config,
        )
        / "representative"
    )
    cached = read_cached_msa_pair(
        representative_cache,
        sequence=sequence,
        config=config,
    )
    if cached is not None:
        return cached

    if fetcher is None and not config.server_url:
        raise ValueError(
            "VHH binder MSA grouping requires an MSA server or an existing cached "
            f"representative MSA for template {template_key_hash}"
        )
    fetched = (fetcher or fetch_msa_colabfold)(sequence, config)
    metadata = {
        "sequence": sequence,
        "sequence_sha256": sha256_text(sequence),
        "sequence_length": len(sequence),
        "binder_scaffold": "vhh",
        "binder_mode": config.binder_mode,
        "source": fetched.source or "server",
        "server_url": (config.server_url or "").rstrip("/"),
        "pairing_strategy": config.pairing_strategy,
        "context_hash": msa_context_hash(config),
        "cached_at": _utc_now_text(),
        "template_key_json": template_key_json,
        "template_key_hash": template_key_hash,
        "template_grouping_mode": VHH_TEMPLATE_FRAMEWORK_MODE,
        "template_role": "representative",
        "vhh_analysis": analysis,
    }
    if fetched.metadata:
        metadata["source_metadata"] = fetched.metadata
    return _write_msa_pair_cache(
        representative_cache,
        sequence=sequence,
        config=config,
        pair=fetched,
        source="vhh_template_representative",
        metadata=metadata,
    )


def _write_vhh_member_msa_pair(
    cache_dir: Path,
    *,
    sequence: str,
    config: ProtenixMsaConfig,
    representative_pair: MsaPair,
    representative: dict[str, Any],
    analysis: dict[str, Any],
) -> MsaPair:
    normalized = normalize_sequence(sequence)
    representative_sequence = str(representative["binder_sequence"])
    template_key_hash = str(analysis["lengths_only_template_key_hash"])
    non_pairing = rewrite_query_in_non_pairing_a3m(
        representative_pair.non_pairing,
        normalized,
    )
    pairing = pairing_from_strategy(
        non_pairing,
        query_seq=normalized,
        pairing_strategy=config.pairing_strategy,
    )
    metadata = {
        "sequence": normalized,
        "sequence_sha256": sha256_text(normalized),
        "sequence_length": len(normalized),
        "binder_scaffold": "vhh",
        "binder_mode": config.binder_mode,
        "source": "vhh_template",
        "server_url": (config.server_url or "").rstrip("/"),
        "pairing_strategy": config.pairing_strategy,
        "context_hash": msa_context_hash(config),
        "cached_at": _utc_now_text(),
        "derived_from_template": True,
        "template_key_json": str(analysis["lengths_only_template_key_json"]),
        "template_key_hash": template_key_hash,
        "template_grouping_mode": VHH_TEMPLATE_FRAMEWORK_MODE,
        "template_role": "member",
        "representative_sequence_sha256": sha256_text(representative_sequence),
        "representative_cache_dir": (
            str(representative_pair.cache_dir) if representative_pair.cache_dir else None
        ),
        "vhh_analysis": analysis,
    }
    write_text_atomic(cache_dir / "non_pairing.a3m", non_pairing)
    write_text_atomic(cache_dir / "pairing.a3m", pairing)
    write_json_atomic(cache_dir / "metadata.json", metadata)
    return MsaPair(
        pairing=pairing,
        non_pairing=non_pairing,
        source="vhh_template",
        cache_dir=cache_dir,
        metadata=metadata,
    )


def _single_sequence_binder_msa(
    campaign_dir: Path,
    *,
    sequence: str,
    scaffold: str,
    config: ProtenixMsaConfig,
) -> MsaPair:
    normalized = normalize_sequence(sequence)
    cache_dir = binder_msa_cache_dir(
        campaign_dir,
        sequence=normalized,
        scaffold=scaffold,
        config=config,
    )
    cached = read_cached_msa_pair(cache_dir, sequence=normalized, config=config)
    if cached is not None:
        return cached

    pair = single_sequence_msa_pair(normalized)
    metadata = {
        "sequence_sha256": sha256_text(normalized),
        "sequence_length": len(normalized),
        "binder_scaffold": scaffold,
        "binder_mode": config.binder_mode,
        "source": "single_sequence",
        "server_url": "",
        "pairing_strategy": config.pairing_strategy,
        "context_hash": msa_context_hash(config),
        "cached_at": _utc_now_text(),
    }
    write_text_atomic(cache_dir / "non_pairing.a3m", pair.non_pairing)
    write_text_atomic(cache_dir / "pairing.a3m", pair.pairing or pair.non_pairing)
    write_json_atomic(cache_dir / "metadata.json", metadata)
    return MsaPair(
        pairing=pair.pairing,
        non_pairing=pair.non_pairing,
        source="single_sequence",
        cache_dir=cache_dir,
        metadata=metadata,
    )


def _write_msa_pair_cache(
    cache_dir: Path,
    *,
    sequence: str,
    config: ProtenixMsaConfig,
    pair: MsaPair,
    source: str,
    metadata: dict[str, Any],
) -> MsaPair:
    normalized = normalize_sequence(sequence)
    non_pairing = normalize_a3m_non_pairing(pair.non_pairing, query_seq=normalized)
    pairing = pair.pairing or pairing_from_strategy(
        non_pairing,
        query_seq=normalized,
        pairing_strategy=config.pairing_strategy,
    )
    write_text_atomic(cache_dir / "non_pairing.a3m", non_pairing)
    if pairing:
        write_text_atomic(cache_dir / "pairing.a3m", pairing)
    write_json_atomic(cache_dir / "metadata.json", metadata)
    return MsaPair(
        pairing=pairing,
        non_pairing=non_pairing,
        source=source,
        cache_dir=cache_dir,
        metadata=metadata,
    )


def _write_cache_entry(
    campaign_dir: Path,
    *,
    sequence: str,
    label: str,
    config: ProtenixMsaConfig,
    pair: MsaPair,
    source: str,
) -> MsaPair:
    normalized = normalize_sequence(sequence)
    cache_dir = target_msa_cache_dir(campaign_dir, sequence=normalized, config=config)
    metadata = {
        "sequence_sha256": sha256_text(normalized),
        "sequence_length": len(normalized),
        "target_label": label,
        "target_mode": config.target_mode,
        "source": source,
        "server_url": (config.server_url or "").rstrip("/"),
        "pairing_strategy": config.pairing_strategy,
        "context_hash": msa_context_hash(config),
        "cached_at": _utc_now_text(),
    }
    if pair.metadata:
        metadata["source_metadata"] = pair.metadata

    non_pairing = normalize_a3m_non_pairing(pair.non_pairing, query_seq=normalized)
    pairing = pair.pairing or pairing_from_strategy(
        non_pairing,
        query_seq=normalized,
        pairing_strategy=config.pairing_strategy,
    )
    write_text_atomic(cache_dir / "non_pairing.a3m", non_pairing)
    if pairing:
        write_text_atomic(cache_dir / "pairing.a3m", pairing)
    write_json_atomic(cache_dir / "metadata.json", metadata)
    return MsaPair(
        pairing=pairing,
        non_pairing=non_pairing,
        source=source,
        cache_dir=cache_dir,
        metadata=metadata,
    )


def _msa_from_colabfold_tar(content: bytes, *, query_seq: str) -> str:
    records: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "msa.tar.gz"
        write_bytes_atomic(tar_path, content)
        with tarfile.open(tar_path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name not in {
                    "bfd.mgnify30.metaeuk30.smag30.a3m",
                    "uniref.a3m",
                }:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                text = handle.read().decode("utf-8", errors="ignore")
                parsed = parse_fasta_string(text)
                for header, sequence in parsed.items():
                    if header.startswith("query_") or header.lower().startswith("query"):
                        continue
                    records.append((header, sequence))

    lines = [">query", normalize_sequence(query_seq)]
    for header, sequence in sorted(records, key=lambda item: item[0]):
        lines.extend([f">{header}", sequence])
    return normalize_a3m_non_pairing("\n".join(lines) + "\n", query_seq=query_seq)


def _optional_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _name_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(value or "").strip().lower())


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )
