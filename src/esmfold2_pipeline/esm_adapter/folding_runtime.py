from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string
from typing import Any

import numpy as np

from esmfold2_pipeline.esm_adapter.imports import _prepend_sys_path


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}
_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits
_CCD_LOADED = False


TARGET_SEQUENCES = {
    "cd45": (
        "GSPGEPQIIFCRSEAAHQGVITWNPPQRSFHNFTLCYIKETEKDCLNLDKNLIKYDLQNLKPY"
        "TKYVLSLHAYIIAKVQRNGSAAMCHFTTKSAPPSQVWNMTVSMTSDNSMHVKCRPPRDRNGPHER"
        "YHLEVEAGNTLVRNESHKNCDFRVKDLQYSTDYTFKAYFHNGDYPGEPFILHHSTSY"
    ),
    "ctla4": (
        "MHVAQPAVVLASSRGIASFVCEYASPGKATEVRVTVLRQADSQVTEVCAATYMMGNELTFLDDS"
        "ICTGTSSGNQVNLTIQGLRAMDTGLYICKVELMYPPPYYLGIGNGTQIYVIDPE"
    ),
    "egfr": (
        "RKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTV"
        "KEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVII"
        "SGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCV"
    ),
    "pd-l1": (
        "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSY"
        "RQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
    ),
    "pdgfr": (
        "GFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYIC"
        "KTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESG"
        "RLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE"
    ),
}


@dataclass(frozen=True)
class ESMFoldingRuntime:
    torch: Any
    F: Any
    optim: Any
    seed_context: Any
    ESMCTokenizer: Any
    ESMCForMaskedLM: Any
    ESMFold2ExperimentalModel: Any
    CUE_AVAILABLE: bool
    TOKENS: list[str]
    ELEMENTS: list[str]
    PROTEIN_3TO1: dict[str, str]
    ProteinInput: Any
    StructurePredictionInput: Any
    load_ccd: Any
    prepare_esmfold2_input: Any
    ProteinChain: Any
    ProteinComplex: Any
    biotite_structure: Any
    MOL_TYPE_NONPOLYMER: int
    TARGET_SEQUENCES: dict[str, str]
    COMPILE: bool = False

    def prepare_esmfold2_tensors(
        self,
        input,
        max_tokens: int | None = None,
        max_atoms: int | None = None,
        max_seqs: int = 16384,
        pad_to_max_seqs: bool = False,
        seed: int | None = None,
        use_vectorized_msa_assembly: bool = True,
    ) -> dict[str, Any]:
        del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
        _ensure_ccd_loaded(self.load_ccd)
        features, _metadata = self.prepare_esmfold2_input(input, seed=seed)
        if max_atoms is not None:
            for key, dim in _ATOM_FEATURE_DIMS.items():
                if key in features:
                    features[key] = _resize_tensor(
                        self.torch,
                        features[key],
                        dim=dim,
                        size=max_atoms,
                    )
        return features

    def to_atom_array(
        self,
        coords: np.ndarray,
        atom_to_token: np.ndarray,
        res_type: np.ndarray,
        residue_index: np.ndarray,
        asym_id: np.ndarray,
        mol_type: np.ndarray,
        ref_atom_name_chars: np.ndarray,
        ref_element: np.ndarray,
        atom_attention_mask: np.ndarray,
        plddt_per_atom: np.ndarray | None = None,
    ):
        atoms = []
        for atom_index, (
            atom_coord,
            token_index,
            atom_name_chars,
            element_index,
            is_not_pad,
        ) in enumerate(
            zip(
                coords,
                atom_to_token,
                ref_atom_name_chars,
                ref_element,
                atom_attention_mask,
            )
        ):
            if not is_not_pad:
                continue
            token_index = int(token_index)
            atoms.append(
                self.biotite_structure.Atom(
                    coord=atom_coord,
                    chain_id=_asym_id_to_chain_label(int(asym_id[token_index])),
                    res_id=int(residue_index[token_index]) + 1,
                    res_name=self.TOKENS[int(res_type[token_index])],
                    atom_name="".join(
                        chr(int(char) + 32) for char in atom_name_chars if char != 0
                    ),
                    element=self.ELEMENTS[int(element_index)],
                    ins_code=" ",
                    hetero=mol_type[token_index] == self.MOL_TYPE_NONPOLYMER,
                    b_factor=(
                        float(plddt_per_atom[atom_index])
                        if plddt_per_atom is not None
                        else 0.0
                    ),
                )
            )
        return self.biotite_structure.array(atoms)

    def build_complex(self, inputs: dict[str, Any], output: dict[str, Any]):
        atom_array = self.to_atom_array(
            coords=output["sample_atom_coords"][0].cpu().numpy(),
            atom_to_token=inputs["atom_to_token"][0].cpu().numpy(),
            res_type=inputs["res_type"][0].cpu().numpy(),
            residue_index=inputs["token_index"][0].cpu().numpy(),
            asym_id=inputs["asym_id"][0].cpu().numpy(),
            mol_type=inputs["mol_type"][0].cpu().numpy(),
            ref_atom_name_chars=inputs["ref_atom_name_chars"][0].cpu().numpy(),
            ref_element=inputs["ref_element"][0].cpu().numpy(),
            atom_attention_mask=inputs["atom_attention_mask"][0].cpu().numpy(),
        )
        return self.ProteinComplex.from_chains(
            [
                self.ProteinChain.from_atomarray(chain)
                for chain in self.biotite_structure.chain_iter(atom_array)
            ]
        )


def load_esm_folding_runtime(esm_repo: str | Path | None = None) -> ESMFoldingRuntime:
    if esm_repo is not None:
        _prepend_sys_path(Path(esm_repo).expanduser().resolve())

    import biotite.structure  # type: ignore
    import torch  # type: ignore
    import torch.nn.functional as functional  # type: ignore
    import torch.optim as optim  # type: ignore
    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM  # type: ignore
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer  # type: ignore
    from transformers.models.esmfold2.modeling_esmfold2_common import (  # type: ignore
        CUE_AVAILABLE,
    )
    from transformers.models.esmfold2.modeling_esmfold2_common import (  # type: ignore
        _seed_context as seed_context,
    )
    from transformers.models.esmfold2.modeling_esmfold2_experimental import (  # type: ignore
        ESMFold2ExperimentalModel,
    )

    from esm.models.esmfold2 import (  # type: ignore
        ELEMENT_NUMBER_TO_SYMBOL,
        ProteinInput,
        StructurePredictionInput,
        load_ccd,
        prepare_esmfold2_input,
    )
    from esm.models.esmfold2.constants import (  # type: ignore
        MOL_TYPE_NONPOLYMER,
        PROTEIN_3TO1,
        RES_TYPE_TO_CCD,
    )
    from esm.utils.structure.protein_chain import ProteinChain  # type: ignore
    from esm.utils.structure.protein_complex import ProteinComplex  # type: ignore

    tokens = ["<pad>", "-"] + [RES_TYPE_TO_CCD[index] for index in range(2, 33)]
    elements = ["X"] * (max(ELEMENT_NUMBER_TO_SYMBOL) + 1)
    elements[0] = "<pad>"
    for atomic_number, symbol in ELEMENT_NUMBER_TO_SYMBOL.items():
        elements[atomic_number] = symbol[:1] + symbol[1:].lower()

    return ESMFoldingRuntime(
        torch=torch,
        F=functional,
        optim=optim,
        seed_context=seed_context,
        ESMCTokenizer=ESMCTokenizer,
        ESMCForMaskedLM=ESMCForMaskedLM,
        ESMFold2ExperimentalModel=ESMFold2ExperimentalModel,
        CUE_AVAILABLE=bool(CUE_AVAILABLE),
        TOKENS=tokens,
        ELEMENTS=elements,
        PROTEIN_3TO1=PROTEIN_3TO1,
        ProteinInput=ProteinInput,
        StructurePredictionInput=StructurePredictionInput,
        load_ccd=load_ccd,
        prepare_esmfold2_input=prepare_esmfold2_input,
        ProteinChain=ProteinChain,
        ProteinComplex=ProteinComplex,
        biotite_structure=biotite.structure,
        MOL_TYPE_NONPOLYMER=MOL_TYPE_NONPOLYMER,
        TARGET_SEQUENCES=dict(TARGET_SEQUENCES),
    )


def _resize_tensor(torch_module, tensor, *, dim: int, size: int):
    current = tensor.shape[dim]
    if current >= size:
        return tensor.narrow(dim, 0, size)

    pad_shape = list(tensor.shape)
    pad_shape[dim] = size - current
    pad = torch_module.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch_module.cat((tensor, pad), dim=dim)


def _ensure_ccd_loaded(load_ccd) -> None:
    global _CCD_LOADED
    if _CCD_LOADED:
        return
    load_ccd()
    _CCD_LOADED = True


def _asym_id_to_chain_label(asym_id: int) -> str:
    if asym_id < 0:
        raise ValueError(f"asym_id must be >= 0, got {asym_id}")
    label = ""
    alphabet_length = len(_CHAIN_ID_ALPHABET)
    while True:
        label = _CHAIN_ID_ALPHABET[asym_id % alphabet_length] + label
        asym_id = asym_id // alphabet_length - 1
        if asym_id < 0:
            return label
