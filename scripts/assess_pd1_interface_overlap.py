#!/usr/bin/env python3
"""Measure how a predicted PD-L1 VHH pose overlaps the PD-1 pose in 4ZQK.

This geometric screen is evidence of interface placement, not binding affinity
or experimental PD-1 blockade.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, Superimposer
import numpy as np


def _heavy_atoms(chain):
    return [
        atom for atom in chain.get_atoms()
        if atom.get_parent().id[0] == " " and atom.element not in {"H", "D"}
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicted_pdb", type=Path)
    parser.add_argument("reference_4zqk", type=Path)
    args = parser.parse_args()
    predicted = PDBParser(QUIET=True).get_structure("predicted", args.predicted_pdb)[0]
    reference = MMCIFParser(QUIET=True).get_structure("4zqk", args.reference_4zqk)[0]
    target = predicted["A"]
    binder = predicted["B"]
    reference_target = reference["A"]
    reference_pd1 = reference["B"]
    fixed, moving = [], []
    for residue in target:
        if residue.id[0] != " " or "CA" not in residue:
            continue
        auth_id = residue.id[1] + 17  # 4ZQK PD-L1 chain A covers auth 18–132.
        if (" ", auth_id, " ") in reference_target and "CA" in reference_target[auth_id]:
            fixed.append(reference_target[auth_id]["CA"])
            moving.append(residue["CA"])
    if len(fixed) < 100:
        raise ValueError(f"expected at least 100 aligned PD-L1 residues, got {len(fixed)}")
    aligner = Superimposer()
    aligner.set_atoms(fixed, moving)
    rotation, shift = aligner.rotran

    binder_atoms = _heavy_atoms(binder)
    pd1_atoms = _heavy_atoms(reference_pd1)
    binder_xyz = np.asarray([atom.coord for atom in binder_atoms]) @ rotation + shift
    pd1_xyz = np.asarray([atom.coord for atom in pd1_atoms])
    distances = np.linalg.norm(binder_xyz[:, None] - pd1_xyz[None, :], axis=-1)
    pd1_near = {
        pd1_atoms[index].get_parent().id[1]
        for index in np.where((distances < 5.0).any(axis=0))[0]
    }
    target_hotspots = (56, 66, 113, 115, 121, 123, 124, 125)
    hotspot_contact = []
    for auth_id in target_hotspots:
        atoms = _heavy_atoms(reference_target[auth_id])
        xyz = np.asarray([atom.coord for atom in atoms])
        distance = np.linalg.norm(binder_xyz[:, None] - xyz[None, :], axis=-1).min()
        if distance < 5.0:
            hotspot_contact.append(auth_id)

    print(json.dumps({
        "reference": "4ZQK human PD-1/PD-L1 complex",
        "pd_l1_target_ca_rmsd_angstrom": round(float(aligner.rms), 3),
        "pd1_vhh_min_heavy_atom_distance_angstrom": round(float(distances.min()), 3),
        "pd1_vhh_heavy_atom_pairs_under_2_angstrom": int((distances < 2).sum()),
        "pd1_residues_with_vhh_atom_within_5_angstrom": sorted(pd1_near),
        "pd_l1_interface_hotspots_contacted_by_vhh": hotspot_contact,
        "pd_l1_interface_hotspot_count": len(target_hotspots),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
