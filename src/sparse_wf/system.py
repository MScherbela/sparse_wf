import json
import numpy as np
import pyscf
from sparse_wf.api import MoleculeArgs


def chain(element: str, distance: float, n: int, **_):
    atom_strings = []
    for i in range(n):
        atom_strings.append(f"{element} {i * distance} 0 0")
    return pyscf.gto.M(atom="; ".join(atom_strings), unit="bohr")


def from_str(atom: str, spin: int = 0, **_):
    return pyscf.gto.M(atom=atom, spin=spin, unit="bohr")


def database(hash: str | None = None, name: str | None = None, comment: str | None = None, **_):
    assert hash or name or comment
    from os import path

    try:
        with open("data/geometries.json") as inp:
            geometries_by_hash = json.load(inp)
    except FileNotFoundError:
        with open(path.dirname(path.realpath(__file__)) + "/../../data/geometries.json") as inp:
            geometries_by_hash = json.load(inp)
    if hash:
        geom = geometries_by_hash[hash]
    elif name:
        geometries_with_name = [g for g in geometries_by_hash.values() if g["name"] == name]
        if len(geometries_with_name) != 1:
            raise ValueError(
                f"Expected exactly one geometry with name {name}, found {len(geometries_with_name)} in database"
            )
        geom = geometries_with_name[0]
    elif comment:
        geometries_with_comment = [g for g in geometries_by_hash.values() if g["comment"] == comment]
        if len(geometries_with_comment) != 1:
            raise ValueError(
                f"Expected exactly one geometry with comment {comment}, found {len(geometries_with_comment)} in database"
            )
        geom = geometries_with_comment[0]
    else:
        raise ValueError("No hash, name, or comment provided")

    atom = "; ".join([f"{charge} {x} {y} {z}" for charge, (x, y, z) in zip(geom["Z"], geom["R"])])
    return pyscf.gto.M(atom=atom, spin=geom.get("spin", 0), charge=geom.get("charge", 0), unit="bohr")


def cumulene(n_carbon: int, angle: float = 0.0):
    ANGSTROM_IN_BOHR = 1.8897161646320724
    CC_bond = 1.34 * ANGSTROM_IN_BOHR
    CH_bond = 1.086 * ANGSTROM_IN_BOHR
    theta = np.deg2rad(120)
    phi = np.deg2rad(angle)

    R_carbon = np.array([CC_bond, 0, 0]) * np.arange(n_carbon)[:, None]

    R_left = R_carbon[0]
    R_right = R_carbon[-1]
    R_hydrogen = np.ones([4, 3]) * CH_bond

    R_hydrogen[0] = R_left + CH_bond * np.array([np.cos(theta), np.sin(theta) * np.cos(0), np.sin(theta) * np.sin(0)])
    R_hydrogen[1] = R_left + CH_bond * np.array(
        [np.cos(theta), np.sin(theta) * np.cos(0 + np.pi), np.sin(theta) * np.sin(0 + np.pi)]
    )
    R_hydrogen[2] = R_right - CH_bond * np.array(
        [np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)]
    )
    R_hydrogen[3] = R_right - CH_bond * np.array(
        [np.cos(theta), np.sin(theta) * np.cos(phi + np.pi), np.sin(theta) * np.sin(phi + np.pi)]
    )
    R = np.vstack([R_carbon, R_hydrogen])
    Z = ["C"] * n_carbon + ["H"] * 4
    n_atoms = len(Z)
    return pyscf.gto.M(atom=[(Z[i], R[i]) for i in range(n_atoms)], unit="bohr")


def get_molecule(molecule_args: MoleculeArgs) -> pyscf.gto.Mole:
    match molecule_args["method"]:
        case "chain":
            molecule = chain(**molecule_args["chain_args"])
        case "from_str":
            molecule = from_str(**molecule_args["from_str_args"])
        case "database":
            molecule = database(**molecule_args["database_args"])
        case "cumulene":
            molecule = cumulene(**molecule_args["cumulene_args"])
    molecule.basis = molecule_args["basis"]
    if molecule_args["pseudopotentials"]:
        molecule.ecp = {atom: "ccecp" for atom in molecule_args["pseudopotentials"]}
    molecule.build()
    return molecule


def get_atomic_numbers(mol: pyscf.gto.Mole):
    # mol.atom_charges() will have the core electrons subtracted, if we want the actual
    # element we need to convert the symbols
    return np.array([pyscf.lib.parameters.ELEMENTS_PROTON[mol.atom_symbol(i)] for i in range(len(mol.atom_charges()))])
