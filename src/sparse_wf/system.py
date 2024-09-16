import json
import pyscf
from sparse_wf.api import MoleculeArgs


def chain(element: str, distance: float, n: int, **_):
    atom_strings = []
    for i in range(n):
        atom_strings.append(f"{element} {i * distance} 0 0")
    return pyscf.gto.M(atom="; ".join(atom_strings), unit="bohr")


def from_str(atom: str, spin: int = 0, **_):
    return pyscf.gto.M(atom=atom, spin=spin, unit="bohr")


def database(
    hash: str | None = None, name: str | None = None, comment: str | None = None, spin: int | None = None, **_
):
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
    if spin is None:
        spin = geom.get("spin", 0)
    return pyscf.gto.M(atom=atom, spin=spin, charge=geom.get("charge", 0), unit="bohr")


def get_molecule(molecule_args: MoleculeArgs) -> pyscf.gto.Mole:
    match molecule_args["method"]:
        case "chain":
            molecule = chain(**molecule_args["chain_args"])
        case "from_str":
            molecule = from_str(**molecule_args["from_str_args"])
        case "database":
            molecule = database(**molecule_args["database_args"])
    molecule.basis = molecule_args["basis"]
    molecule.build()
    return molecule
