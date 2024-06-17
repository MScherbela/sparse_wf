import json
from typing import Any

import pyscf


def chain(element: str, distance: float, n: int, **_):
    atom_strings = []
    for i in range(n):
        atom_strings.append(f"{element} {i * distance} 0 0")
    return pyscf.got.M(atom="; ".join(atom_strings), unit="bohr")


def from_str(atom: str, spin: int = 0, **_):
    return pyscf.gto.M(atom=atom, spin=spin, unit="bohr")


def database(hash: str | None = None, name: str | None = None, comment: str | None = None, **_):
    assert hash is not None or name is not None or comment is not None
    from os import path

    try:
        with open("data/geometries.json") as inp:
            geometries_by_hash = json.load(inp)
    except FileNotFoundError:
        with open(path.dirname(path.realpath(__file__)) + "/../../data/geometries.json") as inp:
            geometries_by_hash = json.load(inp)
    if hash:
        geom = geometries_by_hash[hash]
    if name:
        geometries_by_name = {g["name"]: g for g in geometries_by_hash.values()}
        geom = geometries_by_name[name]
    if comment:
        geometries_by_comment = {g["comment"]: g for g in geometries_by_hash.values()}
        geom = geometries_by_comment[comment]

    atom = "; ".join([f"{charge} {x} {y} {z}" for charge, (x, y, z) in zip(geom["Z"], geom["R"])])
    return pyscf.gto.M(atom=atom, spin=geom["spin"], charge=geom["charge"], unit="bohr")


def get_molecule(method: str, args: dict[str, Any], basis: str) -> pyscf.gto.Mole:
    constructor = globals().get(method)
    assert constructor is not None, f"Coult not find constructor for method {method}."
    molecule = constructor(**args)
    molecule.basis = basis
    molecule.build()
    return molecule
