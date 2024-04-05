"""
This file contains two classes: Atom and Molecule.
An atom consists of an element and coordinates while a molecule
is composed by a set of atoms.

The classes contain simple logic functions to obtain spins, charges
and coordinates for molecules.
"""

import numbers
import re
from collections import Counter
from functools import cached_property, total_ordering
from typing import Sequence, cast

import jax.numpy as jnp
import numpy as np
import pyscf

from .constants import ANGSTROM_TO_BOHR
from .element import (
    ELEMENT_BY_ATOMIC_NUM,
    ELEMENT_BY_LOWER_SYMBOL,
    Element,
)
from ..api import Position

Symbol = str | int


class Atom:
    element: Element
    position: np.ndarray

    def __init__(self, symbol: Symbol, position: Position = (0, 0, 0), units="bohr"):
        len(position)
        assert units in ["bohr", "angstrom"]
        if isinstance(symbol, str):
            self.element = ELEMENT_BY_LOWER_SYMBOL[symbol.lower()]
        elif isinstance(symbol, numbers.Integral):
            self.element = ELEMENT_BY_ATOMIC_NUM[symbol]
        else:
            raise ValueError()
        if position is None:
            position = (0, 0, 0)
        assert len(position) == 3
        self.position = np.array(position)
        if units == "angstrom":
            self.position *= ANGSTROM_TO_BOHR

    @property
    def atomic_number(self):
        return self.element.atomic_number

    @property
    def symbol(self):
        return self.element.symbol

    @property
    def spins(self):
        return (
            (self.atomic_number + self.element.spin) // 2,
            (self.atomic_number - self.element.spin) // 2,
        )

    def __str__(self):
        return self.element.symbol

    def __repr__(self):
        return f"{self.element.symbol} {str(self.position)}"

    @staticmethod
    def from_repr(rep):
        symbol = rep.split(" ")[0]
        position = " ".join(rep.split(" ")[1:])
        position = re.findall(r"([+-]?[0-9]+([.][0-9]*)?|[.][0-9]+)", position)
        position = [float(p[0]) for p in position]
        return Atom(symbol, position)


@total_ordering
class Molecule:
    atoms: Sequence[Atom]
    _spins: tuple[int, int] | None
    _name: str | None

    def __init__(
        self,
        atoms: Sequence[Atom],
        spins: tuple[int, int] | None = None,
        name: str | None = None,
    ) -> None:
        self.atoms = tuple(atoms)
        if spins is not None:
            self._spins = cast(tuple[int, int], tuple(spins))
        else:
            self._spins = None
        self._name = name

    @cached_property
    def charges(self):
        return tuple(a.atomic_number for a in self.atoms)

    @cached_property
    def np_positions(self):
        positions = np.array([a.position for a in self.atoms], dtype=np.float32)
        positions -= positions.mean(0, keepdims=True)
        return positions

    @cached_property
    def positions(self):
        return jnp.array(self.np_positions, dtype=jnp.float32)

    @cached_property
    def spins(self):
        if self._spins is not None:
            return self._spins
        if len(self.atoms) == 1:
            atom = self.atoms[0]
            n_ele = atom.element.atomic_number
            return (n_ele + atom.element.spin) // 2, (n_ele - atom.element.spin) // 2
        n_electrons = sum(self.charges)
        return ((n_electrons + 1) // 2, n_electrons // 2)

    @cached_property
    def atomic_spins(self):
        return tuple(a.spins for a in self.atoms)

    def to_pyscf(self, basis="STO-6G", verbose: int = 3):
        mol = pyscf.gto.Mole(
            atom=[[a.symbol, p] for a, p in zip(self.atoms, self.np_positions)],
            unit="bohr",
            basis=basis,
            verbose=verbose,
        )
        mol.spin = self.spins[0] - self.spins[1]
        mol.charge = sum(self.charges) - sum(self.spins)
        mol.build()
        return mol

    def __str__(self) -> str:
        if self._name is not None:
            return self._name
        result = ""
        if len(self.atoms) == 1:
            result = str(self.atoms[0])
        else:
            vals = dict(Counter(str(a) for a in self.atoms))
            result = "".join(str(key) + (str(val) if val > 1 else "") for key, val in vals.items())
        if sum(self.spins) != sum(self.charges):
            result += f"{self.spins[0]}_{self.spins[1]}"
        return result

    def __repr__(self) -> str:
        atoms = "\n".join(map(repr, self.atoms))
        result = f"Spins: {self.spins}\n{atoms}"
        if self._name is not None:
            result = f"{self._name}\n{result}"
        return result

    @staticmethod
    def from_repr(rep):
        return Molecule([Atom.from_repr(r) for r in rep.split("\n")[1:]])

    def __lt__(self, other):
        return (sum(self.spins), self.spins, self.charges, self._name) < (
            sum(other.spins),
            other.spins,
            other.charges,
            self._name,
        )
