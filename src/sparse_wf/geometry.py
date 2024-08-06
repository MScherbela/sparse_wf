# %%
import numpy as np
import json
import hashlib
import ase
import ase.io
import pathlib

BOHR_IN_ANGSTROM = 0.529177249
ROUND_R_DECIMALS = 5
PERIODIC_TABLE = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()
)


class NoIndentList:
    def __init__(self, values):
        self.values = values

    def to_json(self):
        return "[@@" + ", ".join([str(v) for v in self.values]) + "@@]"


def dump_to_json(data, fname):
    def default_encoder(o):
        if hasattr(o, "to_json"):
            return o.to_json()
        return o.__dict__

    class CustomEncoder(json.JSONEncoder):
        def encode(self, obj):
            s = json.JSONEncoder(default=default_encoder, indent=2).encode(obj)
            s = s.replace('"[@@', "[").replace('@@]"', "]")
            return s

        def iterencode(self, o):
            yield self.encode(o)

    with open(fname, "w") as f:
        json.dump(data, f, cls=CustomEncoder)


class Geometry:
    def __init__(self, R, Z, charge=0, spin=None, comment="", name=""):
        self.R = np.array(R)
        self.Z = np.array(Z, int)
        assert self.R.shape[0] == len(self.Z)
        assert self.R.shape[1] == 3

        self.charge = int(charge)
        if spin is None:
            spin = (sum(Z) - charge) % 2
        self.spin = int(spin)
        self.comment = comment
        self.name = name

    @classmethod
    def from_xyz(cls, fname, comment="", name=""):
        with open(fname) as f:
            n_atoms = f.readline().strip()
            try:
                n_atoms = int(n_atoms)
                comment_from_file = f.readline().strip()
            except ValueError:
                # Non-standard XYZ file, which does not contain nr of atoms in the first line
                comment_from_file = n_atoms

            R, Z = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    break
                tokens = line.split()
                assert len(tokens) == 4
                R.append([float(x) / BOHR_IN_ANGSTROM for x in tokens[1:]])
                Z.append(int(tokens[0]) if tokens[0].isdigit() else PERIODIC_TABLE.index(tokens[0].capitalize()) + 1)
        return cls(R, Z, 0, None, comment or comment_from_file, name or comment_from_file)

    @property
    def n_el(self):
        return int(np.sum(self.Z)) - self.charge

    @property
    def n_atoms(self):
        return len(self.Z)

    @property
    def n_heavy_atoms(self):
        return len(self.Z[self.Z > 1])

    def n_electrons_in_cutoff(self, cutoff: float, padding: float = 1.0):
        dist = np.linalg.norm(self.R[:, None, :] - self.R[None, :, :], axis=-1)
        in_cutoff = (dist < (cutoff + 2 * padding)).astype(int)
        n_el_in_cutoff = np.sum(in_cutoff * self.Z, axis=-1)
        return np.max(n_el_in_cutoff)

    @property
    def hash(self):
        R = np.round(self.R, decimals=ROUND_R_DECIMALS).astype(float).data.tobytes()
        Z = np.array(self.Z, int).data.tobytes()
        charge = np.array(self.charge, int).data.tobytes()
        spin = np.array(self.spin, int).data.tobytes()
        byte_string = R + Z + charge + spin
        return hashlib.md5(byte_string).hexdigest()

    @property
    def datset_entry(self):
        return self.hash + "__" + self.comment

    def __len__(self):
        return len(self.R)

    def to_json(self):
        data_dict = dict(
            name=self.name,
            comment=self.comment,
            R=NoIndentList(np.array(self.R, float).round(ROUND_R_DECIMALS).tolist()),
            Z=NoIndentList(np.array(self.Z, int).round(ROUND_R_DECIMALS).tolist()),
        )
        if self.charge != 0:
            data_dict["charge"] = self.charge
        if self.spin != 0:
            data_dict["spin"] = self.spin
        return data_dict

    def as_ase(self):
        return ase.Atoms(self.Z, self.R * BOHR_IN_ANGSTROM)

    def as_pyscf_molecule(self, basis_set):
        import pyscf.gto

        molecule = pyscf.gto.Mole()
        molecule.atom = [[Z_, tuple(R_)] for R_, Z_ in zip(R, Z)]
        molecule.unit = "bohr"
        molecule.basis = basis_set
        molecule.cart = True
        molecule.spin = self.spin  # 2*n_up - n_down
        molecule.charge = self.charge
        # maximum memory in megabytes (i.e. 10e3 = 10GB)
        molecule.max_memory = 10e3
        molecule.build()
        return molecule

    def __repr__(self):
        return f"<Geometry {self.name}, {self.datset_entry}, {self.n_el} el>"


def _get_default_geom_fname():
    return pathlib.Path(__file__).parent.joinpath("../../data/geometries.json").resolve()


def load_geometries(geom_db_fname=None) -> dict[str, Geometry]:
    geom_db_fname = geom_db_fname or _get_default_geom_fname()
    with open(geom_db_fname, "r") as f:
        geometries = json.load(f)
    if geometries is None:
        geometries = dict()
    geometries = {h: Geometry(**g) for h, g in geometries.items()}
    return geometries


def save_geometries(geometries, geom_db_fname=None):
    geom_db_fname = geom_db_fname or _get_default_geom_fname()
    geoms = load_geometries(geom_db_fname) | geometries
    dump_to_json(geoms, geom_db_fname)


if __name__ == "__main__":
    import ase.visualize
    import glob

    geoms = load_geometries()
    for h, g in geoms.items():
        if "cumulene" in g.comment:
            print(g.comment, g.n_electrons_in_cutoff(9.0))

    # fnames = glob.glob("/home/mscherbela/tmp/mpconf196/geometries/FGG*.xyz")
    # fnames = sorted(fnames)
    # geoms = [Geometry.from_xyz(fname) for fname in fnames]
    # n_el_in_cutoff = [g.n_electrons_in_cutoff(3.0) for g in geoms]
    # ind_sorted = np.argsort(n_el_in_cutoff)
    # n_el_in_cutoff = [n_el_in_cutoff[i] for i in ind_sorted]
    # geoms = [geoms[i] for i in ind_sorted]
    # fnames = [fnames[i] for i in ind_sorted]
    # ase_geoms = [g.as_ase() for g in geoms]
    # for n, f, g in zip(n_el_in_cutoff, fnames, geoms):
    #     print(f"{f.split("/")[-1]:<10}: {n:4d} / {g.n_el:4d}")

    # ase.visualize.view(ase_geoms)
