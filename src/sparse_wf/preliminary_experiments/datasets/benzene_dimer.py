#%%
from sparse_wf.geometry import Geometry, save_geometries
import numpy as np

class Atom:
    def __init__(self, symbol, coords, units):
        self.symbol = symbol
        self.coords = coords
        self.units = units

    @property
    def Z(self):
        return {'H': 1, 'C': 6}[self.symbol]

    @property
    def R(self):
        return np.array(self.coords) * 1.8897259886


def get_benzene_dimer(dist_in_angstrom):
    mol = [
            Atom(symbol='C', coords=(0.00000, 1.396792, 0.00000), units='angstrom'),
            Atom(symbol='C', coords=(0.00000, -1.396792, 0.00000), units='angstrom'),
            Atom(symbol='C', coords=(1.209657, 0.698396, 0.00000), units='angstrom'),
            Atom(symbol='C', coords=(-1.209657, -0.698396, 0.00000), units='angstrom'),
            Atom(symbol='C', coords=(-1.209657, 0.698396, 0.00000), units='angstrom'),
            Atom(symbol='C', coords=(1.209657, -0.698396, 0.00000), units='angstrom'),
            Atom(symbol='H', coords=(0.00000, 2.484212, 0.00000), units='angstrom'),
            Atom(symbol='H', coords=(2.151390, 1.242106, 0.00000), units='angstrom'),
            Atom(symbol='H', coords=(-2.151390, -1.242106, 0.00000), units='angstrom'),
            Atom(symbol='H', coords=(-2.151390, 1.242106, 0.00000), units='angstrom'),
            Atom(symbol='H', coords=(2.151390, -1.242106, 0.00000), units='angstrom'),
            Atom(symbol='H', coords=(0.00000, -2.484212, 0.00000), units='angstrom'),

            Atom(symbol='C', coords=(0.00000, 0.00000, 1.396792 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='C', coords=(0.00000, 0.00000, -1.396792 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='C', coords=(1.209657, 0.00000, 0.698396 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='C', coords=(-1.209657, 0.00000, -0.698396 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='C', coords=(-1.209657, 0.00000, 0.698396 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='C', coords=(1.209657, 0.00000, -0.698396 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='H', coords=(0.00000, 0.00000, 2.484212 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='H', coords=(2.151390, 0.00000, 1.242106 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='H', coords=(-2.151390, 0.00000, -1.242106 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='H', coords=(-2.151390, 0.00000, 1.242106 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='H', coords=(2.151390, 0.00000, -1.242106 + dist_in_angstrom), units='angstrom'),
            Atom(symbol='H', coords=(0.00000, 0.00000, -2.484212 + dist_in_angstrom), units='angstrom')]

    R = np.array([atom.R for atom in mol])
    Z = np.array([atom.Z for atom in mol])
    return Geometry(R, Z, name="benzene_dimer_T", comment=f"benzene_dimer_T_{dist_in_angstrom:.2f}A")

geoms = [get_benzene_dimer(d) for d in [5.5]]
geom_benzene_single = Geometry(geoms[0].R[:12], geoms[0].Z[:12], name="C6H6_benzene", comment="C6H6_benzene")
save_geometries(geoms + [geom_benzene_single])





