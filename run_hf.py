import pyscf
from seml import Experiment
from sparse_wf.system import database


ex = Experiment()


@ex.automain
def main(
    hash: str | None = None,
    name: str | None = None,
    comment: str | None = None,
    basis: str = "sto-6g",
    x2c: bool = False,
    pp: list[str] = [],
    restricted: bool = False,
    newton: bool = False,
):
    print(locals())
    mol = database(hash=hash, name=name, comment=comment)
    mol.basis = basis
    if pp:
        mol.ecp = {atom: "ccecp" for atom in pp}
    mol.build()

    if restricted:
        mf = mol.RHF()
    else:
        mf = mol.UHF()

    if x2c:
        mf = mf.x2c()

    if newton:
        mf = mf.newton()
    else:
        # use damping for the 5 first iterations
        mf.damp = 0.5
        mf.diis_start_cycle = 5
        # use dynamic level shifts
        pyscf.scf.addons.dynamic_level_shift_(mf, factor=0.5)

    # set run params
    mf.verbose = 100
    mf.max_cycle = 200
    energy = mf.kernel()
    return dict(
        energy=energy,
        converged=mf.converged,
    )
