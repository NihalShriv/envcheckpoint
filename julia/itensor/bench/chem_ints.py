"""Write MO integrals of a linear H_n chain for bench2.jl (MODE=chem).
Usage: python chem_ints.py <natoms> <out.txt> [R_bohr=1.8] [basis=sto-6g]
Format: "n nelec", "ecore", then "h i j v" (1-based) and "g i j k l v" for (ij|kl), all nonzero entries."""
import sys
import numpy as np
from pyscf import gto, scf, ao2mo

na, out = int(sys.argv[1]), sys.argv[2]
R = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8
basis = sys.argv[4] if len(sys.argv) > 4 else "sto-6g"
mol = gto.M(atom=[("H", (0, 0, i * R)) for i in range(na)], unit="Bohr", basis=basis, verbose=0)
mf = scf.RHF(mol).run()
C = mf.mo_coeff; n = C.shape[1]
h = C.T @ mf.get_hcore() @ C
g = ao2mo.restore(1, ao2mo.kernel(mol, C), n)
with open(out, "w") as f:
    f.write(f"{n} {mol.nelectron}\n{mol.energy_nuc()}\n")
    for i in range(n):
        for j in range(n):
            if abs(h[i, j]) > 1e-12:
                f.write(f"h {i+1} {j+1} {h[i, j]:.16e}\n")
    for (i, j, k, l), v in np.ndenumerate(g):
        if abs(v) > 1e-12:
            f.write(f"g {i+1} {j+1} {k+1} {l+1} {v:.16e}\n")
print(f"n={n} nelec={mol.nelectron} E_HF={mf.e_tot:.10f}")
