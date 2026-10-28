"""TeNPy checkpointed environments: exactness test and memory/time benchmark (envck tree).

  python tenpy_test_and_bench.py exact            # small J1-J2 cylinder, U(1): standard vs checkpointed energies
  python tenpy_test_and_bench.py bench Lx Ly chi K   # one run (K=0 standard), prints peak RSS and time per sweep
"""
import sys, os, time, resource, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python', 'tenpy'))
import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import checkpointed_env as ce


class J1J2Square(CouplingMPOModel):
    """Spin-1/2 J1-J2 Heisenberg model on a square cylinder (periodic in y), conserving Sz."""
    def init_sites(self, model_params):
        from tenpy.networks.site import SpinHalfSite
        return SpinHalfSite(conserve=model_params.get('conserve', 'Sz'))

    def init_terms(self, model_params):
        J1, J2 = model_params.get('J1', 1.), model_params.get('J2', 0.5)
        for pairs, J in (('nearest_neighbors', J1), ('next_nearest_neighbors', J2)):
            for u1, u2, dx in self.lat.pairs[pairs]:
                self.add_coupling(J, u1, 'Sz', u2, 'Sz', dx)
                self.add_coupling(0.5 * J, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)


def model(Lx, Ly, conserve='Sz'):
    return J1J2Square(dict(lattice='Square', Lx=Lx, Ly=Ly, bc_MPS='finite', bc_y='cylinder', bc_x='open',
                           J1=1., J2=0.5, conserve=conserve))


def run(M, chi_list, k, max_sweeps):
    (ce.use_checkpointed_env(k) if k else ce.use_standard_env())
    N = M.lat.N_sites
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * (N // 2), bc='finite')
    params = dict(trunc_params=dict(chi_max=max(chi_list.values()), svd_min=1e-14), chi_list=chi_list, mixer=False,
                  max_sweeps=max_sweeps, min_sweeps=max_sweeps, max_E_err=0., lanczos_params=dict(N_max=4, N_min=2))
    eng = dmrg.TwoSiteDMRGEngine(psi, M, params)
    E, psi = eng.run()
    return E, eng


if sys.argv[1] == 'exact':
    for conserve in ('Sz', None):
        M = model(6, 4, conserve)
        res = {}
        for k in (0, 3, 5):
            E, eng = run(M, {0: 16, 2: 32, 4: 64}, k, 8)
            sweeps = [float(e) for e in eng.sweep_stats['E']]
            res[k] = sweeps
            extra = f" dropped={eng.env.n_dropped}" if k else ""
            print(f"conserve={conserve} k={k}: E={E:.14f} sweeps={len(sweeps)}{extra}")
        for k in (3, 5):
            d = max(abs(a - b) for a, b in zip(res[k], res[0]))
            print(f"conserve={conserve} k={k}: max |E_sweep - E_standard| = {d:.3e}")
else:
    Lx, Ly, chi, k = map(int, sys.argv[2:6])
    M = model(Lx, Ly)
    chi_list = {0: min(64, chi), 2: min(256, chi), 4: chi}
    t0 = time.time()
    E, eng = run(M, chi_list, k, 6)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # Linux: KiB -> GB
    times = [float(t) for t in eng.sweep_stats['time']]
    print(json.dumps(dict(Lx=Lx, Ly=Ly, chi=chi, k=k, E=float(E), peak_rss_gb=rss_gb, total_s=time.time() - t0,
                          sweep_time_cum=times)))
