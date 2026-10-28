"""quimb checkpointed environments: exactness test and memory/time benchmark (envck tree).

  python quimb_test.py exact
  python quimb_test.py bench L chi K        # one run (K=0 standard); prints peak RSS and time
"""
import sys, os, time, resource, json
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_here, '..', 'python', 'quimb'), os.path.join(_here, '..', 'python', 'tenpy'), _here]
import quimb.tensor as qtn
import checkpointed_me as cm


def j1j2_cylinder_mpo(Lx, Ly):
    """Square J1-J2 (J2=0.5) cylinder MPO built with TeNPy, converted to a dense quimb MPO."""
    from tenpy_test_and_bench import model as tenpy_model
    Hm = tenpy_model(Lx, Ly, conserve=None).H_MPO
    N = Hm.L
    arrs = []
    for i in range(N):
        W = Hm.get_W(i).to_ndarray()                     # legs wL, wR, p, p*
        if i == 0:
            W = W[Hm.IdL[0]]
        if i == N - 1:
            W = W[..., Hm.IdR[N], :, :] if i > 0 else W[Hm.IdR[N]]
        arrs.append(W)
    return qtn.MatrixProductOperator(arrs, shape='lrud' if N > 1 else 'ud')


def run(L, chis, k, nsweeps, Ly=None):
    (cm.use_checkpointed_me(k) if k else cm.use_standard_me())
    H = j1j2_cylinder_mpo(L // Ly, Ly) if Ly else qtn.MPO_ham_heis(L, j=1.0, cyclic=False)
    p0 = qtn.MPS_rand_state(L, 8, seed=1)    # same start for every run
    dmrg = qtn.DMRG2(H, bond_dims=chis, cutoffs=1e-14, p0=p0)
    dmrg.solve(tol=1e-30, max_sweeps=nsweeps, verbosity=0)
    return [float(e) for e in dmrg.energies]


if sys.argv[1] == 'exact':
    for Ly in (None, 4):
      ref = run(24, [16, 32, 64], 0, 6, Ly)
      for k in (3, 5):
        e = run(24, [16, 32, 64], k, 6, Ly)
        print(f"Ly={Ly} k={k}: max |E_sweep - E_standard| over {len(e)} sweeps = "
              f"{max(abs(a - b) for a, b in zip(e, ref)):.3e}  E={e[-1]:.14f}  rebuilt={cm.CheckpointedMovingEnvironment.n_rebuilt}")
else:
    L, chi, k = map(int, sys.argv[2:5])
    Ly = int(sys.argv[5]) if len(sys.argv) > 5 else None    # cylinder width: J1-J2 cylinder with L/Ly columns
    t = time.time()
    e = run(L, [min(64, chi), min(256, chi), chi], k, 4, Ly)
    print(json.dumps(dict(L=L, chi=chi, k=k, E=e[-1], peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                          total_s=time.time() - t)))
