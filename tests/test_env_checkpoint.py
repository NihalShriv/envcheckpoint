"""Test for the proposed TeNPy option ``env_checkpoint_every`` (pytest style, for tests/ in the PR).

Run against the patched TeNPy copies: PYTHONPATH=<dir with patched tenpy> pytest test_env_checkpoint.py
"""
import pytest
from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg


def _run(k, conserve):
    M = XXZChain(dict(L=24, Jxx=1., Jz=1., hz=0., bc_MPS='finite', conserve=conserve))
    psi = MPS.from_product_state(M.lat.mps_sites(), ['up', 'down'] * 12, bc='finite')
    opts = dict(trunc_params=dict(chi_max=48, svd_min=1e-14), mixer=False, max_sweeps=6, min_sweeps=6,
                max_E_err=0., lanczos_params=dict(N_max=4, N_min=2))
    if k:
        opts['env_checkpoint_every'] = k
    eng = dmrg.TwoSiteDMRGEngine(psi, M, opts)
    E, _ = eng.run()
    n_stored = sum(1 for key in eng.env.cache.keys() if key.startswith(('LP_', 'RP_')))
    return [float(e) for e in eng.sweep_stats['E']], E, n_stored


@pytest.mark.parametrize('conserve', ['Sz', None])
@pytest.mark.parametrize('k', [2, 3, 5])
def test_env_checkpoint_exact(k, conserve):
    ref, E_ref, n_ref = _run(None, conserve)
    res, E, n = _run(k, conserve)
    assert res == ref          # identical contractions -> identical energies
    assert E == E_ref
    assert n < n_ref           # fewer environments stored
