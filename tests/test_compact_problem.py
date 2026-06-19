import numpy as np
import pytest
from scipy.stats import chi
from problem_setup.compact_problem.likelihood_model import CompactModel

def naive_rejection_sampling(params, nsamp, R, operator, seed=42):
    """
    A baseline rejection sampler to verify our Inverse Transform Sampling trick against.
    """
    np.random.seed(seed)
    N_batch = params.shape[0]
    m = operator.shape[0]
    u_exact = params @ operator.T
    
    # Isotropic variance
    variances = CompactModel.base_variance / nsamp
    stds = np.sqrt(variances)
    
    e_out = np.zeros((N_batch, m))
    for i in range(N_batch):
        sigma = stds
        while True:
            # Draw standard normal and scale by isotropic std
            e = np.random.normal(loc=0.0, scale=sigma, size=m)
            if np.linalg.norm(e) <= R:
                e_out[i] = e
                break
    return u_exact + e_out

def test_exact_solver_shape():
    """
    Test that the exact solver returns the correct shape.
    """
    N_batch = 15
    d = CompactModel.d
    m = CompactModel.m
    
    params = np.random.randn(N_batch, d)
    u_exact = CompactModel.exact_sol(params)
    
    assert u_exact.shape == (N_batch, m)

def test_inexact_solver_truncation():
    """
    Test that all drawn noise vectors strictly satisfy the compact boundary condition ||e||_2 <= R.
    """
    np.random.seed(1234)
    N_batch = 1000
    d = CompactModel.d
    m = CompactModel.m
    R = CompactModel.truncation_radius_R
    
    params = np.random.randn(N_batch, d)
    u_exact = CompactModel.exact_sol(params)
    
    # Check for both small and large nsamp values
    for nsamp in [1, 10, 100]:
        u_inexact = CompactModel.inexact_sol(params, nsamp)
        noise = u_inexact - u_exact
        norms = np.linalg.norm(noise, axis=1)
        
        # Strictly assert R boundary
        assert np.all(norms <= R), f"Noise norm exceeded R for nsamp={nsamp}"

def test_sampling_statistical_equivalence():
    """
    Test that Inverse Transform Sampling and Rejection Sampling are statistically equivalent.
    We draw a large number of samples for a single parameter and compare their norms.
    """
    # Fix seed for reproducibility
    np.random.seed(42)
    
    N_batch = 2000
    nsamp = 5
    d = CompactModel.d
    m = CompactModel.m
    R = CompactModel.truncation_radius_R
    operator = CompactModel.exact_fw_operator
    
    # Draw many samples for a single parameter vector u
    u_param = np.array([[1.0, -0.5, 0.2, 0.8, -0.3]])
    params = np.tile(u_param, (N_batch, 1))
    
    # 1. Generate samples via our Scipy Chi Inverse Transform Sampling
    u_exact = CompactModel.exact_sol(params)
    u_its = CompactModel.inexact_sol(params, nsamp)
    noise_its = u_its - u_exact
    norms_its = np.linalg.norm(noise_its, axis=1)
    
    # 2. Generate samples via Rejection Sampling baseline
    u_rej = naive_rejection_sampling(params, nsamp, R, operator, seed=42)
    noise_rej = u_rej - u_exact
    norms_rej = np.linalg.norm(noise_rej, axis=1)
    
    # 3. Verify statistical metrics of the L2 norms
    mean_its, std_its = np.mean(norms_its), np.std(norms_its)
    mean_rej, std_rej = np.mean(norms_rej), np.std(norms_rej)
    
    print(f"\n[ITS] Mean norm: {mean_its:.5f}, Std: {std_its:.5f}")
    print(f"[REJ] Mean norm: {mean_rej:.5f}, Std: {std_rej:.5f}")
    
    # Assert means and standard deviations match closely (within 3 standard errors)
    assert np.allclose(mean_its, mean_rej, atol=1e-2), "ITS and Rejection Sampling mean norms do not match!"
    assert np.allclose(std_its, std_rej, atol=1e-2), "ITS and Rejection Sampling standard deviations do not match!"

def test_simple_truncation_200_samples():
    """
    A simple component test to explicitly verify that none of the 200 generated 
    ITS samples falls outside of the expected truncation region (||e||_2 <= R).
    """
    np.random.seed(999)
    N_batch = 200
    nsamp = 3
    d = CompactModel.d
    m = CompactModel.m
    R = CompactModel.truncation_radius_R
    
    # Generate some random input parameters
    params = np.random.randn(N_batch, d)
    u_exact = CompactModel.exact_sol(params)
    
    # Generate ITS samples
    u_its = CompactModel.inexact_sol(params, nsamp)
    noise = u_its - u_exact
    
    # Verify norms of the noise vector
    norms = np.linalg.norm(noise, axis=1)
    
    # Assert that all 200 samples are strictly within the truncation ball of radius R
    assert np.all(norms <= R), "Found a sample falling outside the expected truncation region!"
    assert len(norms) == 200

