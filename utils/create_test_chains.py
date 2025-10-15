import numpy as np
from problem_setup.toy_problem.likelihood_model import IPModelInexactAdaptHD
from core.mcmc import NoisyMetropolisCal, TwoStageMCMC
from particles.smc_samplers import ArrayRandomWalk

def powers_of_two(upto):
    # Compute the largest exponent such that 2**k < upto
    max_exp = int(np.floor(np.log2(upto-1)))
    return 2 ** np.arange(1, max_exp + 1)  # start from 2^1 = 2

def powers_of_two_start_from_one(upto):
    # Compute the largest exponent such that 2**k < upto
    max_exp = int(np.floor(np.log2(upto-1)))
    return 2 ** np.arange(0, max_exp + 1)  # start from 2^1 = 2

def get_chain_seq(prior, data_noise_var, observed_data, chain_len_pm, fine_over_coarse_nsamp):
    """
    Generate a sequence of DA (Delayed Acceptance) chains with varying
    coarse and fine chain lengths based on powers of two.

    Returns
    -------
    chain_setup : list of dict
        Metadata describing each chain setup.
    chain_seq : list
        List of constructed AdaptiveMCMCSequence objects.
    """
    chain_seq = []
    chain_setup = []

    max_fine_chain_lengths = powers_of_two(upto=chain_len_pm)

    model_coarse = IPModelInexactAdaptHD(
        prior=prior,
        data_noise_cov=data_noise_var,
        data=observed_data
    )

    count = 0
    for max_fine_chain_len in max_fine_chain_lengths:
        # Compute the maximum coarse chain length corresponding to this fine length
        max_coarse_chain_len = np.floor(
            0.5 * fine_over_coarse_nsamp * (chain_len_pm - max_fine_chain_len) + 1
        ).astype(int)

        coarse_chain_lengths = powers_of_two_start_from_one(upto=max_coarse_chain_len)

        for coarse_chain_len in coarse_chain_lengths:
            # --- Create coarse and fine models ---
            model_coarse = IPModelInexactAdaptHD(
                prior=prior,
                data_noise_cov=data_noise_var,
                data=observed_data
            )

            # --- Construct coarse MCMC chain ---
            mcmc_coarse = NoisyMetropolisCal()
            mcmc_fine = ArrayRandomWalk()
            two_stage_chain = TwoStageMCMC(mcmc1=mcmc_coarse,
                                           mcmc2=mcmc_fine,
                                           len_chain1=coarse_chain_len,
                                           len_chain2=max_fine_chain_len,
                                           coarse_model=model_coarse,
                                           fine_over_coarse=fine_over_coarse_nsamp)

            # --- Store results ---
            chain_seq.append(two_stage_chain)
            chain_setup.append({
                "chain_id": count,
                "fine_chain_len": int(max_fine_chain_len),
                "coarse_chain_len": int(coarse_chain_len),
                "fine_over_coarse_nsamp": int(fine_over_coarse_nsamp)
            })
            count += 1

    return chain_setup, chain_seq
