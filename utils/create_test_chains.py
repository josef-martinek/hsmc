import numpy as np

def powers_of_two(upto):
    # Compute the largest exponent such that 2**k < upto
    max_exp = int(np.floor(np.log2(upto-1)))
    return 2 ** np.arange(1, max_exp + 1)  # start from 2^1 = 2

chain_len_pm = 32
fine_over_coarse_nsamp = 20
max_fine_chain_lengths = powers_of_two(upto=chain_len_pm)
print(max_fine_chain_lengths)
max_coarse_chain_lengths = np.floor(0.5*fine_over_coarse_nsamp*(chain_len_pm-max_fine_chain_lengths)+1).astype(int)
print(max_coarse_chain_lengths)
coarse_chain_lengths = powers_of_two(upto=max_coarse_chain_lengths[3])
print(coarse_chain_lengths)

def get_chain_seq(prior, sc, observed_data, chain_len_pm):
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
    fine_over_coarse_nsamp = 20
    chain_seq = []
    chain_setup = []

    max_fine_chain_lengths = powers_of_two(upto=chain_len_pm)

    for max_fine_chain_len in max_fine_chain_lengths:
        # Compute the maximum coarse chain length corresponding to this fine length
        max_coarse_chain_len = np.floor(
            0.5 * fine_over_coarse_nsamp * (chain_len_pm - max_fine_chain_len) + 1
        ).astype(int)

        coarse_chain_lengths = powers_of_two(upto=max_coarse_chain_len)

        for coarse_chain_len in coarse_chain_lengths:
            # --- Create coarse and fine models ---
            model_fine = IPModelInexactAdaptHD(
                prior=prior,
                data_noise_cov=sc.data_noise_var,
                data=observed_data
            )
            model_coarse = IPModelInexactAdaptHD(
                prior=prior,
                data_noise_cov=sc.data_noise_var,
                data=observed_data
            )

            # --- Construct coarse MCMC chain ---
            mcmc_coarse = NoisyMetropolisCal()
            chain_coarse = AdaptiveMCMCSequence(
                mcmc=mcmc_coarse, len_chain=coarse_chain_len
            )

            # --- Construct delayed-acceptance fine chain ---
            da_step = DA(
                coarse_chain=chain_coarse,
                coarse_model=model_coarse,
                fine_over_coarse=fine_over_coarse_nsamp
            )
            chain_da = AdaptiveMCMCSequence(
                mcmc=da_step, len_chain=max_fine_chain_len
            )

            # --- Store results ---
            chain_seq.append(chain_da)
            chain_setup.append({
                "fine_chain_len": int(max_fine_chain_len),
                "coarse_chain_len": int(coarse_chain_len),
                "fine_over_coarse_nsamp": int(fine_over_coarse_nsamp),
                "max_fine_chain_len": int(max_fine_chain_len),
                "max_coarse_chain_len": int(max_coarse_chain_len)
            })

    return chain_setup, chain_seq