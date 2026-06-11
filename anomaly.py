import numpy as np

import config


def mad_anomaly_index(values, threshold=config.MAD_THRESHOLD,
                      consistency=config.MAD_CONSISTENCY):
    # Trojanski model ima jednu klasu s neuobičajeno MALOM maskom. Tražimo
    # outliere ispod medijana pomoću MAD testa (indeks = |x - med| / (1.4826 * MAD)).
    v = np.asarray(values, dtype=np.float64)
    median = np.median(v)
    mad = np.median(np.abs(v - median))

    if mad < 1e-12:                 # sve vrijednosti iste -> nema anomalije
        indices = np.zeros_like(v)
    else:
        indices = np.abs(v - median) / (consistency * mad)

    # Sumnjive su samo male maske (ispod medijana).
    flagged = (indices > threshold) & (v < median)
    return indices, flagged
