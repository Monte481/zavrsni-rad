"""MAD-based outlier detection on per-class mask L1 norms.

A trojaned model has one class with an anomalously SMALL reverse-engineered
trigger. We use the median absolute deviation (MAD) test from the Neural
Cleanse paper. Only values below the median count — large triggers are not
suspicious.

anomaly_index = |x - median| / (1.4826 * MAD)
"""

import numpy as np

import config


def mad_anomaly_index(values, threshold=config.MAD_THRESHOLD,
                      consistency=config.MAD_CONSISTENCY):
    """Return per-class anomaly indices and a boolean outlier mask.

    Args:
        values: iterable of mask L1 norms, one per class (length NUM_CLASSES).

    Returns:
        indices: np.ndarray of anomaly indices (same length as `values`).
        flagged: boolean np.ndarray; True for classes considered outliers
                 *on the small side* (potential backdoor targets).
    """
    v = np.asarray(values, dtype=np.float64)
    median = np.median(v)
    mad = np.median(np.abs(v - median))

    # Avoid division by zero (all values identical -> no anomaly).
    if mad < 1e-12:
        indices = np.zeros_like(v)
    else:
        indices = np.abs(v - median) / (consistency * mad)

    # Only "smaller than median" outliers indicate a backdoor.
    flagged = (indices > threshold) & (v < median)
    return indices, flagged
