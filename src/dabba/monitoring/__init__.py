"""ML monitoring: drift detection, performance tracking.

Drift detection uses Kolmogorov-Smirnov tests (scipy.stats.ks_2samp)
comparing live inference batches against the training distribution.
No external monitoring dependency required.
"""
