## PEC (Potential energy curve)
This directory is for plotting the potential energy curves for H2 and LiH in three approximations:
- singular value decomposition (SVD)
- optimization of J
- optimization of J and U.

pec_JF.py : SVD and optimization of J

pec_JFU.py : SVD and optimization of J+U

plot_diff.py : plots the differences of SVD/J/J+U to FCI (Need to run both pec_JF.py and pec_JFU first)

data : results obtained (in slightly different format)
