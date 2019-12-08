#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from openfermion.hamiltonians import MolecularData
# from openfermionpyscf import run_pyscf
# from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
# from openfermion.transforms import get_sparse_operator
# from openfermion.utils import eigenspectrum, get_chemist_two_body_coefficients
# from openfermion.ops import SymbolicOperator
# from openfermion.ops import FermionOperator
# from pyscf import fci
from scipy.optimize import minimize

import numpy as np
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import time
# start = time.time()
import itertools

import math

# mole_name = "LiH"
# for k in range(1, 4):
with open('time_data.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

time_opt = [[], [], []]
time_other = [[], [], []]
k_other = [[], [], []]
k_opt = [[], [], []]
nqubit_other = [[], [], []]
nqubit_opt = [[], [], []]
mole_name = ["H2O", "H2", "LiH"]
for row in data:
    mole_id = -1
    for i in range(3):
        if row[0] == mole_name[i]:
            mole_id = i
            break
    if mole_id == -1:
        continue

    for t in row[4:]:
        if row[3] == "Other":
            try:
                time_other[mole_id].append(float(t))
                nqubit_other[mole_id].append(float(row[1]))
                k_other[mole_id].append(float(row[2]))
            except ValueError:
                pass
        else:
            try:
                time_opt[mole_id].append(float(t))
                nqubit_opt[mole_id].append(float(row[1]))
                k_opt[mole_id].append(float(row[2]))
            except ValueError:
                pass

# Start fitting

k_other_fit = sum(k_other, [])
k_opt_fit = sum(k_opt, [])
nqubit_other_fit = sum(nqubit_other, [])
nqubit_opt_fit = sum(nqubit_opt, [])
time_opt_fit = sum(time_opt, [])
time_other_fit = sum(time_other, [])

a_other, b_other = np.polyfit([n * np.log(2) for n in nqubit_other_fit],
                              [np.log(t) for t in time_other_fit], 1)
a_opt, b_opt = np.polyfit(
    [np.log(n * n * k / 4) for n, k in zip(nqubit_opt_fit, k_opt_fit)],
    [np.log(t) for t in time_opt_fit], 1)
# Start plotting

cmap = plt.get_cmap("tab10")
color_ground = cmap(0)
color_1st = cmap(3)

linestyle_fci = "solid"
linestyle_svd = "dotted"
linestyle_jf = "dotted"

markerfacecolor_svd = "white"

marker_fci = "|"
marker_svd = "o"
marker_jf = "o"

plt.figure()
plt.rcParams["font.size"] = 18
# plt.plot(dist_graph,
#          ground_fci,
#          label="FCI ground",
#          color=color_ground,
#          linestyle=linestyle_fci,
#          marker=marker_fci,
#          linewidth=2.0)
plt.plot(
    [np.power(2, n) for n in nqubit_other_fit],
    [
        pow(math.e, b_other) * np.power(2, a_other * n)
        for n in nqubit_other_fit
    ],
    label="fit {:.5g}*2^({:.5g}*n)".format(np.power(math.e, b_other), a_other),
    color="gray",
    linestyle="solid",
    # marker="o",
    linewidth=2.0)
for i in range(3):
    plt.plot([np.power(2, n) for n in nqubit_other[i]],
             time_other[i],
             label="{}".format(mole_name[i]),
             color=cmap(i),
             linestyle="none",
             marker="x",
             linewidth=2.0)
plt.xlabel("2^nqubit")
plt.ylabel("time [sec]")
plt.legend(bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=0,
           fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.savefig("time_other.png", bbox_inches='tight')

plt.figure()
plt.plot(
    [k * n * n / 4 for n, k in zip(nqubit_opt_fit, k_opt_fit)],
    [
        pow(math.e, b_opt) * np.power(n * n * k / 4, a_opt)
        for n, k in zip(nqubit_opt_fit, k_opt_fit)
    ],
    label="fit {:.5g}*(N)^{:.5g}".format(np.power(math.e, b_opt), a_opt),
    color="gray",
    linestyle="solid",
    # marker="o",
    linewidth=2.0)
for i in range(3):
    plt.plot([k * n * n / 4 for n, k in zip(nqubit_opt[i], k_opt[i])],
             time_opt[i],
             label="{}".format(mole_name[i]),
             color=cmap(i),
             linestyle="none",
             marker="x",
             linewidth=2.0)
plt.xlabel("number of optimization variables $N$")
plt.ylabel("time [sec]")
plt.legend(bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=0,
           fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.savefig("time_opt.png", bbox_inches='tight')