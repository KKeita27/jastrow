#!/usr/bin/env python
# -*- coding: utf-8 -*-

from openfermion.hamiltonians import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.transforms import get_sparse_operator
from openfermion.utils import eigenspectrum, get_chemist_two_body_coefficients
from openfermion.ops import SymbolicOperator
from openfermion.ops import FermionOperator
from pyscf import fci
from scipy.optimize import minimize

import numpy as np
import csv

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time
start = time.time()
import itertools

mole_name = "LiH"
k = 2
with open('potential_JF_{}_k={}.csv'.format(mole_name, k)) as f:
    reader = csv.reader(f)
    dataJF = [row for row in reader]

dist_graph = [float(x) for x in dataJF[0]]
# print(dist)
ground_fci = [float(x) for x in dataJF[2]]
ground_svd = [float(x) for x in dataJF[7]]
ground_jf = [float(x) for x in dataJF[12]]
ex1_fci = [float(x) for x in dataJF[3]]
ex1_svd = [float(x) for x in dataJF[8]]
ex1_jf = [float(x) for x in dataJF[13]]

with open('potential_JFU_{}_k={}.csv'.format(mole_name, k)) as f:
    reader = csv.reader(f)
    dataJFU = [row for row in reader]

ground_jfu = [float(x) for x in dataJFU[12]]
ex1_jfu = [float(x) for x in dataJFU[13]]

hartree = 627.5

cmap = plt.get_cmap("tab10")
color_ground = cmap(0)
color_1st = cmap(3)

linestyle_fci = "solid"
linestyle_svd = "dotted"
linestyle_jf = "dotted"
linestyle_jfu = "dotted"

markerfacecolor_svd = "white"

marker_fci = "|"
marker_svd = "o"
marker_jf = "o"
marker_jfu = "x"
plt.figure()
plt.rcParams["font.size"] = 18
# plt.plot(dist_graph,
#          ground_fci,
#          label="FCI ground",
#          color=color_ground,
#          linestyle=linestyle_fci,
#          marker=marker_fci,
#          linewidth=2.0)
plt.plot(dist_graph,
         [hartree * abs(x - y) for x, y in zip(ground_svd, ground_fci)],
         label="SVD ground to FCI",
         color=color_ground,
         linestyle=linestyle_svd,
         marker=marker_svd,
         markerfacecolor=markerfacecolor_svd,
         fillstyle="none",
         linewidth=2.0)
plt.plot(dist_graph,
         [hartree * abs(x - y) for x, y in zip(ground_jf, ground_fci)],
         label="JF ground to FCI",
         color=color_ground,
         linestyle=linestyle_jf,
         marker=marker_jf,
         fillstyle="full",
         linewidth=2.0)
plt.plot(dist_graph,
         [hartree * abs(x - y) for x, y in zip(ground_jfu, ground_fci)],
         label="JF+U ground to FCI",
         color=color_ground,
         linestyle=linestyle_jf,
         marker=marker_jfu,
         fillstyle="full",
         linewidth=2.0)
# plt.plot(dist_graph,
#          ex1_fci,
#          label="FCI 1st",
#          color=color_1st,
#          linestyle=linestyle_fci,
#          marker=marker_fci,
#          linewidth=2.0)
plt.plot(dist_graph, [hartree * abs(x - y) for x, y in zip(ex1_svd, ex1_fci)],
         label="SVD 1st to FCI",
         color=color_1st,
         linestyle=linestyle_svd,
         markerfacecolor=markerfacecolor_svd,
         marker=marker_svd,
         fillstyle="none",
         linewidth=2.0)
plt.plot(dist_graph, [hartree * abs(x - y) for x, y in zip(ex1_jf, ex1_fci)],
         label="JF 1st to FCI",
         color=color_1st,
         linestyle=linestyle_jf,
         marker=marker_jf,
         fillstyle="full",
         linewidth=2.0)
plt.plot(dist_graph, [hartree * abs(x - y) for x, y in zip(ex1_jfu, ex1_fci)],
         label="JF+U 1st to FCI",
         color=color_1st,
         linestyle=linestyle_jf,
         marker=marker_jfu,
         fillstyle="full",
         linewidth=2.0)
plt.xlabel("Distance [au]")
plt.ylabel("Difference to FCI [kcal/mol]")
# plt.ylim([1e-4, 1e4])
plt.legend(bbox_to_anchor=(1.05, 1),
           loc='upper left',
           borderaxespad=0,
           fontsize=18)
plt.yscale('log')
plt.title("{} k={}".format(mole_name, k))
plt.savefig("diff_JF_JFU_{}_k={}.png".format(mole_name, k),
            bbox_inches='tight')
