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

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import time
start = time.time()
import itertools

import csv
#define constants
basis = "sto-3g"  #basis set
multiplicity = 1  #spin multiplicity
charge = 0  #total charge for the molecule

# # dist = np.arange(0.4, 4.0, .2)  # for H2
nqubit = 4
mole_name = "H2"
mole1 = "H"
mole2 = "H"
k_svd = 4  # maximum value for SVD
k_jf = 4  #maximum value for JF
d = .65

# # dist = np.arange(0.7, 4.0, .2)  # for LiH
# nqubit = 12
# mole_name = "LiH"
# mole1 = "Li"
# mole2 = "H"
# k_svd = 36  # maximum value for SVD
# k_jf = 1  #maximum value for JF
# d = 1.1

comment = "H is now Hermite"

half_nqubit = nqubit / 2

# dist = [0.65]
# Basis for half_nqubit x half_nqubit hermitian matrix
H_basis = np.zeros((half_nqubit**2, half_nqubit, half_nqubit), dtype=complex)
H_basis_count = 0
for i in range(half_nqubit):
    H_basis[H_basis_count][i][i] = 1
    H_basis_count += 1
    # print("H_basis[{}] = np.diag({})".format(i, basis))

for i in range(1, half_nqubit):
    for j in range(i):
        H_basis[H_basis_count][i][j] = 1.
        H_basis[H_basis_count][j][i] = 1.
        H_basis_count += 1
        H_basis[H_basis_count][i][j] = 1j
        H_basis[H_basis_count][j][i] = -1j
        H_basis_count += 1

# lists for results
error_svd = []
error_jf = []
error_jfu = []

# time count
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print("k={}, d={}".format(k_svd, d))

# molecule declaration
distance = d
geometry = [(mole1, (0, 0, 0)),
            (mole2, (0, 0, distance))]  #xyz coordinates for atoms
description = str(distance)  #description for the psi4 output file
molecule = MolecularData(geometry, basis, multiplicity, charge, description)
molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
fermionic_hamiltonian = get_fermion_operator(
    molecule.get_molecular_hamiltonian())

# matrix for hamiltonian
h = np.array([0, [], [[]], [[[]]], [[[[]]]]])
h[0] = fermionic_hamiltonian.terms[()]
h[2] = np.zeros((nqubit, nqubit))
h[4] = np.zeros((nqubit, nqubit, nqubit, nqubit))

# input hamiltonian into h
for i, j, k, l in itertools.product(range(nqubit), repeat=4):
    h[4][i][j][k][l] = fermionic_hamiltonian.terms.get(
        ((i, 1), (j, 1), (k, 0), (l, 0)), 0)

# get chemist-ordered matrix (h2_correction unused)
h2_correction, V = get_chemist_two_body_coefficients(h[4], spin_basis=True)

# reshape into matrix
Vmat = V.reshape(half_nqubit**2, half_nqubit**2)

# SVD decomposition
u1, lam, u2 = np.linalg.svd(Vmat)

# Record the errors of SVD expansion in each k
V_svd = np.tensordot(u1[:, 0], u2[0, :], 0) * lam[0]
error_svd.append(np.linalg.norm(Vmat - V_svd))
for i in range(1, k_svd):
    V_svd += np.tensordot(u1[:, i], u2[i, :], 0) * lam[i]
    error_svd.append(np.linalg.norm(Vmat - V_svd))

# 第一添え字をxにするため
u1 = u1.T

u1mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
u2mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)

u11mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
u12mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)

u21mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
u22mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)

w1 = np.empty((k_svd, half_nqubit), dtype=complex)
w2 = np.empty((k_svd, half_nqubit), dtype=complex)

for i in range(k_svd):
    u1mat[i] = u1[i].reshape((half_nqubit, half_nqubit))
    w1[i], u11mat[i] = np.linalg.eig(u1mat[i])
    u12mat[i] = u11mat[i].T
    # u2mat[i] = u2[i].reshape((half_nqubit, half_nqubit))
    u21mat[i] = u11mat[i]
    w2[i] = w1[i]
    u22mat[i] = u11mat[i].T
    # w2[i], u21mat[i] = np.linalg.eig(u2mat[i])
    # u22mat[i] = np.linalg.inv(u21mat[i])
    print(
        np.linalg.norm(u1mat[i] -
                       u11mat[i].dot(np.diag(w1[i]).dot(u12mat[i]))))

J = np.empty((k_svd, half_nqubit, half_nqubit), dtype=float)
for i in range(k_svd):
    J[i] = np.tensordot(w1[i], w2[i], 0)

V_jf = np.zeros((half_nqubit, half_nqubit, half_nqubit, half_nqubit),
                dtype=complex)
for x in range(k_svd):
    for p, q, r, s, i, j in itertools.product(range(half_nqubit), repeat=6):
        V_jf[p][q][r][s] += lam[x] * u11mat[x][p][i] * u12mat[x][i][q] * J[x][
            i][j] * u21mat[x][r][j] * u22mat[x][j][s]

cost_history = []


# half_nqubit^2対称行列を受け取り、その独立成分を持ったベクトルを返す。
def symmetric_matrix_to_vector(J_tmp):
    Jsymvec = np.zeros(half_nqubit * (half_nqubit + 1) / 2)
    counter = 0
    # Jsymvec = np.ndarray()
    for i in range(half_nqubit):
        for j in range(i, half_nqubit):
            Jsymvec[counter] = J_tmp[i][j]
            counter += 1
    return Jsymvec


# 上で作ったベクトルを受け取り、対称行列を返す
def vector_to_symmetric_matrix(Jsymvectmp):
    J_tmp = np.zeros((half_nqubit, half_nqubit))
    counter = 0
    for i in range(half_nqubit):
        for j in range(i, half_nqubit):
            J_tmp[i][j] = Jsymvectmp[counter]
            J_tmp[j][i] = Jsymvectmp[counter]
            counter += 1
    return J_tmp


for k in range(k_jf):
    J = np.empty((k_svd, half_nqubit, half_nqubit), dtype=float)
    for i in range(k_svd):
        J[i] = np.tensordot(w1[i], w2[i], 0)

    Jvec = np.array([symmetric_matrix_to_vector(J[i]) for i in range(k + 1)])

    def cost(Jvec_tmp):
        V_jf_tmp = np.zeros(
            (half_nqubit, half_nqubit, half_nqubit, half_nqubit), dtype=float)
        Jvec_tmp = Jvec_tmp.reshape(k + 1, half_nqubit * (half_nqubit + 1) / 2)
        J_tmp = np.array(
            [vector_to_symmetric_matrix(Jvec_tmp[i]) for i in range(k + 1)])
        for x in range(k + 1):
            for p, q, r, s, i, j in itertools.product(range(half_nqubit),
                                                      repeat=6):
                V_jf_tmp[p][q][r][s] += lam[x] * u11mat[x][p][i] * u12mat[x][
                    i][q] * J_tmp[x][i][j] * u21mat[x][r][j] * u22mat[x][j][s]
        return np.linalg.norm(V_jf_tmp - V)
        # return 0

    # print(J.shape)
    # 最適化を開始
    # cost_history.append(1.)
    # cost_history.append(cost(J[:k + 1, :, :]))
    method = "BFGS"
    options = {"disp": True, "maxiter": 50, "gtol": 1e-6}

    # minimizeに使えるようにパラメーターを1列に並べなおす
    print("Start optimization:{0}".format(time.time() - start) + "[sec]")
    # print(Jvec.size)
    opt = minimize(cost, Jvec, method=method)
    # callback=lambda y: cost_history.append(cost(y)))
    Jvec = opt.x
    Jvec = Jvec.reshape((k + 1, half_nqubit * (half_nqubit + 1) / 2))
    print("End optimization:{0}".format(time.time() - start) + "[sec]")
    J = np.array([vector_to_symmetric_matrix(Jvec[i]) for i in range(k + 1)])

    # 再度初期化
    V_jf = np.zeros((half_nqubit, half_nqubit, half_nqubit, half_nqubit),
                    dtype=complex)
    for x in range(k + 1):
        for p, q, r, s, i, j in itertools.product(range(half_nqubit),
                                                  repeat=6):
            V_jf[p][q][r][s] += lam[x] * u11mat[x][p][i] * u12mat[x][i][q] * J[
                x][i][j] * u21mat[x][r][j] * u22mat[x][j][s]
    error_jf.append(np.linalg.norm(V_jf - V))


def log_of_unitary(U):
    Lam, U1 = np.linalg.eig(U)
    logLam = [np.log(x).imag for x in Lam]
    return U1.dot(np.diag(logLam).dot(np.conjugate(U1.T)))


from scipy.linalg import expm
# print("test:{}".format(
#     np.linalg.norm(u11mat[0] - expm(1j * log_of_unitary(u11mat[0])))))

# Hermiteの基底で展開（フィット）
H = np.array([log_of_unitary(u11mat[i]) for i in range(k_jf)])

#成分を拾ってきて初期値とする
Hvec = np.zeros((k_jf, half_nqubit**2))
for k in range(k_jf):
    H_basis_count = 0
    for i in range(half_nqubit):
        Hvec[k][H_basis_count] = np.real(H[k][i][i])
        H_basis_count += 1

    for i in range(1, half_nqubit):
        for j in range(i):
            Hvec[k][H_basis_count] = np.real((H[k][i][j] + H[k][j][i]) / 2)
            H_basis_count += 1
            Hvec[k][H_basis_count] = np.imag((H[k][i][j] - H[k][j][i]) / 2)
            H_basis_count += 1

H_tmp = np.zeros((k_jf, half_nqubit, half_nqubit), dtype=complex)
Hvec_tmp = Hvec.reshape(k_jf, half_nqubit**2)
for k in range(k_jf):
    for j in range(half_nqubit**2):
        H_tmp[k] += Hvec_tmp[k][j] * H_basis[j]


def cost_H(Hvec_tmp):
    H_tmp = np.zeros((k_jf, half_nqubit, half_nqubit), dtype=complex)
    Hvec_tmp = Hvec_tmp.reshape(k_jf, half_nqubit**2)
    for i in range(k_jf):
        for j in range(half_nqubit**2):
            H_tmp[i] += Hvec_tmp[i][j] * H_basis[j]
    return np.linalg.norm(H - H_tmp)


method = "BFGS"
options = {"disp": True, "maxiter": 50, "gtol": 1e-6}

print("Start H optimization:{0}".format(time.time() - start) + "[sec]")
opt = minimize(cost_H, Hvec, method=method)
# callback=lambda y: cost_history.append(cost_H(y)))
print("End H optimization:{0}".format(time.time() - start) + "[sec]")
Hvec = opt.x
# Hのエルミート基底での展開ここまで

for k in range(k_jf):
    print("k={}".format(k))
    J = np.empty((k_svd, half_nqubit, half_nqubit), dtype=float)
    for i in range(k_svd):
        J[i] = np.tensordot(w1[i], w2[i], 0)

    # JとHを最適化するためのコスト関数
    def cost(data_tmp):
        Jvec_tmp = data_tmp[:(k + 1) * half_nqubit * (half_nqubit + 1) /
                            2].reshape(
                                (k + 1, half_nqubit * (half_nqubit + 1) / 2))
        J_tmp = np.array(
            [vector_to_symmetric_matrix(Jvec_tmp[i]) for i in range(k + 1)])
        Hvec_tmp = data_tmp[(k + 1) * half_nqubit * (half_nqubit + 1) /
                            2:].reshape(k + 1, half_nqubit**2)
        H_tmp = np.zeros((k + 1, half_nqubit, half_nqubit), dtype=complex)
        for i in range(k + 1):
            for j in range(half_nqubit**2):
                H_tmp[i] += Hvec_tmp[i][j] * H_basis[j]
        U_tmp = np.array([expm(1j * H_tmp[i]) for i in range(k + 1)])
        V_jf_tmp = np.zeros(
            (half_nqubit, half_nqubit, half_nqubit, half_nqubit),
            dtype=complex)
        for x in range(k + 1):
            for p, q, r, s, i, j in itertools.product(range(half_nqubit),
                                                      repeat=6):
                V_jf_tmp[p][q][r][s] += lam[x] * U_tmp[x][p][i] * (
                    np.conjugate(U_tmp[x])
                ).T[i][q] * J_tmp[x][i][j] * U_tmp[x][r][j] * (np.conjugate(
                    U_tmp[x])).T[j][s]
        return np.linalg.norm(V_jf_tmp - V)

    # 最適化を開始
    cost_history.append(1.)
    # cost_history.append(cost(J[:k + 1, :, :]))
    method = "BFGS"
    options = {"disp": True, "maxiter": 50, "gtol": 1e-6}

    # minimizeに使えるようにパラメーターを1列に並べなおす

    print("Hvec size={}".format(Hvec.size))
    Jvec_tmp = np.array(
        [symmetric_matrix_to_vector(J[i]) for i in range(k + 1)])
    datavec = np.append(
        Jvec_tmp.reshape((k + 1) * half_nqubit * (half_nqubit + 1) / 2),
        Hvec.reshape(k_jf, half_nqubit**2)[:k + 1, :].reshape(
            (k + 1) * half_nqubit**2))
    print("Start optimization:{0}".format(time.time() - start) + "[sec]")
    opt = minimize(cost, datavec, method=method)
    # callback=lambda y: cost_history.append(cost(y)))
    print("End optimization:{0}".format(time.time() - start) + "[sec]")
    datavec = opt.x
    Jvec_tmp = datavec[:(k + 1) * half_nqubit * (half_nqubit + 1) / 2].reshape(
        (k + 1, half_nqubit * (half_nqubit + 1) / 2))
    J = np.array(
        [vector_to_symmetric_matrix(Jvec_tmp[i]) for i in range(k + 1)])
    Hvec_opt = datavec[(k + 1) * half_nqubit * (half_nqubit + 1) / 2:].reshape(
        k + 1, -1)
    H = np.zeros((k + 1, half_nqubit, half_nqubit), dtype=complex)
    for i in range(k + 1):
        for j in range(half_nqubit**2):
            H[i] += Hvec_opt[i][j] * H_basis[j]
    U = np.array([expm(1j * H[i]) for i in range(k + 1)])

    # 最適化した結果を保存しておく
    with open("J_data/J_U_{}_k={}_d={}.csv".format(mole_name, k, d),
              'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(["J"])
        writer.writerows(J.reshape(k + 1, -1))
        writer.writerow(["H"])
        writer.writerows(H.reshape(k + 1, -1))
        writer.writerow(["U"])
        writer.writerows(U.reshape(k + 1, -1))

    V_jf = np.zeros((half_nqubit, half_nqubit, half_nqubit, half_nqubit),
                    dtype=complex)
    for x in range(k + 1):
        for p, q, r, s, i, j in itertools.product(range(half_nqubit),
                                                  repeat=6):
            V_jf[p][q][r][s] += lam[x] * U[x][p][i] * (np.conjugate(
                U[x])).T[i][q] * J[x][i][j] * U[x][r][j] * (np.conjugate(
                    U[x])).T[j][s]
    error_jfu.append(np.linalg.norm(V_jf - V))

elapsed_time = time.time() - start

# 結果を出力
with open("error_{}_{}sec.csv".format(mole_name, elapsed_time), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow([comment] + ["d="] + [d])
    writer.writerow(["k"] + range(1, k_svd + 1))
    writer.writerow(["error_SVD"] + error_svd)
    writer.writerow(["error_JF"] + error_jf)
    writer.writerow(["error_jfu"] + error_jfu)
    # writer.writerow(["cost_history"] + cost_history)

# 結果をプロット
plt.rcParams["font.size"] = 18
plt.plot(range(1, k_svd + 1), error_svd, label="error SVD", marker="x")
plt.plot(range(1, k_jf + 1), error_jf, label="error JF", marker="x")
plt.plot(range(1, k_jf + 1), error_jfu, label="error JF + U", marker="x")
# plt.plot(lam, color="blue", label="singular value")
# plt.plot(eigval, color="blue", label="eigenvalue")
plt.xlabel("$k_{SVD}$")
plt.ylabel("Error")
plt.legend(bbox_to_anchor=(0., 1.05),
           loc='lower left',
           borderaxespad=0,
           fontsize=18)
plt.savefig('error_{}_{}sec.png'.format(mole_name, elapsed_time),
            bbox_inches='tight')
plt.yscale('log')
plt.savefig('error_log_{}_{}sec.png'.format(mole_name, elapsed_time),
            bbox_inches='tight')

# plt.figure()
# plt.plot(cost_history, label="cost")
# plt.ylabel("Cost")
# plt.yscale('log')
# plt.savefig('cost_history_{}_{}sec.png'.format(mole_name, elapsed_time),
#             bbox_inches='tight')