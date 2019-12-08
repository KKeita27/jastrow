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

#define constants
basis = "sto-3g"  #basis set
multiplicity = 1  #spin multiplicity = 2S+1
charge = 0  #total charge for the molecule

# input for H2
dist = np.arange(0.4, 4.0, .2)  # distance list
nqubit = 4
mole_name = "H2"
mole1 = "H"
mole2 = "H"
k_svd = 3  # expansion order for SVD
k_jf = k_svd  # expansion order for JF(U)
num_states = 2  # number of states to record the energy

# # input for LiH
# dist = np.arange(0.7, 4.0, .4)  # distance list
# # dist = [1.5]
# nqubit = 12
# mole_name = "LiH"
# mole1 = "Li"
# mole2 = "H"
# k_svd = 1  # expansion order for SVD
# k_jf = k_svd  # expansion order for JF(U)
# num_states = 2  # number of states to record the energy

# dist = np.arange(0.7, 3.5, .5)  # for H2O
# # dist = [1.]
# ang = 120.
# nqubit = 14
# mole_name = "H2O"
# mole1 = "O"
# mole2 = "H"
# mole3 = "H"

half_nqubit = nqubit / 2

dist_graph = []
energy_fci_hist = [[], [], [], []]
energy_svd_hist = [[], [], [], []]
energy_jf_hist = [[], [], [], []]
timer_label = [
    "FCI eigensolve", "SVD", "SVD eigensolve", "optimization",
    "optimization eigensolve"
]

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

import csv

# preparation to record the elapsed times
with open("time_JFU_{}_k={}.csv".format(mole_name, k_svd), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow([mole_name])
    writer.writerow(["k", k_svd, "num_states", num_states])
    writer.writerow(["", ""] + timer_label)

# compute energies for each bond length
for d in dist:
    # prepare timers
    timer_start = np.zeros(5)
    timer_end = np.zeros(5)

    # record d for the graph
    dist_graph.append(d)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print("d={}".format(d))

    # prepare the molecule and its Hamiltonian
    distance = d
    geometry = [
        (mole1, (0, 0, 0)),
        (mole2, (0, 0, distance)),  #xyz coordinates for atoms
    ]  #xyz coordinates for atoms
    description = str(distance)  #description for the psi4 output file
    molecule = MolecularData(geometry, basis, multiplicity, charge,
                             description)
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    fermionic_hamiltonian = get_fermion_operator(
        molecule.get_molecular_hamiltonian())

    # ハミルトニアンを行列に変換
    h = np.array([0, [], [[]], [[[]]], [[[[]]]]])
    h[0] = fermionic_hamiltonian.terms[()]
    h[1] = np.zeros(nqubit)
    h[2] = np.zeros((nqubit, nqubit))
    h[3] = np.zeros((nqubit, nqubit, nqubit))
    h[4] = np.zeros((nqubit, nqubit, nqubit, nqubit))
    for i, j in itertools.product(range(nqubit), repeat=2):
        h[2][i][j] = fermionic_hamiltonian.terms.get(((i, 1), (j, 0)), 0)
    for i, j, k, l in itertools.product(range(nqubit), repeat=4):
        h[4][i][j][k][l] = fermionic_hamiltonian.terms.get(
            ((i, 1), (j, 1), (k, 0), (l, 0)), 0)

    # FCI解を対角化により得る
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    hamiltonian_matrix = get_sparse_operator(jw_hamiltonian)
    from scipy.sparse.linalg import eigsh

    elapsed_time = time.time() - start
    timer_start[0] = elapsed_time

    eigval, eigvec = eigsh(hamiltonian_matrix, k=num_states, which="SA")

    elapsed_time = time.time() - start
    timer_end[0] = elapsed_time

    for i in range(num_states):
        energy_fci_hist[i].append(eigval[i])

    # ここからSVD
    # ハミルトニアンをchemist orderingに変換 (h2_correctionは不使用)
    h2_correction, V = get_chemist_two_body_coefficients(h[4], spin_basis=True)
    Vmat = V.reshape(half_nqubit**2, half_nqubit**2)

    print("Start SVD for d={}: {}".format(d, time.time() - start) + "[sec]")

    elapsed_time = time.time() - start
    timer_start[1] = elapsed_time

    # SVDを実行
    u1, lam, u2 = np.linalg.svd(Vmat)

    elapsed_time = time.time() - start
    timer_end[1] = elapsed_time

    # k_svdまでのSVDで近似したものをV_svdとする
    V_svd = np.tensordot(u1[:, 0], u2[0, :], 0) * lam[0]
    for i in range(1, k_svd):
        V_svd += np.tensordot(u1[:, i], u2[i, :], 0) * lam[i]
    V_svd = V_svd.reshape((half_nqubit, half_nqubit, half_nqubit, half_nqubit))

    # ハミルトニアンh[4]に近似したVを戻す。空間軌道に対応するスピン軌道4つ分に同じ値を代入。
    for p, q, r, s in itertools.product(range(half_nqubit), repeat=4):
        h[4][2 * p][2 * q][2 * r][2 * s] = V_svd[p][s][q][r]
        h[4][2 * p + 1][2 * q][2 * r][2 * s + 1] = V_svd[p][s][q][r]
        h[4][2 * p][2 * q + 1][2 * r + 1][2 * s] = V_svd[p][s][q][r]
        h[4][2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = V_svd[p][s][q][r]

    # さらにh[4]をハミルトニアンに戻す
    for i, j, k, l in itertools.product(range(nqubit), repeat=4):
        fermionic_hamiltonian.terms[((i, 1), (j, 1), (k, 0),
                                     (l, 0))] = h[4][i][j][k][l]

    # 対角化を実行してエネルギーを記録。
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    hamiltonian_matrix = get_sparse_operator(jw_hamiltonian)
    from scipy.sparse.linalg import eigsh
    elapsed_time = time.time() - start
    timer_start[2] = elapsed_time
    eigval, eigvec = eigsh(hamiltonian_matrix, k=num_states, which="SA")
    elapsed_time = time.time() - start
    timer_end[2] = elapsed_time
    for i in range(num_states):
        energy_svd_hist[i].append(eigval[i])

    # ここからJF + U

    # 第一添え字をxにするため
    u1 = u1.T

    u1mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
    u2mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)

    u11mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
    # u12mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)

    u21mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
    # u22mat = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)

    w1 = np.empty((k_svd, half_nqubit), dtype=complex)
    w2 = np.empty((k_svd, half_nqubit), dtype=complex)

    for i in range(k_svd):
        u1mat[i] = u1[i].reshape((half_nqubit, half_nqubit))
        w1[i], u11mat[i] = np.linalg.eig(u1mat[i])
        # u12mat[i] = np.linalg.inv(u11mat[i])
        u2mat[i] = u2[i].reshape((half_nqubit, half_nqubit))
        w2[i], u21mat[i] = np.linalg.eig(u2mat[i])
        # u22mat[i] = np.linalg.inv(u21mat[i])

    # J = np.empty((k_svd, half_nqubit, half_nqubit), dtype=complex)
    # for i in range(k_svd):
    #     J[i] = np.tensordot(w1[i], w1[i], 0)

    # V_jf = np.zeros((half_nqubit, half_nqubit, half_nqubit, half_nqubit),
    #                 dtype=complex)
    # for x in range(k_svd):
    #     for p, q, r, s in itertools.product(range(half_nqubit), repeat=4):
    #         for i in range(half_nqubit):
    #             for j in range(half_nqubit):
    #                 V_jf[p][q][r][s] += lam[x] * u11mat[x][p][i] * u12mat[x][
    #                     i][q] * J[x][i][j] * u21mat[x][r][j] * u22mat[x][j][s]

    # print("{} (should be close to zero)".format(np.linalg.norm(V_jf - V_svd)))
    # →きちんと1e-16程度になる

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

    # ユニタリ行列Uについて、U=exp(iH)となるようなHを返す。
    def log_of_unitary(U):
        Lam, U1 = np.linalg.eig(U)
        logLam = [np.log(x).imag for x in Lam]
        return U1.dot(np.diag(logLam).dot(np.conjugate(U1.T)))

    # Hermite行列からユニタリ行列を作るために使用
    from scipy.linalg import expm

    # Hermiteの基底で展開（フィット）
    H = np.array([log_of_unitary(u11mat[i]) for i in range(k_jf)])

    #成分を拾ってきて初期値とする。Hは初めは実行列なはずだが、
    # 誤差の関係か虚部も持ってしまうことが多い。
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
    print("cost={}".format(np.linalg.norm(H - H_tmp)))

    def cost_H(Hvec_tmp):
        H_tmp = np.zeros((k_jf, half_nqubit, half_nqubit), dtype=complex)
        Hvec_tmp = Hvec_tmp.reshape(k_jf, half_nqubit**2)
        for i in range(k_jf):
            for j in range(half_nqubit**2):
                H_tmp[i] += Hvec_tmp[i][j] * H_basis[j]
        return np.linalg.norm(H - H_tmp)

    method = "BFGS"
    options = {"disp": True, "maxiter": 50, "gtol": 1e-6}

    # minimizeに使えるようにパラメーターを1列に並べなおす
    print("Start H optimization:{0}".format(time.time() - start) + "[sec]")
    opt = minimize(cost_H, Hvec, method=method)
    # callback=lambda y: cost_history.append(cost_H(y)))
    print("End H optimization:{0}".format(time.time() - start) + "[sec]")
    Hvec = opt.x

    # ここまでHのフィット

    # ここからJとUの最適化開始

    J = np.empty((k_svd, half_nqubit, half_nqubit), dtype=float)
    for i in range(k_svd):
        # w2[i]はw1[i]を用いても良いはず
        J[i] = np.tensordot(w1[i], w2[i], 0)

    # k = k_jf - 1

    # J, U を最適化する際のコスト関数を定義
    def cost(data_tmp):
        # Jについてのデータを受け取る
        Jvec_tmp = data_tmp[:k_jf * half_nqubit * (half_nqubit + 1) /
                            2].reshape(
                                (k_jf, half_nqubit * (half_nqubit + 1) / 2))
        J_tmp = np.array(
            [vector_to_symmetric_matrix(Jvec_tmp[i]) for i in range(k_jf)])

        # Hについてのデータを受け取り、Uを作る
        Hvec_tmp = data_tmp[k_jf * half_nqubit * (half_nqubit + 1) /
                            2:].reshape(k_jf, half_nqubit**2)
        H_tmp = np.zeros((k_jf, half_nqubit, half_nqubit), dtype=complex)
        for i in range(k_jf):
            for j in range(half_nqubit**2):
                H_tmp[i] += Hvec_tmp[i][j] * H_basis[j]
        U_tmp = np.array([expm(1j * H_tmp[i]) for i in range(k_jf)])

        # 以上のデータをもとにVを作り、もとのものとの差を返す
        V_jf_tmp = np.zeros(
            (half_nqubit, half_nqubit, half_nqubit, half_nqubit),
            dtype=complex)
        for x in range(k_jf):
            for p, q, r, s, i, j in itertools.product(range(half_nqubit),
                                                      repeat=6):
                V_jf_tmp[p][q][r][s] += lam[x] * U_tmp[x][p][i] * (
                    np.conjugate(U_tmp[x])
                ).T[i][q] * J_tmp[x][i][j] * U_tmp[x][r][j] * (np.conjugate(
                    U_tmp[x])).T[j][s]
        return np.linalg.norm(V_jf_tmp - V)

    # 最適化を開始
    method = "BFGS"
    options = {"disp": True, "maxiter": 50, "gtol": 1e-6}

    # minimizeに使えるようにパラメーターを1列に並べなおす
    Jvec_tmp = np.array(
        [symmetric_matrix_to_vector(J[i]) for i in range(k_jf)])
    datavec = np.append(
        Jvec_tmp.reshape(k_jf * half_nqubit * (half_nqubit + 1) / 2),
        Hvec.reshape(k_jf,
                     half_nqubit**2)[:k_jf, :].reshape(k_jf * half_nqubit**2))
    print("Start optimization:{0}".format(time.time() - start) + "[sec]")
    opt = minimize(cost, datavec, method=method)
    # callback=lambda y: cost_history.append(cost(y)))
    print("End optimization:{0}".format(time.time() - start) + "[sec]")
    datavec = opt.x

    # 最適化の結果からJとUを作成
    Jvec_tmp = datavec[:k_jf * half_nqubit * (half_nqubit + 1) / 2].reshape(
        (k_jf, half_nqubit * (half_nqubit + 1) / 2))
    J = np.array(
        [vector_to_symmetric_matrix(Jvec_tmp[i]) for i in range(k_jf)])
    Hvec_opt = datavec[k_jf * half_nqubit * (half_nqubit + 1) / 2:].reshape(
        k_jf, -1)
    H = np.zeros((k_jf, half_nqubit, half_nqubit), dtype=complex)
    for i in range(k_jf):
        for j in range(half_nqubit**2):
            H[i] += Hvec_opt[i][j] * H_basis[j]
    U = np.array([expm(1j * H[i]) for i in range(k_jf)])

    # 最適化した結果を保存
    with open("J_data/J_U_{}_k={}_d={}.csv".format(mole_name, k_svd, d),
              'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(["J"])
        writer.writerows(J.reshape(k_jf, -1))
        writer.writerow(["H"])
        writer.writerows(H.reshape(k_jf, -1))

    # JとUを最適化したVを計算
    V_jf = np.zeros((half_nqubit, half_nqubit, half_nqubit, half_nqubit),
                    dtype=complex)
    for x in range(k_jf):
        for p, q, r, s, i, j in itertools.product(range(half_nqubit),
                                                  repeat=6):
            V_jf[p][q][r][s] += lam[x] * U[x][p][i] * (np.conjugate(
                U[x])).T[i][q] * J[x][i][j] * U[x][r][j] * (np.conjugate(
                    U[x])).T[j][s]

    # ハミルトニアンに最適化したVを戻して、ハミルトニアンの固有値を計算
    for p, q, r, s in itertools.product(range(half_nqubit), repeat=4):
        h[4][2 * p][2 * q][2 * r][2 * s] = V_jf[p][s][q][r]
        h[4][2 * p + 1][2 * q][2 * r][2 * s + 1] = V_jf[p][s][q][r]
        h[4][2 * p][2 * q + 1][2 * r + 1][2 * s] = V_jf[p][s][q][r]
        h[4][2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = V_jf[p][s][q][r]

        for i, j, k, l in itertools.product(range(nqubit), repeat=4):
            fermionic_hamiltonian.terms[((i, 1), (j, 1), (k, 0),
                                         (l, 0))] = h[4][i][j][k][l]

    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    hamiltonian_matrix = get_sparse_operator(jw_hamiltonian)
    from scipy.sparse.linalg import eigsh
    elapsed_time = time.time() - start
    timer_start[4] = elapsed_time
    eigval, eigvec = eigsh(hamiltonian_matrix, k=num_states, which="SA")
    elapsed_time = time.time() - start
    timer_end[4] = elapsed_time
    for i in range(num_states):
        energy_jf_hist[i].append(eigval[i])

    # ここからはプロットを行う

    cmap = plt.get_cmap("tab10")
    colors = [0, 3, 2, 1]

    linestyle_fci = "solid"
    linestyle_svd = "dotted"
    linestyle_jf = "dotted"

    markerfacecolor_svd = "white"

    marker_fci = "|"
    marker_svd = "o"
    marker_jf = "o"

    plt.figure()
    plt.rcParams["font.size"] = 18
    for i in range(num_states):
        plt.plot(dist_graph,
                 energy_fci_hist[i],
                 label="FCI {}-excitation".format(i),
                 color=cmap(colors[i]),
                 linestyle=linestyle_fci,
                 marker=marker_fci,
                 linewidth=2.0)
        plt.plot(dist_graph,
                 energy_svd_hist[i],
                 label="SVD {}-excitation".format(i),
                 color=cmap(colors[i]),
                 linestyle=linestyle_svd,
                 marker=marker_svd,
                 markerfacecolor=markerfacecolor_svd,
                 fillstyle="none",
                 linewidth=2.0)
        plt.plot(dist_graph,
                 energy_jf_hist[i],
                 label="JF+U {}-excitation".format(i),
                 color=cmap(colors[i]),
                 linestyle=linestyle_jf,
                 marker=marker_jf,
                 fillstyle="full",
                 linewidth=2.0)
    plt.xlabel("distance")
    plt.ylabel("Energy expectation value")
    plt.legend(bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderaxespad=0,
               fontsize=18)
    plt.savefig("potential_JFU_{}_k={}.png".format(mole_name, k_svd),
                bbox_inches='tight')

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # 結果をcsvにして保存する
    with open("potential_JFU_{}_k={}.csv".format(mole_name, k_svd),
              'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(dist_graph)
        writer.writerow(["FCI energy"])
        writer.writerows(energy_fci_hist)
        writer.writerow(["SVD energy"])
        writer.writerows(energy_svd_hist)
        writer.writerow(["JF+U energy"])
        writer.writerows(energy_jf_hist)

    with open("time_JFU_{}_k={}.csv".format(mole_name, k_svd), 'a') as file:
        writer = csv.writer(file, lineterminator='\n')
        # writer.writerow(["d", d])
        writer.writerow([d] + ["start"] + timer_start.tolist())
        writer.writerow([d] + ["end"] + timer_end.tolist())

with open("time_JFU_{}_k={}.csv".format(mole_name, k_svd), 'a') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(["total time", elapsed_time])