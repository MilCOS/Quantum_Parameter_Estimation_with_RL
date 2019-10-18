# lower level
import numpy as np
import qutip as qt


def tedious_sld(rho0, rho1, dw):

    zerodm = qt.ket2dm((qt.zero_ket(2)))
    rho_ave = rho0  # <<< shouldn't do "(rho0+rho1)/2" which influent the pureness or mixture of the state
    p_ave, psi_ave = rho_ave.eigenstates()
    p0, psi0 = rho0.eigenstates()  # calculate the eigenvalues, eigenstates
    p1, psi1 = rho1.eigenstates()  # calculate the eigenvalues, eigenstates
    mat_sld = zerodm  # init the SLD matrix

    # num_in_row = [0, 2, 4] # prepare row, col, data for sparse matirx (fast_csr format in Qutip)
    # col_idx = [0, 1, 0, 1]
    # data = []
    # print(p_ave)
    # print(p1-p0)
    for i in range(2):
        dlnp = ((p1[i] - p0[i]) / dw) / p_ave[i] if p_ave[i] > 10e-6 else 0
        ket1 = dlnp * qt.ket2dm(psi0[i])
        for j in range(2):
            delta = 0.0
            if i == j: delta = 1.0

            dpsi2 = (psi1[j] - psi0[j]) / dw
            pdp = 2 * (p_ave[j] - p_ave[i]) / (p_ave[j] + p_ave[i]) * (psi_ave[i].dag() * dpsi2).data[0, 0]  if (p_ave[j]+p_ave[i])>10e-6 else 0
            ket2 = pdp * psi_ave[i] * psi_ave[j].dag()

            mat_sld += ket1 * delta + ket2 * (1 - delta)

    # num_in_row = np.array(num_in_row,dtype='int32')
    # col_idx = np.array(col_idx,dtype='int32')
    # data = np.array(data,dtype=type(data[0]))
    # mat_sld.data = qt.sparse.fast_csr_matrix((data,col_idx,num_in_row)) # construct fast_csr_matrix
    return mat_sld, rho_ave


def pure_sld(rho0, rho1, dw):

    return 2 * (rho1 - rho0) / dw, (rho1 + rho0) / 2


def qfisher(rho, sld):

    return (qt.expect(sld * sld, rho)).real


def qfisher2(rho0, rho1, dw):
    rho_ave = rho0  # <<< shouldn't do "(rho0+rho1)/2" which influent the pureness or mixture of the state
    p_ave, psi_ave = rho_ave.eigenstates()
    p0, psi0 = rho0.eigenstates()  # calculate the eigenvalues, eigenstates
    p1, psi1 = rho1.eigenstates()  # calculate the eigenvalues, eigenstates

    qfi = 0
    for i in range(2):
        # dpsi1 = (psi1[i] - psi0[i])/dw
        dlnp = ((p1[i] - p0[i]) / dw) ** 2 / p_ave[i] if p_ave[i] >= 10e-6 else 0
        qfi += dlnp
        for j in range(2):
            delta = 0.0
            if i == j: delta = 1.0

            dpsi2 = (psi1[j] - psi0[j]) / dw
            # print(i,j,': ',(dpsi2.dag()*psi0[i]).data[0,0])
            dsec = (1 - delta) * 2 * (p_ave[j] - p_ave[i]) ** 2 / (p_ave[j] + p_ave[i]) * np.absolute(
                (psi_ave[i].dag() * dpsi2).data[0, 0]) ** 2 if (p_ave[j]+p_ave[i])>10e-6 else 0
            qfi += dsec
            # mat_sld.data[i,j]=elem
    return qfi.real


def cfisher(POVM, rho0, rho1, dw):
    cfi = 0
    # rho_ave = (rho0+rho1)/2
    rho_ave = rho0  # <<< shouldn't do "(rho0+rho1)/2" which influent the pureness or mixture of the state
    for povm in POVM:
        p0 = qt.expect(povm, rho0)
        p1 = qt.expect(povm, rho1)
        p_ave = qt.expect(povm, rho_ave)

        dp = (np.log(p1 / p0)) / dw

        cfi += dp ** 2 * p_ave

    return cfi.real

