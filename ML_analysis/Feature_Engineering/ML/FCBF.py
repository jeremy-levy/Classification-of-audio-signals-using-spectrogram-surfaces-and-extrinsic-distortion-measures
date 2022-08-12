from ML.__mRMR import SU


def FCBF(X, C, features_name, delta=0.05):
    """
    https://www.aaai.org/Papers/ICML/2003/ICML03-111.pdf
    """

    N = X.shape[1]
    S_list = []
    for i in range(N):
        SU_i_c = SU(X[:, i], C)

        if SU_i_c > delta:
            S_list.append((i, SU_i_c, features_name[i]))
    
    S_list = sorted(S_list, key=lambda x: x[1], reverse=True)
    S_list.append(("end", ""))
    S_list.append(("end", ""))
 
    idx_p = 0
    Fp = S_list[idx_p]
    
    while Fp[0] != "end":
        idx_q = idx_p + 1
        Fq = S_list[idx_q]
        while Fq[0] != "end":
            if SU(X[:, Fp[0]], X[:, Fq[0]]) > Fq[1]:
                S_list.remove(Fq)
                Fq = S_list[idx_q]
            else:
                idx_q += 1
                Fq = S_list[idx_q]
        Fp = S_list[idx_p]
        idx_p += 1
    
    S_list.remove(("end", ""))
    S_list.remove(("end", ""))
    return S_list