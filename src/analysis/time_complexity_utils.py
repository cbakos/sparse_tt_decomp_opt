import numpy as np


def tt_solve_time_complexity(s: float, r: float, I: float, d: float, num_iterations: float) -> float:
    t1 = s ** 3 * r ** 2 * I ** 2  # prepare local system
    t2 = s ** 3 * r * I ** 3  # in-between contractions
    t3 = s ** 6 * I ** 6  # direct solve of local system
    t4 = s ** 3 * I ** 3  # SVD truncation
    return num_iterations * d * (t1 + t2 + t3 + t4)


def matrix2mpo_time_complexity(z: float, r: float, I: float, d: float) -> float:
    return z + d * I ** 2 * r ** 2


def tt_svd_vec_time_complexity(s: float, I: float, n: float) -> float:
    return n * I * s ** 3


def tt_svd_mat_time_complexity(r: float, I: float, n: float) -> float:
    return n ** 2 * I ** 2 * r ** 3


def tt_sol2vec_sol_time_complexity(s: float, d: float, n: float) -> float:
    return d * n * s ** 4


def cg_time_complexity(z: float, cond_num: float) -> float:
    return z * np.sqrt(cond_num)


# Setting 1: we can keep s, number of iterations, d are log(n)
def tt_solve_time_complexity_log_assumption(r: float, I: float, n: float) -> float:
    return tt_solve_time_complexity(s=np.log(n), r=r, I=I, d=np.log(n), num_iterations=np.log(n))


def tt_solve_time_complexity_full_log_assumption(n: float) -> float:
    return tt_solve_time_complexity_log_assumption(r=np.log(n), I=np.log(n), n=n)


# assume log(n) except TT-ranks, those are const, s=2
def tt_solve_time_complexity_full_const_s_log_assumption(n: float) -> float:
    return tt_solve_time_complexity(s=2, r=np.log(n), I=np.log(n), d=np.log(n), num_iterations=np.log(n))


# assume log(n) except: s=2, r=2
def tt_solve_time_complexity_full_const_I_s_log_assumption(n: float) -> float:
    return tt_solve_time_complexity(s=2, r=np.log(n), I=2, d=np.log(n), num_iterations=np.log(n))


# assume log(n) except: I=2
def tt_solve_time_complexity_full_const_I_log_assumption(n: float, I: float) -> float:
    return tt_solve_time_complexity(s=np.log(n), r=np.log(n), I=I, d=np.log(n), num_iterations=np.log(n))


def tt_solve_with_conversions_time_complexity(s: float, r: float, I: float, d: float, num_it: float, z: float,
                                              n: float) -> float:
    v = (tt_solve_time_complexity(s=s, r=r, I=I, d=d, num_iterations=num_it) +
         matrix2mpo_time_complexity(z=z, r=r, I=I, d=d) +
         tt_sol2vec_sol_time_complexity(s=s, d=d, n=n) +
         tt_svd_vec_time_complexity(s=s, I=I, n=n))
    return v


def tt_solve_with_conversions_time_complexity_some_log(s: float, r: float, I: float, n: float) -> float:
    return tt_solve_with_conversions_time_complexity(s=s, r=r, I=I, n=n, d=np.log(n), num_it=np.log(n), z=n)
