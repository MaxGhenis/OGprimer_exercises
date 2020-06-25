import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time


def u_crra(c, sigma):
	""" Constant relative risk aversion utility function.

	Args:
		c: Consumption.
		sigma: Coefficient of relative risk aversion.

	Returns:
		Utility.
	"""
    return (c ** (1 - sigma) - 1) / (1 - sigma)



def du_crra(c, sigma):
	""" Derivative of constant relative risk aversion utility function.

	Args:
		c: Consumption.
		sigma: Coefficient of relative risk aversion.

	Returns:
		Derivative of utility with respect to consumption.
	"""
    return c ** (-sigma)


def w_t(b_2_t, b_3_t, alpha, A, n):
    """ Equilibrium wage rate as a function of the distribution of capital,
        capital income share, productivity, and total labor.
    
    Args:
        b_2_t: Savings at age 2 as of time t.
        b_3_t: Savings at age 3 as of time t.
        alpha: Capital income share.
        A: Productivity.
        n: Total labor. 

    Returns:
        Equilibrium wage rate as of time t.
    """
    return (1 - alpha) * A * ((b_2_t + b_3_t) / n) ** alpha


def wpath(Kpath, alpha, A, L):
    """ Wage path given aggregate capital path.
    """
    return np.power((1 - alpha) * A * Kpath / L, alpha)


def r_t(b_2_t, b_3_t, alpha, A, L, delta):
    """ Equilibrium interest rate as a function of the distribution of capital,
        capital income share, productivity, total labor, and depreciation rate.
    
    Args:
        b_2_t: Savings at age 2 as of time t.
        b_3_t: Savings at age 3 as of time t.
        alpha: Capital income share.
        A: Productivity.
        n: Total labor.
        delta: Depreciation rate.

    Returns:
        Equilibrium interest rate as of time t.
    """
    return alpha * A * (L / (b_2_t + b_3_t)) ** (1 - alpha) - delta


def rpath(Kpath, alpha, A, L, delta):
    """ Interest rate path given aggregate capital path.
    """
    return alpha * A * np.power(L / Kpath, 1 - alpha) - delta


def c(b_2, b_3, w, r, nvec):
    c_1 = nvec[0] * w - b_2
    c_2 = nvec[1] * w + (1 + r) * b_2 - b_3
    c_3 = nvec[2] * w + (1 + r) * b_3
    return c_1, c_2, c_3


def feasible(f_params, bvec_guess):
    """ Identifies whether constraints can be satisfied given economic
        parameters and an initial guess of steady-state savings.

    Args:
        f_params: Tuple (nvec, A, alpha, delta).
        bvec_guess: np.array([scalar, scalar]).
    
    Returns:
        Tuple of Boolean vectors:
        - b_cnstr (length 2, denotes which element of bvec_guess is likely
                   responsible for consumption nonnegativity constraint
                   violations identified in c_cnstr)
        - c_cnstr (length 3, true if c_s <= 0)
        - K_cnstr (length 1, true if K <= 0)
    """
    # Extract params.
    nvec, A, alpha, delta = f_params
    b_2, b_3 = bvec_guess
    # Calculate total labor supply.
    L = sum(nvec)
    # Calculate equilibrium wage and interest rates.
    w = w_t(b_2, b_3, alpha, A, L)
    r = r_t(b_2, b_3, alpha, A, L, delta)
    # Calculate consumption levels via Equation 2.7.
    c_1, c_2, c_3 = c(b_2, b_3, w, r, nvec)
    # Calculate K via market-clearing condition, Equation 2.23.
    K = b_2 + b_3
    # Define constraints violations.
    c_cnstr = [c_1 < 0, c_2 < 0, c_3 < 0]
    K_cnstr = K < 0
    # Identify the element of bvec_guess likely responsible.
    b_cnstr = [False, False]
    if c_1 < 0:
        b_cnstr[0] = True
    if c_2 < 0:
        b_cnstr = [True, True]
    if c_3 < 0:
        b_cnstr[1] = False
    # Return
    return b_cnstr, c_cnstr, K_cnstr


def infeasible(f_params, bvec_guess):
    """ Simple True/False indicating whether it's infeasible.
    """
    b_cnstr, c_cnstr, K_cnstr = feasible(f_params, bvec_guess)
    return any([any(b_cnstr), any(c_cnstr), K_cnstr])


def ss_opt_fun(bvec, beta, sigma, nvec, L, A, alpha, delta):
    """ Function to root-find to get steady-state solution.

    Args:
        bvec: Vector of steady-state savings [b2, b3].
        params: Tuple of (beta, sigma, nvec, L, A, alpha, delta).

    Returns:
        Vector of Euler errors.
    """
    # (nvec, A, alpha, delta
    if infeasible((nvec, A, alpha, delta), bvec):
        return [100, 100]
    b_2, b_3 = bvec
    n_1, n_2, n_3 = nvec
    # Start with LHS and RHS of equations 2.29 and 2.30.
    # Replace 1/1/0.2 with n1/n2/n3.
    # 2.29: u'(n_1*w(b_2,b_3)-b_2) =
    #       beta*(1+r(b_2,b_3))u'(n_2*w(b_2,b_3)+[1+r(b_2,b_3)]b_2-b_3)
    # 2.30: u'(n_2*w(b_2,b_3)+[1+r(b_2,b_3)]b_2-b_3)=
    #       B(1+r(b_2,b_3))u'([1+r(b_2,b_3)]b_3+n_3*w(b_2,b_3))
    w = w_t(b_2, b_3, alpha, A, L)
    r = r_t(b_2, b_3, alpha, A, L, delta)
    lhs_2_29 = du_crra(n_1 * w - b_2, sigma)
    rhs_2_29 = beta * (1 + r) * du_crra(n_2 * w + (1 + r) * b_2 - b_3)
    lhs_2_30 = du_crra(n_2 * w + (1 + r) * b_2 - b_3, sigma)
    rhs_2_30 = beta * (1 + r) * du_crra((1 + r) * b_3 + n_3 * w)
    return [lhs_2_29 - rhs_2_29, lhs_2_30 - rhs_2_30]

def get_SS(params, bvec_guess, SS_graphs=False):
    """ Calculates steady-state solution.

    Args:
        params: Tuple of (beta, sigma, nvec, L, A, alpha, delta, SS_tol).
        bvec_guess: Initial guess of steady-state savings.
        SS_graphs: Boolean that generates a figure of the steady-state
            distribution of consumption and savings if set to True.
            Defaults to False.
    
    Returns:
        Dictionary with the steady-state solution values for the
        following endogenous objects:
        b_ss: Length-2 vector of steady-state savings b_2_ss and b_3_ss.
        c_ss: Length-3 vector of steady-state consumption.
        w_ss: Steady-state wage.
        r_ss: Steady-state interest rate.
        K_ss: Steady-state capital.
        Y_ss: Steady-state income.
        C_ss: Steady-state consumption  (how does this differ from c_ss?).
        EulErr_ss: Length-2 vector of the two Euler errors from the resulting
            steady-state solution given in difference form
            beta(1+r)u'(c_{s+1}) - u'(c_s).
        RCerr_ss: Resource constraint error which should be close to zero.
            It is given by Y-C-delta*K.
        ss_time: Run time in seconds.
    """
    # Extract parameters.
    start_time = time.clock()
    beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    # Solve.
    opt_res = opt.root(ss_opt_fun, bvec_guess, params[:-1], tol=SS_tol)
    # Extract features from the optimization result.
    b2_ss, b3_ss = b_ss = opt_res.x
    EulErr_ss = opt_res.fun
    # Calculate other quantities.
    w_ss = w_t(b2_ss, b3_ss, alpha, A, L)
    r_ss = r_t(b2_ss, b3_ss, alpha, A, L, delta)
    c_ss = c(b2_ss, b3_ss, w_ss, r_ss, nvec)
    K_ss = sum(b_ss)
    C_ss = sum(c_ss)
    # Calculate output via production function, not market-clearing.
    Y_ss = A * (K_ss ** alpha) * (L ** (1 - alpha))
    RCerr_ss = Y_ss - C_ss - delta * K_ss
    ss_time = time.clock() - start_time
    if SS_graphs:
        pd.Series(c_ss).plot.bar()
        plt.title('Consumption')
        plt.show()
        pd.Series(b_ss).plot.bar()
        plt.title('Savings')
        plt.show()
    return {'b_ss': b_ss,
            'c_ss': c_ss,
            'w_ss': w_ss,
            'r_ss': r_ss,
            'K_ss': K_ss,
            'Y_ss': Y_ss,
            'C_ss': C_ss,
            'EulErr_ss': EulErr_ss,
            'RCerr_ss': RCerr_ss,
            'ss_time': ss_time
            }


def b_3_2_opt_fun(b_3_2, bvec, Kpath, beta, sigma, nvec, L, A, alpha, delta):
    """ Function to root-find to get steady-state solution.

    Args:
        b_3_2: Savings of middle-aged s=2 for the last period of his life.
        bvec: Initial savings [b_2_1, b_3_1].
        Kpath: Transition path of aggregate capital.
        beta:
        sigma:
        nvec:
        L:
        A:
        alpha:
        delta:

    Returns:
        Euler error.
    """
    # # (nvec, A, alpha, delta
    # if infeasible((nvec, A, alpha, delta), bvec):
    #     return [100, 100]
    b_2_1, b_3_1 = bvec
    n_1, n_2, n_3 = nvec
    w = wpath(Kpath, alpha, A, L)
    r = rpath(Kpath, alpha, A, L, delta)
    lhs = du_crra(n_2 * w[0] + (1 + r[0]) * b_2_1 - b_3_2, sigma)
    rhs = beta * (1 + r[1]) * du_crra((1 + r[2] * b_3_2 + n_2 * w[1]), sigma)
    return [lhs - rhs]


def tpi_b_3_2(b_3_2_guess, bvec, Kpath, beta, sigma, nvec, L, A, alpha, delta,
              tol):
    """ Solves for b_3_2, the first step of TPI.
    """
    return opt.root(b_3_2_opt_fun, b_3_2_guess,
                    (bvec, Kpath, beta, sigma, nvec, L, A, alpha, delta),
                    tol=tol).x
