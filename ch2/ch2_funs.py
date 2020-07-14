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


def du_crra(c, sigma, cstr_filler=9999.):
    """ Derivative of constant relative risk aversion utility function.

    Args:
        c: Consumption.
        sigma: Coefficient of relative risk aversion.
        cstr_filler: Filler to provide for nonpositive values of c.
            Defaults to 9999.

    Returns:
        Derivative of utility with respect to consumption.
    """
    if c <= 0:
        c = cstr_filler
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


def get_wpath(Kpath, alpha, A, L, m):
    """ Wage path given aggregate capital path.
    """
    wpath0 = np.power((1 - alpha) * A * Kpath / L, alpha)
    # Append the final value of wpath for each "extra" period.
    return np.append(wpath0, [wpath0[-1] for i in range(m)])


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


def get_rpath(Kpath, alpha, A, L, delta, m):
    """ Interest rate path given aggregate capital path.
    """
    rpath0 = alpha * A * np.power(L / Kpath, 1 - alpha) - delta
    # Append the final value of rpath for each "extra" period.
    return np.append(rpath0, [rpath0[-1] for i in range(m)])


def c(b_2, b_3, w, r, nvec):
    """ Consumption 
    """
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
    rhs_2_29 = beta * (1 + r) * du_crra(n_2 * w + (1 + r) * b_2 - b_3, sigma)
    lhs_2_30 = du_crra(n_2 * w + (1 + r) * b_2 - b_3, sigma)
    rhs_2_30 = beta * (1 + r) * du_crra((1 + r) * b_3 + n_3 * w, sigma)
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


def b_3_2_opt_fun(b_3_2, b0, wpath, rpath, beta, sigma, nvec):
    """ Function to root-find to get steady-state solution

    Args:
        b_3_2: Savings of middle-aged s=2 for the last period of his life.
        b0: Initial savings [b_2_1, b_3_1].
        wpath: Transition path of aggregate capital.
        rpath:
        beta:
        sigma:
        nvec:

    Returns:
        Euler error for optimal savings decision for the initial middle-aged 
        s = 2 individual for the last period of his life b_{3,2}.
    """
    c2 = nvec[1] * wpath[0] + (1 + rpath[0]) * b0[0] - b_3_2
    c3 = (1 + rpath[1]) * b_3_2 + nvec[2] * wpath[1]
    lhs = du_crra(c2, sigma)
    rhs = beta * (1 + rpath[1]) * du_crra(c3, sigma)
    # Ensure nonnegative consumption.
    if (c2 <= 0) or (c3 <= 0):
        return 9999.
    return lhs - rhs


def tpi_b_3_2(b_3_2_guess, b0, wpath, rpath, beta, sigma, nvec, tol):
    """ Solves for b_3_2, the first step of TPI.
    """
    return opt.root(b_3_2_opt_fun, b_3_2_guess,
                    (b0, wpath, rpath, beta, sigma, nvec),
                    tol=tol).x


def l2_norm(v1, v2):
    """ Sum of squared percent deviations.
    """
    return np.power(v1 / v2 - 1, 2).sum()


def tpi_pair_opt_fun(bvec, wpath, rpath, beta, sigma, nvec, t):
    """
    Args:
        bvec: Vector of two savings levels to solve for, e.g. b_2_2 and b_3_3.
        wpath:
        rpath:
        beta:
        sigma:
        nvec:
        t: Time period to solve for, i.e. b_{2,t} an b_{3,t+1}.

    Returns:
        List of Euler errors for two equations.
    """
    n_1, n_2, n_3 = nvec
    cvec = np.array([n_1 * wpath[t-2] - bvec[0],
                     n_2 * wpath[t-1] + (1 + rpath[t-1]) * bvec[0] - bvec[1],
                     (1 + rpath[t]) * bvec[1] + n_3 * wpath[t]])
    # Equation 2.32.
    lhs_2_32 = du_crra(cvec[0], sigma)
    rhs_2_32 = beta * (1 + rpath[t-1]) * du_crra(cvec[1], sigma)
    # Equation 2.33.
    lhs_2_33 = du_crra(cvec[1], sigma)
    rhs_2_33 = beta * (1 + rpath[t]) * du_crra(cvec[2], sigma)
    # Construct Euler errors.
    eul_errs = np.array([rhs_2_32 - lhs_2_32, rhs_2_33 - lhs_2_33])
    # Check that consumption is positive.
    cvec_cstr = cvec <= 0
    eul_errs[cvec_cstr[:-1]] = 9999.
    eul_errs[cvec_cstr[1:]] = 9999.
    return eul_errs


def tpi_pair(bvec_guess, wpath, rpath, beta, sigma, nvec, t, tol):
    """ Solve for a pair of savings at a time t.

    Args:
        bvec_guess: Guessed vector of two savings levels to solve for,
            e.g. b_2_2 and b_3_3.
        wpath:
        rpath:
        beta:
        sigma:
        nvec:
        t: Time period to solve for, i.e. b_{2,t} an b_{3,t+1}.
        tol: Tolerance.

    Returns:
        Result of opt.root running tpi_pair_opt_fun with provided args.
    """
    return opt.root(tpi_pair_opt_fun, bvec_guess,
        (wpath, rpath, beta, sigma, nvec, t),
        tol=tol).x


def tpi_iteration(b0, b_ss, Kpath, beta, sigma, nvec, alpha, A, L, delta, T, m,
                  tol):
    """ Single iteration.

    Returns:
        Path of savings bpath with shape (2, T+m).
    """
    # Calculate wpath and rpath.
    wpath = get_wpath(Kpath, alpha, A, L, m)
    rpath = get_rpath(Kpath, alpha, A, L, delta, m)
    # Initialize savings path with one column per age and one row per period.
    # e.g. bpath[0][1] corresponds to b_{2,3}.
    bpath = np.zeros([2, T + m - 1])
    # Calculate b_{3,2}.
    b_3_2_guess = b_ss[1]  # Guess the steady state.
    bpath[1][0] = tpi_b_3_2(b_3_2_guess, b0, wpath, rpath, beta, sigma, nvec,
                            tol)
    # Calculate remaining periods.
    for t in range(2, T + m):
        res = tpi_pair(b_ss, wpath, rpath, beta, sigma, nvec, t, tol)
        bpath[0][t-2] = res[0]
        bpath[1][t-1] = res[1]
    return bpath


def convex_combo(Kpath, Kpath_prime, xi):
    return xi * Kpath_prime + (1 - xi) * Kpath


def tpi(b0_ratios, bvec_guess, beta, sigma, nvec, L, A, alpha, delta, T, m, xi,
        tol):
    """ Full equilibrium transition path.

    Args:
        b0_ratios: Initial distribution of savings [b_{2,1}, b_{3,1}] as
            multiples of b_ss (steady-state values).
        bvec_guess: Initial guess for the steady-state savings vector.
        beta:
        sigma:
        nvec:
        L:
        A:
        alpha:
        delta:
        T: 
        m: Number of additional periods to solve for beyond T.
        xi: Kpath blending parameter.
        tol: Tolerance for calculating the steady state, b_{3,2},
            each [b_{2,t}, b_{3_t+1}], and Kpath.

    Returns:
        Equilibrium transition path of capital bpath, size (2, T+m),
        e.g. bpath[0][2] is b_{s=2, t=3}.
    """
    # Calculate steady state.
    ss = get_SS((beta, sigma, nvec, L, A, alpha, delta, tol), bvec_guess)
    b_ss = ss['b_ss']
    # Calculate b0.
    b0 = b_ss * b0_ratios
    # Calculate initial Kpath using linear interpolation.
    Kpath = np.linspace(sum(b0), sum(b_ss), num=T)
    Kpath_diff = tol + 1
    # Loop until Kpath convergence is reached.
    counter = 0
    while Kpath_diff > tol:
        bpath = tpi_iteration(b0, b_ss, Kpath, beta, sigma, nvec, alpha, A, L,
                              delta, T, m, tol)
        # Sum up the savings by period and drop extras.
        Kpath_prime = bpath.sum(axis=0)[:T]
        Kpath_diff = l2_norm(Kpath, Kpath_prime)
        Kpath = convex_combo(Kpath, Kpath_prime, xi)
        # Keep a counter to report status in case it takes too many iterations.
        counter += 1
        if counter % 100 == 0:
            print(str(counter) + ': ' + str(Kpath_diff))
        if counter > 1000:
            return
    return bpath