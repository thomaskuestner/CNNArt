import numpy as np

def prox_l1(x, gamma):
    """
    The proximal operator for the l1_norm

    Parameters:
    -----------
    x : vector or matrix to calculate the norm from
    """
    sgn = np.sign(x)
    S = np.abs(x) - gamma
    S[S < 0.0] = 0.0
    return sgn * S

def prox_l1_01(x, gamma):
    """
    The proximal operator for the l1_norm with value constraint x in [0,1]

    Parameters:
    -----------
    x : vector or matrix to calculate the norm from
    """
    sgn = np.sign(x)
    S = np.abs(x) - gamma
    S[S < 0.0] = 0.0
    S = sgn * S
    S[S<0.0] = 0
    S[S>1.0] = 1
    return S

class ProximalGradSolver():
    def __init__(self, gamma, alpha, g, h, dg, prox_h):
        """
        Solves a problem of the form: min alpha*g(x) + h(x),
        where g(x) is smooth and differentiable and h(x) is non-differentiable (e.g. a regularization term or
        indicator function for a value constraint. Alpha is a trade-off variable and lambda is the step-size for
        the proximal gradient step.

        Parameters:
        -----------
        lambda : step size for proximal gradient step
        alpha : trade-off variable (problem dependent)
        g : the differentiable function
        dg : the first derivative of g
        prox_h : the proximal operator for h
        """
        self.gamma = gamma
        self.alpha = alpha
        self.g = g
        self.h = h
        self.dg = dg
        self.prox_h = prox_h

    def proximal_step(self, x, gamma):
        """
        Proximal gradient update step

        Parameters:
        -----------
        x : the current argument
        gamma : the step size
        """
        x_new = self.prox_h(x - gamma*self.alpha*self.dg(x), gamma)
        return x_new

    def minimize(self, x0, n_iter=50):
        """
        Proximal gradient minimization

        Parameters:
        -----------
        x0 : the initial argument
        n_iter : number of steps
        """
        result = x0
        for n in np.arange(n_iter):
            result = self.proximal_step(result, self.gamma)
            costG = self.g(result)
            costH = self.h(result)
            print("Iteration :"+str(n)+" with g(x): "+str(costG)+" and h(x): "+str(costH))

        return result
