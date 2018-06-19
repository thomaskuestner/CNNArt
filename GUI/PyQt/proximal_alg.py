import numpy as np

def prox_l1(x, gamma):

    sgn = np.sign(x)
    S = np.abs(x) - gamma
    S[S < 0.0] = 0.0
    return sgn * S  

def prox_l1_01(x, gamma):

    sgn = np.sign(x)
    S = np.abs(x) - gamma
    S[S < 0.0] = 0.0
    S = sgn * S
    S[S<0.0] = 0
    S[S>1.0] = 1
    return S

class ProximalGradSolver():
    def __init__(self, gamma, alpha, g, h, dg, prox_h):

        self.gamma = gamma
        self.alpha = alpha
        self.g = g
        self.h = h
        self.dg = dg
        self.prox_h = prox_h

    def proximal_step(self, x, gamma):

        x_new = self.prox_h(x - gamma*self.alpha*self.dg(x), gamma)
        return x_new

    def minimize(self, x0, n_iter=50):

       result = x0
       for n in np.arange(n_iter):
          result = self.proximal_step(result, self.gamma)
          costG = self.g(result)
          costH = self.h(result)
          # print("Iteration :"+str(n)+" with g(x): "+str(costG)+" and h(x): "+str(costH))
       return result