# import theano
# import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b
import ...deepvis.proximal_alg
import numpy as np
import tensorflow as tf

class TensorFlowTheanoFunction(object):
  def __init__(self, inputs, outputs):
    self._inputs = inputs
    self._outputs = outputs

  def __call__(self, *args, **kwargs):
    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self._inputs[argpos]] = arg
    return tf.get_default_session().run(self._outputs, feeds)

class Visualizer():

    def __init__(self, calcGrad, calcCost, input):
        """
            Visualizer for Deep Neural Networks. Solves an inverse problem to find a suited input
            that minimizes the cost function given in calcCost.

                Parameters:
                -----------
              calcCost : function handle that computes the cost function for a given input
              calcGrad : function handle that computes the gradient of the cost function
              input : an input image (used for regularization or just to get the shape of the input)
        """
        self.calcGrad = calcGrad
        self.calcCost = calcCost
        self.input = np.asarray(input, dtype=np.float32)
        self.inp_shape = input.shape

    def optimize(self, x0, cost):
        return 0

    def map(self, x0):
        return self.optimize(x0, self.cost)


class DeepVisualizer(Visualizer):

    def __init__(self, calcGrad, calcCost, input, alpha=0.01):
        """
                Deep Visualization for Deep Neural Networks. Solves an inverse problem to find a suited input
                that minimizes the cost function given in calcCost.

                Parameters:
                -----------
              calcCost : function handle that computes the cost function for a given input
              calcGrad : function handle that computes the gradient of the cost function
              input : an input image (used for regularization or just to get the shape of the input)
              alpha : l2-regularization on the wanted input image to obtain feasible results
        """

        Visualizer.__init__(self, calcGrad, calcCost, input)

        self.alpha = alpha

    def costFun(self, x):
        """
                Function that computes the cost value for a given x

                Parameters:
                -----------
                  x : input data
        """
        tmp = x.reshape(self.inp_shape)
        c = np.float64(self.calcCost(np.asarray(tmp, dtype=np.float32))) + self.alpha * np.dot(x.T, x)
        return c

    def gradFun(self, x):
        """
                    Function that computes the gradient of the cost function at x

                    Parameters:
                    -----------
                    x : input data
        """
        tmp = x.reshape(self.inp_shape)
        g = np.ravel(
            np.asarray(self.calcGrad(np.asarray(tmp, dtype=np.float32)), dtype=np.float64)) + 2 * self.alpha * x
        return g

    def optimize(self, x0):
        """
                Solves the inverse problem

                Parameters:
                -----------
              x0 : initial solution
        """
        (result, f, d) = fmin_l_bfgs_b(lambda x: self.costFun(x), np.ravel(x0), lambda x: self.gradFun(x))
        print("optimization completed with cost: " + str(f))
        return result.reshape(self.inp_shape)


class SubsetSelection(Visualizer):

    def __init__(self, calcGrad, calcCost, input, alpha=0.01, gamma=0.1):
        """
            Subset selection for Deep Neural Networks. Solves an inverse problem to find a suited input
            that minimizes the cost function given in calcCost.

            Parameters:
            -----------
              calcCost : function handle that computes the cost function for a given input
              calcGrad : function handle that computes the gradient of the cost function
              input : an input image (used for regularization or just to get the shape of the input)
              alpha : l2-regularization on the wanted input image to obtain feasible results
              gamma : step size for the proximal gradient algorithm
        """
        Visualizer.__init__(self, calcGrad, calcCost, input)
        self.alpha = alpha
        self.gamma = gamma

    def costFun(self, S, x):
        """
              Function that computes the cost value for a given x

              Parameters:
              -----------
                x : input data
        """
        a=self.calcCost(S * x)
        return self.calcCost(S * x)

    def gradFun(self, S, x):
        """
                Function that computes the gradient of the cost function at x

              Parameters:
              -----------
                x : input data
        """
        a=self.calcGrad(S * x) * x
        a=np.squeeze(a,axis=0)
        return a

    def optimize(self, x0, n_iter=50):
        """
                    Solves the inverse problem

                    Parameters:
                    -----------
                  x0 : initial solution
                  n_iter : number of proximal gradient steps used for optimization
        """
        x0 = np.asarray(x0, dtype=np.float32)
        opt = proximal_alg.ProximalGradSolver(self.gamma, self.alpha, lambda x: self.costFun(x, self.input),
                                              lambda x: np.sum(np.abs(x)), lambda x: self.gradFun(x, self.input),
                                              proximal_alg.prox_l1_01)
        result = opt.minimize(x0, n_iter=n_iter)
        return result

