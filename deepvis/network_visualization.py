import theano
import theano.tensor as T
from scipy.optimize import fmin_l_bfgs_b
import proximal_alg
import numpy as np

def make_mosaic(im, nrows, ncols, border=1):

    import numpy.ma as ma

    nimgs = len(im)
    imshape = im[0].shape

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    im
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = im[i]

    return mosaic

def plot_feature_map(model, layer_id, X, n=256, ax=None, **kwargs):
    """
    """
    import keras.backend as K
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    layer = model.layers[layer_id]


    try:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
        activations = get_activations([X, 0])[0]
    except:
        # Ugly catch, a cleaner logic is welcome here.
        raise Exception("This layer cannot be plotted.")

    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
                                                                                               activations.ndim))

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    fig = plt.figure(figsize=(15, 15))

    # Compute nrows and ncols for images
    n_mosaic = len(activations)
    nrows = int(np.round(np.sqrt(n_mosaic)))
    ncols = int(nrows)
    if (nrows ** 2) < n_mosaic:
        ncols += 1

    # Compute nrows and ncols for mosaics
    if activations[0].shape[0] < n:
        n = activations[0].shape[0]

    nrows_inside_mosaic = int(np.round(np.sqrt(n)))
    ncols_inside_mosaic = int(nrows_inside_mosaic)

    if nrows_inside_mosaic ** 2 < n:
        ncols_inside_mosaic += 1

    for i, feature_map in enumerate(activations):
        mosaic = make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

        ax = fig.add_subplot(nrows, ncols, i + 1)

        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
                                                                                        layer.name,
                                                                                        layer.__class__.__name__))

    fig.tight_layout()
    return fig

def plot_all_feature_maps(model, X, n=256, ax=None, **kwargs):
    figs = []

    for i, layer in enumerate(model.layers):

        try:
            fig = plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
        except:
            pass
        else:
            figs.append(fig)

    return figs


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
    
    def __init__(self, calcGrad, calcCost, input, alpha = 0.01):
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
	c = np.float64(self.calcCost(np.asarray(tmp,dtype=np.float32))) + self.alpha * np.dot(x.T, x)
	return c

    def gradFun(self, x):
	"""
	Function that computes the gradient of the cost function at x
	
	Parameters:
	-----------
	  x : input data
	"""
        tmp = x.reshape(self.inp_shape)
        g = np.ravel(np.asarray(self.calcGrad(np.asarray(tmp,dtype=np.float32)),dtype=np.float64)) + 2*self.alpha*x
        return g
    
    def optimize(self, x0):
        """
        Solves the inverse problem
        
        Parameters:
        -----------
	  x0 : initial solution
        """
        (result,f,d) = fmin_l_bfgs_b(lambda x:self.costFun(x), np.ravel(x0),lambda x: self.gradFun(x))
        print("optimization completed with cost: " + str(f))
        return result.reshape(self.inp_shape)
      
   
class SubsetSelection(Visualizer):
    
    def __init__(self, calcGrad, calcCost, input, alpha = 0.01, gamma = 0.1):
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
	return self.calcCost( S * x )

    def gradFun(self, S, x):
      	"""
	Function that computes the gradient of the cost function at x
	
	Parameters:
	-----------
	  x : input data
	"""
        return self.calcGrad( S * x ) * x #todo: sum over the dimensions of x which are not present in s!
    
    def optimize(self, x0, n_iter = 50):
        """
        Solves the inverse problem
        
        Parameters:
        -----------
	  x0 : initial solution
	  n_iter : number of proximal gradient steps used for optimization
        """
        x0 = np.asarray(x0, dtype=np.float32)
        opt = proximal_alg.ProximalGradSolver(self.gamma, self.alpha, lambda x: self.costFun(x,self.input), lambda x: np.sum(np.abs(x)), lambda x: self.gradFun(x, self.input), proximal_alg.prox_l1_01)
        result = opt.minimize(x0, n_iter = n_iter)
        return result

	
