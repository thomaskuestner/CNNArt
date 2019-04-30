import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from configGUI import proximal_alg
from configGUI.proximal_alg import ProximalGradSolver


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
        """Function that computes the cost value for a given x

        Parameters:
        -----------
          x : input data"""

        return self.calcCost(S * x)

    def gradFun(self, S, x):
        """Function that computes the gradient of the cost function at x

            Parameters:
            -----------
              x : input data"""

        return self.calcGrad(S * x) * x  # todo: sum over the dimensions of x which are not present in s!

    def optimize(self, x0, n_iter=50):
        """
        Solves the inverse problem
        
        Parameters:
        -----------
          x0 : initial solution
          n_iter : number of proximal gradient steps used for optimization
        """
        x0 = np.asarray(x0, dtype=np.float32)
        opt = ProximalGradSolver(self.gamma, self.alpha, lambda x: self.costFun(x, self.input),
                                 lambda x: np.sum(np.abs(x)), lambda x: self.gradFun(x, self.input),
                                 proximal_alg.prox_l1_01)
        result = opt.minimize(x0, n_iter=n_iter)

        return result

## modified from ann_viz
## for drawing network architecture
def cnn2d_visual(model, view=True, filename="network.gv", title="My Neural Network"):
    """Vizualizez a Sequential model.
    # Arguments
        model: A Keras model instance.
        view: whether to display the model after generation.
        filename: where to save the vizualization. (a .gv file)
        title: A title for the graph
    """
    from graphviz import Digraph
    import keras
    input_layer = 0
    hidden_layers_nr = -1
    layer_types = []
    hidden_layers = []
    output_layer = 0
    for layer in model.layers:
        if(layer == model.layers[0]):
            input_layer = int(str(layer.input_shape).split(",")[1][1:-1])
            if (type(layer) == keras.layers.core.Dense):
                last_layer_nodes = input_layer
                nodes_up = input_layer
                hidden_layers.append(int(str(layer.output_shape).split(",")[1][1:-1]))
                layer_types.append("Dense")
            else:
                last_layer_nodes = 1
                nodes_up = 1
                input_layer = 1
                if (type(layer) == keras.layers.convolutional.Conv2D):
                    layer_types.append("Conv2D")
                elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                    layer_types.append("MaxPooling2D")
                elif (type(layer) == keras.layers.core.Dropout):
                    layer_types.append("Dropout")
                elif (type(layer) == keras.layers.core.Flatten):
                    layer_types.append("Flatten")
                elif (type(layer) == keras.layers.core.Activation):
                    layer_types.append("Activation")
                else:
                    layer_types.append("InputLayer")

        else:
            hidden_layers_nr += 1
            if (type(layer) == keras.layers.core.Dense):
                hidden_layers.append(int(str(layer.output_shape).split(",")[1][1:-1]))
                layer_types.append("Dense")
            else:
                hidden_layers.append(1)
                if (type(layer) == keras.layers.convolutional.Conv2D):
                    layer_types.append("Conv2D")
                elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                    layer_types.append("MaxPooling2D")
                elif (type(layer) == keras.layers.core.Dropout):
                    layer_types.append("Dropout")
                elif (type(layer) == keras.layers.core.Flatten):
                    layer_types.append("Flatten")
                elif (type(layer) == keras.layers.core.Activation):
                    layer_types.append("Activation")


    output_layer = int(str(model.layers[-1].output_shape).split(",")[1][1:-1])
    g = Digraph('g', format='png', filename=filename)
    n = 0
    g.graph_attr.update(splines="false", nodesep='1', ranksep='2')
    #Input Layer
    with g.subgraph(name='cluster_input') as c:
        if(type(model.layers[1]) == keras.layers.core.Dense):
            the_label = title+'\n\n\n\nInput Layer'
            if (int(str(model.layers[0].input_shape).split(",")[1][1:-1]) > 10):
                the_label += " (+"+str(int(str(model.layers[0].input_shape).split(",")[1][1:-1]) - 10)+")"
                input_layer = 10
            c.attr(color='white')
            for i in range(0, input_layer):
                n += 1
                c.node(str(n))
                c.attr(label=the_label)
                c.attr(rank='same')
                c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

        elif(type(model.layers[1]) == keras.layers.convolutional.Conv2D):
            #Conv2D Input visualizing
            the_label = title+'\n\n\n\nInput Layer'
            c.attr(color="white", label=the_label)
            c.node_attr.update(shape="square")
            pxls = str(model.layers[0].input_shape).split(',')
            clr = int(pxls[3][1:-1])
            if (clr == 1):
                clrmap = "Grayscale"
                the_color = "black:white"
            elif (clr == 3):
                clrmap = "RGB"
                the_color = "#e74c3c:#3498db"
            else:
                clrmap = ""
            c.node_attr.update(fontcolor="white", fillcolor=the_color, style="filled")
            n += 1
            c.node(str(n), label="Image\n"+pxls[1]+" x"+pxls[2]+" pixels\n"+clrmap, fontcolor="white")
        else:
            raise ValueError("ANN Visualizer: Layer not supported for visualizing")
    for i in range(1, hidden_layers_nr+1):
        with g.subgraph(name="cluster_"+str(i+1)) as c:
            if (layer_types[i] == "Dense"):
                c.attr(color='white')
                c.attr(rank='same')
                #If hidden_layers[i] > 10, dont include all
                the_label = ""
                if (int(str(model.layers[i].output_shape).split(",")[1][1:-1]) > 10):
                    the_label += " (+"+str(int(str(model.layers[i].output_shape).split(",")[1][1:-1]) - 10)+")"
                    hidden_layers[i] = 10
                c.attr(labeljust="right", labelloc="b", label=the_label)
                for j in range(0, hidden_layers[i]):
                    n += 1
                    c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                last_layer_nodes = hidden_layers[i]
                nodes_up += hidden_layers[i]
            elif (layer_types[i] == "Conv2D"):
                c.attr(style='filled', color='#5faad0')
                n += 1
                kernel_size = str(model.layers[i].get_config()['kernel_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['kernel_size']).split(',')[1][1 : -1]
                filters = str(model.layers[i].get_config()['filters'])
                c.node("conv_"+str(n), label="Convolutional Layer\nKernel Size: "+kernel_size+"\nFilters: "+filters, shape="square")
                c.node(str(n), label=filters+"\nFeature Maps", shape="square")
                g.edge("conv_"+str(n), str(n))
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), "conv_"+str(n))
                last_layer_nodes = 1
                nodes_up += 1
            elif (layer_types[i] == "MaxPooling2D"):
                c.attr(color="white")
                n += 1
                pool_size = str(model.layers[i].get_config()['pool_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['pool_size']).split(',')[1][1 : -1]
                c.node(str(n), label="Max Pooling\nPool Size: "+pool_size, style="filled", fillcolor="#8e44ad", fontcolor="white")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), str(n))
                last_layer_nodes = 1
                nodes_up += 1
            elif (layer_types[i] == "Flatten"):
                n += 1
                c.attr(color="white")
                c.node(str(n), label="Flattening", shape="invtriangle", style="filled", fillcolor="#2c3e50", fontcolor="white")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), str(n))
                last_layer_nodes = 1
                nodes_up += 1
            elif (layer_types[i] == "Dropout"):
                n += 1
                c.attr(color="white")
                c.node(str(n), label="Dropout Layer", style="filled", fontcolor="white", fillcolor="#f39c12")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), str(n))
                last_layer_nodes = 1
                nodes_up += 1
            elif (layer_types[i] == "Activation"):
                n += 1
                c.attr(color="white")
                fnc = model.layers[i].get_config()['activation']
                c.node(str(n), shape="octagon", label="Activation Layer\nFunction: "+fnc, style="filled", fontcolor="white", fillcolor="#00b894")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), str(n))
                last_layer_nodes = 1
                nodes_up += 1


    with g.subgraph(name='cluster_output') as c:
        if (type(model.layers[-1]) == keras.layers.core.Dense):
            c.attr(color='white')
            c.attr(rank='same')
            c.attr(labeljust="1")
            for i in range(1, output_layer+1):
                n += 1
                c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.edge(str(h), str(n))
            c.attr(label='Output Layer', labelloc="bottom")
            c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

    g.attr(arrowShape="none")
    g.edge_attr.update(arrowhead="none", color="#707070")
    if view == True:
         g.render(filename=filename, view=False)
