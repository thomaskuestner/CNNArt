#import theano
#import theano.tensor as T
#from scipy.optimize import fmin_l_bfgs_b
#import proximal_alg
import numpy as np
import h5py
import matplotlib.pyplot as plt

def on_click_axes(event):
    """Enlarge or restore the selected axis."""

    ax = event.inaxes
    layersName, layersWeights = getLayersWeights()
    if ax is None:
        # Occurs when a region not in an axis is clicked...
        return
    if event.button is 1:
        #event.canvas.matplotlibwidget_static_2.setVisible(True)
        f = plt.figure()
        if ax.name=='arrow':
            return

        w = layersWeights[ax.name].value
        if w.ndim == 4:
            w = np.transpose(w, (3, 2, 0, 1))
            mosaic_number = w.shape[0]
            nrows = int(np.round(np.sqrt(mosaic_number)))
            ncols = int(nrows)

            if nrows ** 2 < mosaic_number:
                ncols += 1

            f = plot_mosaic(w[:mosaic_number, 0], nrows, ncols, f)
            plt.suptitle("Weights of Layer '{}'".format(ax.name))
            f.show()
        else:
            pass
    else:
        # No need to re-draw the canvas if it's not a left or right click
        return
    event.canvas.draw()

def getLayersWeights():
    model = h5py.File('layer2ge.h5', 'r')
    layersName = []
    layersWeights = {}

    for i in model['layers']:
        layerIndex = 'layers' + '/' + i

        for n in model[layerIndex]:
            layerName = layerIndex + '/' + n
            layersName.append(n)

            weightsPath = layerName + '/' + 'weights'
            layersWeights[n] = model[weightsPath]
    #model.close()
    return layersName,layersWeights


def make_mosaic(im, nrows, ncols, border=1):

    import numpy.ma as ma

    nimgs = len(im)
    imshape = im[0].shape

    mosaic = ma.masked_all((nrows * imshape[0] ,
                            ncols * imshape[1] ),
                           dtype=np.float32)


    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * imshape[0]:row * imshape[0] + imshape[0],
        col * imshape[1]:col * imshape[1] + imshape[1]] = im[i]

    return mosaic


def on_click(event):
    """Enlarge or restore the selected axis."""
    ax = event.inaxes
    if ax is None:
        # Occurs when a region not in an axis is clicked...
        return
    if event.button is 1:
        # On left click, zoom the selected axes
        ax._orig_position = ax.get_position()
        ax.set_position([0.1, 0.1, 0.85, 0.85])
        for axis in event.canvas.figure.axes:
            # Hide all the other axes...
            if axis is not ax:
                axis.set_visible(False)
    elif event.button is 3:
        # On right click, restore the axes
        try:
            ax.set_position(ax._orig_position)
            for axis in event.canvas.figure.axes:
                axis.set_visible(True)
        except AttributeError:
            # If we haven't zoomed, ignore...
            pass
    else:
        # No need to re-draw the canvas if it's not a left or right click
        return
    event.canvas.draw()


def plot_mosaic(im, nrows, ncols, fig,**kwargs):

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    nimgs = len(im)
    imshape = im[0].shape

    import matplotlib.pyplot as plt
    import numpy as np

    # ax = fig.add_subplot()


    mosaic = np.zeros(imshape)


    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        ax = fig.add_subplot(nrows, ncols,i+1)
        ax.set_xlim(0,imshape[0]-1)
        ax.set_ylim(0,imshape[1]-1)

        mosaic = im[i]

        ax.imshow(mosaic, **kwargs)
        ax.set_axis_off()

    fig.canvas.mpl_connect('button_press_event', on_click)


    return fig


def get_weights_mosaic(model, layer_id, n):
    """
    """

    # Get Keras layer
    layer = model.layers[layer_id]

    # Check if this layer has weight values
    if not hasattr(layer, "weights"):
        raise Exception("The layer {} of type {} does not have weights.".format(layer.name,
                                                                                layer.__class__.__name__))

    weights = layer.weights[0].container.data
    weights = np.transpose(weights, (3, 2, 0, 1))

    # For now we only handle Conv layer like with 4 dimensions
    if weights.ndim != 4:
        raise Exception("The layer {} has {} dimensions which is not supported.".format(layer.name, weights.ndim))

    # n define the maximum number of weights to display
    if weights.shape[0] < n:
        n = weights.shape[0]



    #mosaic = make_mosaic(weights[:n, 0], nrows, ncols, border=1)
    im = weights[:n, 0]


    return im


def plot_weights(model, layer_id, n, ax=None):
    """Plot the weights of a specific layer. ndim must be 4.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()


    layer = model.layers[layer_id]

    im = get_weights_mosaic(model, layer_id, n)

    # Create the mosaic of weights
    nrows = int(np.round(np.sqrt(n)))
    ncols = int(nrows)

    if nrows ** 2 < n:
        ncols += 1

    plt.suptitle("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))

    fig=plot_mosaic(im, nrows, ncols,fig)
    fig.show()





    return fig


def plot_all_weights(model, n=64, **kwargs):

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    layers_to_show = []

    for i, layer in enumerate(model.layers[:]):
        if hasattr(layer, "weights"):
            if len(layer.weights)==0:
                continue
            weights = layer.weights[0].container.data
            if weights.ndim == 4:
                layers_to_show.append((i, layer))



    n_mosaic = len(layers_to_show)
    #n_mosaic = len(model.layers)
    nrows = int(np.round(np.sqrt(n_mosaic)))
    ncols = int(nrows)

    if nrows ** 2 < n_mosaic:
        ncols += 1

    for i, (layer_id, layer) in enumerate(layers_to_show):

        fig=plot_weights(model, layer_id, n, ax=None)

        fig.suptitle("Layer #{} called '{}' of type {}".format(layer_id, layer.name, layer.__class__.__name__))




    return fig



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
