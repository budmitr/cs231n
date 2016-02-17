import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  Arbitrary architecture convnet
  """

  def __init__(self, input_dim=(3, 32, 32), layers=[], weight_scale=1e-3, reg=0.0, 
               dropout=0.5, num_classes=10, seed=None, dtype=np.float32):
    """
    Initialize the network by given params

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - layers: List of tuples [('layername1', params1), ..., ('layernameN', paramsN)]
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dropout: Scalar between 0 and 1 giving dropout strength
    - num_classes: Number of scores to produce from the final affine layer.
    - seed: numpy random seed for dropout layer.
    - dtype: numpy datatype to use for computation.
    """

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.layers = layers
    self.num_layers = len(self.layers) + 1 # additional affine before softmax

    self.dropout_param = {'mode': 'train', 'p': dropout}
    if seed is not None:
      self.dropout_param['seed'] = seed

    # this is unoptimized and creates some extra list elements
    # TODO: optimize!
    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers)]

    dimension = input_dim
    for l, layer in enumerate(self.layers):
      strl = str(l)
      layertype = layer[0]
      layerparams = layer[1] if len(layer) > 1 else None

      # initialize params for every layer by its type
      # skip relu -- nothing to init
      if layertype == 'conv':
        depth, size = layerparams
        shape = (depth, dimension[-3], size, size)
        self.params['W' + strl] = weight_scale * np.random.randn(*shape).astype(dtype)
        self.params['b' + strl] = np.zeros(depth).astype(dtype)
        dimension = (depth, dimension[-2], dimension[-1])

      if layertype == 'affine':
        shape = (np.prod(dimension), layerparams)
        self.params['W' + strl] = weight_scale * np.random.randn(*shape).astype(dtype)
        self.params['b' + strl] = np.zeros(layerparams).astype(dtype)
        dimension = layerparams

      if layertype == 'batchnorm_s':
        self.params['gamma' + strl] = np.ones(dimension[-3])
        self.params['beta' + strl] = np.zeros(dimension[-3])

      if layertype == 'batchnorm':
        self.params['gamma' + strl] = np.ones(dimension)
        self.params['beta' + strl] = np.zeros(dimension)

      if layertype == 'pool':
        dimension = (dimension[0], dimension[1] / 2, dimension[2] / 2)

    # extra affine before softmax
    strl = str(self.num_layers - 1)
    self.params['W' + strl] = weight_scale * np.random.randn(np.prod(dimension), num_classes).astype(dtype)
    self.params['b' + strl] = np.zeros(num_classes).astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient
    """

    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    self.dropout_param['mode'] = mode
    for bn_param in self.bn_params:
      bn_param[mode] = mode

    h, caches = X, []

    # FORWARD PASS
    for l, layer in enumerate(self.layers):
      strl = str(l)
      layertype = layer[0]
      layerparams = layer[1] if len(layer) > 1 else None

      if layertype == 'conv':
        W, b = self.params['W' + strl], self.params['b' + strl]
        filter_size = W.shape[-1]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        h, c = conv_forward_fast(h, W, b, conv_param)
        caches.append(c)

      if layertype == 'affine':
        h, c = affine_forward(h, self.params['W' + strl], self.params['b' + strl])
        caches.append(c)

      if layertype == 'relu':
        h, c = relu_forward(h)
        caches.append(c)

      if layertype == 'pool':
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        h, c = max_pool_forward_fast(h, pool_param)
        caches.append(c)

      if layertype == 'batchnorm':
        gamma, beta = self.params['gamma' + strl], self.params['beta' + strl]
        h, c = batchnorm_forward(h, gamma, beta, self.bn_params[l])
        caches.append(c)

      if layertype == 'batchnorm_s':
        gamma, beta = self.params['gamma' + strl], self.params['beta' + strl]
        h, c = spatial_batchnorm_forward(h, gamma, beta, self.bn_params[l])
        caches.append(c)

      if layertype == 'dropout':
        h, c = dropout_forward(h, self.dropout_param)
        caches.append(c)


    # extra affine
    strl = str(self.num_layers - 1)
    h, c = affine_forward(h, self.params['W' + strl], self.params['b' + strl])
    caches.append(c)

    # LOSS COMPUTATION

    scores = h
    if y is None:
      return scores

    loss, grads = 0, {}
    loss, g = softmax_loss(scores, y)
    for l in xrange(self.num_layers):
      windex = 'W' + str(l)
      if windex in self.params:
        loss += 0.5 * self.reg * np.sum(self.params[windex] ** 2)

    # BACKWARD PASS

    # extra affine
    strl = str(self.num_layers - 1)
    c = caches.pop()
    g, grads['W' + strl], grads['b' + strl] = affine_backward(g, c)
    grads['W' + strl] += self.reg * self.params['W' + strl]

    for l, layer in enumerate(reversed(self.layers)):
      strl = str(len(self.layers) - l - 1)
      layertype = layer[0]

      if layertype == 'conv':
        c = caches.pop()
        g, grads['W' + strl], grads['b' + strl] = conv_backward_fast(g, c)
        grads['W' + strl] += self.reg * self.params['W' + strl]

      if layertype == 'affine':
        c = caches.pop()
        g, grads['W' + strl], grads['b' + strl] = affine_backward(g, c)
        grads['W' + strl] += self.reg * self.params['W' + strl]

      if layertype == 'relu':
        c = caches.pop()
        g = relu_backward(g, c)

      if layertype == 'pool':
        c = caches.pop()
        g = max_pool_backward_fast(g, c)

      if layertype == 'batchnorm':
        c = caches.pop()
        g, grads['gamma' + strl], grads['beta' + strl] = batchnorm_backward(g, c)

      if layertype == 'batchnorm_s':
        c = caches.pop()
        g, grads['gamma' + strl], grads['beta' + strl] = spatial_batchnorm_backward(g, c)

      if layertype == 'dropout':
        c = caches.pop()
        g = dropout_backward(g, c)

    return loss, grads