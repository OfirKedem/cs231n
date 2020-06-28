from builtins import range
import numpy as np
from torch import nn

nn.Module



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    reshaped_x = np.reshape(x, [N, -1])  # [N, D]
    out = reshaped_x.dot(w) + b  # [N, M]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.reshape(dout @ w.T, x.shape)  # [N, d_1, ... , d_k]

    N = x.shape[0]
    reshaped_x = np.reshape(x, [N, -1])  # [N, D]
    dw = reshaped_x.T @ dout  # [D, M]

    db = np.ones(N) @ dout  # [M, ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * ((x > 0) * 1.0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # normalize
        mean = np.mean(x, axis=0)  # [D, ]
        var = np.var(x, axis=0)  # [D, ]
        norm_x = (x - mean) / np.sqrt(var + eps)  # [N, D]

        # compute output
        out = norm_x * gamma + beta

        # update running average
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        # cache
        cache = (x, mean, var, gamma, beta, eps, norm_x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # normalize
        norm_x = (x - running_mean) / np.sqrt(running_var + eps)  # [N, D]

        # compute output
        out = norm_x * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    x, mean, var, gamma, beta, eps, norm_x = cache

    dl_dnorm_x = dout * gamma  # [N, D]
    dl_dvar = np.sum(dl_dnorm_x * (x - mean) * ((var + eps) ** (-1.5)) * (-0.5), axis=0)  # [D, ]
    dl_dmean = np.sum(dl_dnorm_x * (-1.0 / np.sqrt(var + eps)), axis=0) \
               - 2 * dl_dvar * np.sum(x - mean, axis=0) / N  # [D, ]

    dx = dl_dnorm_x / np.sqrt(var + eps) + dl_dvar * 2 * (x - mean) / N + dl_dmean / N  # [N, D]
    dgamma = np.sum(dout * norm_x, axis=0)  # [D, ]
    dbeta = np.sum(dout, axis=0)  # [D, ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    x, mean, var, gamma, beta, eps, norm_x = cache

    dl_dnorm_x = dout * gamma  # [N, D]
    dl_dvar = np.sum(dl_dnorm_x * (mean - x), axis=0) * ((var + eps) ** (-1.5))  # [D, ]
    dl_dmean = np.sum(dl_dnorm_x * (-1.0 / np.sqrt(var + eps)), axis=0) \
               - 2 * dl_dvar * np.sum(x - mean, axis=0) / N  # [D, ]

    dx = dl_dnorm_x / np.sqrt(var + eps) + dl_dvar * (x - mean) / N + dl_dmean / N  # [N, D]
    dgamma = np.sum(dout * norm_x, axis=0)  # [D, ]
    dbeta = np.sum(dout, axis=0)  # [D, ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x = x.T  # [D, N]

    # normalize
    mean = np.mean(x, axis=0)  # [N, ]
    var = np.var(x, axis=0)  # [N, ]
    norm_x = (x - mean) / np.sqrt(var + eps)  # [D, N]

    norm_x = norm_x.T  # [N, D]

    # compute output
    out = norm_x * gamma + beta  # [N, D]

    # cache
    cache = (x.T, mean, var, gamma, beta, eps, norm_x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = dout.shape
    x, mean, var, gamma, beta, eps, norm_x = cache

    dl_dnorm_x = dout * gamma  # [N, D]

    # traspose
    dl_dnorm_x = dl_dnorm_x.T  # [D, N]
    x = x.T  # [D, N]

    dl_dvar = np.sum(dl_dnorm_x * (mean - x), axis=0) * ((var + eps) ** (-1.5))  # [N, ]
    dl_dmean = np.sum(dl_dnorm_x * (-1.0 / np.sqrt(var + eps)), axis=0) \
               - 2 * dl_dvar * np.sum(x - mean, axis=0) / D  # [N, ]

    dx = (dl_dnorm_x / np.sqrt(var + eps) + dl_dvar * (x - mean) / D + dl_dmean / D).T  #  [N, D]
    dgamma = np.sum(dout * norm_x, axis=0)  # [D, ]
    dbeta = np.sum(dout, axis=0)  # [D, ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get constance
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # compute output dimensions
    H_tag = int(1 + (H + 2 * pad - HH) / stride)
    W_tag = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_tag, W_tag))  # [N, F, H', W']

    # pad x
    pad_x = np.pad(x, ((0, 0),  # N
                       (0, 0),  # C
                       (pad, pad),  # H
                       (pad, pad)))  # W

    # do convolution
    for hh in range(H_tag):
        for ww in range(W_tag):
            crop = pad_x[:, :, (hh * stride):(hh * stride + HH), (ww * stride):(ww * stride + WW)]  # [N, C, HH, WW]
            crop = np.reshape(crop, [N, -1])  # [N, D]; D == C * HH * WW
            for f in range(F):
                curr_filter = w[f, :, :, :]  # [C, HH, WW]
                curr_filter = np.ravel(curr_filter)  # [D, ]
                out[:, f, hh, ww] = np.dot(crop, curr_filter) + b[f]  # [N, ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get constance
    x, w, b, conv_param = cache
    stride, pad = conv_param["stride"], conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_tag, W_tag = dout.shape[2:]

    # initialize gradients
    dw = np.zeros_like(w)  # [F, C, HH, WW]
    db = np.zeros_like(b)  # [F, ]

    # pad x ,dx
    pad_x = np.pad(x, ((0, 0),  # N
                       (0, 0),  # C
                       (pad, pad),  # H
                       (pad, pad)))  # W
    pad_dx = np.zeros_like(pad_x)

    # do convolution
    for hh in range(H_tag):
        for ww in range(W_tag):
            crop = pad_x[:, :, (hh * stride):(hh * stride + HH), (ww * stride):(ww * stride + WW)]  # [N, C, HH, WW]
            crop = np.reshape(crop, [N, -1])  # [N, D]; D == C * HH * WW
            for f in range(F):
                curr_filter = w[f, :, :, :]  # [C, HH, WW]
                curr_filter = np.ravel(curr_filter)  # [D, ]
                # out[:, f, hh, ww] = np.dot(crop, curr_filter) + b[f]  # [N, ]

                crop_dw = np.dot(dout[:, f, hh, ww], crop)  # [D, ]
                dw[f, :, :, :] += np.reshape(crop_dw, [C, HH, WW])  # [C, HH, WW]

                db[f] += dout[:, f, hh, ww].sum()

                crop_dx = np.outer(dout[:, f, hh, ww], curr_filter)  # [N, D]
                pad_dx[:, :, (hh * stride):(hh * stride + HH), (ww * stride):(ww * stride + WW)] \
                    += np.reshape(crop_dx, [N, C, HH, WW])

    # un-pad dx
    dx = pad_dx[:, :, pad:-pad, pad:-pad]  # [N, C, H, W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get constance
    stride, pool_height, pool_width = pool_param["stride"], pool_param["pool_height"], pool_param["pool_width"]
    N, C, H, W = x.shape

    # compute output dimensions
    H_tag = int(1 + (H - pool_height) / stride)
    W_tag = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, H_tag, W_tag))  # [N, C, H', W']

    # do max pool
    for hh in range(H_tag):
        for ww in range(W_tag):
            crop = x[:, :, (hh * stride):(hh * stride + pool_height),
                   (ww * stride):(ww * stride + pool_width)]  # [N, C, pool_height, pool_width]
            crop = np.reshape(crop, [N, C, -1])  # [N, C, D]; D == pool_height * pool_width
            # for c in range(C):
            #     out[:, c, hh, ww] = np.amax(crop[:, c, :], axis=1)
            out[:, :, hh, ww] = np.amax(crop, axis=2)  # [N, C]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get constants
    x, pool_param = cache
    stride, pool_height, pool_width = pool_param["stride"], pool_param["pool_height"], pool_param["pool_width"]
    N, C, H, W = x.shape
    H_tag, W_tag = dout.shape[2:]

    dx = np.zeros_like(x)

    # do max pool
    for hh in range(H_tag):
        for ww in range(W_tag):
            crop = x[:, :, (hh * stride):(hh * stride + pool_height),
                   (ww * stride):(ww * stride + pool_width)]  # [N, C, pool_height, pool_width]
            crop = np.reshape(crop, [N, C, -1])  # [N, C, D]; D == pool_height * pool_width

            max_idx = np.argmax(crop, axis=2)  # [N, C]
            mask = np.zeros([N, C, pool_height * pool_width])
            for n in range(N):
                for c in range(C):
                    mask[n, c, max_idx[n, c]] = 1
            mask = np.reshape(mask, [N, C, pool_height, pool_width])  # [N, C, pool_height, pool_width]

            dx[:, :, (hh * stride):(hh * stride + pool_height), (ww * stride):(ww * stride + pool_width)] \
                += (dout[:, :, hh, ww] * mask.transpose([2, 3, 0, 1])).transpose([2, 3, 0, 1])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    # reshape x
    new_N = N * H * W  # size of dim over which the statistics are computed
    transpose_x = x.transpose([0, 2, 3, 1])  # [N, H, W, C]
    new_dim_x = np.reshape(transpose_x, [new_N, C])

    # do batch norm
    out, cache = batchnorm_forward(new_dim_x, gamma, beta, bn_param)  # out [new_N, C]

    # reshape out back to x original shape
    out = np.reshape(out, [N, H, W, C])  # [N, H, W, C]
    out = np.transpose(out, [0, 3, 1, 2])  # [N, C, H, W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape

    # reshape x
    new_N = N * H * W  # size of dim over which the statistics are computed
    transpose_dout = dout.transpose([0, 2, 3, 1])  # [N, H, W, C]
    new_dim_dout = np.reshape(transpose_dout, [new_N, C])

    # do batch norm
    dx, dgamma, dbeta = batchnorm_backward(new_dim_dout, cache)  # out [new_N, C]

    # reshape out back to x original shape
    dx = np.reshape(dx, [N, H, W, C])  # [N, H, W, C]
    dx = np.transpose(dx, [0, 3, 1, 2])  # [N, C, H, W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape

    # split to groups
    x = np.reshape(x, [N, G, -1])  # [N, G, D] ; D == C // G * H * W
    x = np.transpose(x, [2, 0, 1])  # [D, N, G]

    # normalize per data point per group
    mean = np.mean(x, axis=0)  # [N, G]
    var = np.var(x, axis=0)  # [N, G]
    norm_x = (x - mean) / np.sqrt(var + eps)  # [D, N, G]

    # reshape back to original shape
    norm_x = np.transpose(norm_x, [1, 2, 0])  # [N, G, D]
    norm_x = np.reshape(norm_x, [N, C, H, W])  # [N, C, H, W]
    x = np.transpose(x, [1, 2, 0])  # [N, G, D]
    x = np.reshape(x, [N, C, H, W])  # [N, C, H, W]

    # compute output
    gamma = gamma[np.newaxis, :, np.newaxis, np.newaxis]  # [1, C, 1, 1]
    beta = beta[np.newaxis, :, np.newaxis, np.newaxis]  # [1, C, 1, 1]
    out = norm_x * gamma + beta  # [N, C, H, W]

    # cache
    cache = (x, mean, var, gamma, beta, eps, norm_x, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    x, mean, var, gamma, beta, eps, norm_x, G = cache

    dl_dnorm_x = dout * gamma  # [N, C, H, W]

    # split to groups
    D = (C // G) * H * W
    x = np.reshape(x, [N, G, -1])  # [N, G, D]
    x = np.transpose(x, [2, 0, 1])  # [D, N, G]
    dl_dnorm_x = np.reshape(dl_dnorm_x, [N, G, -1])  # [N, G, D]
    dl_dnorm_x = np.transpose(dl_dnorm_x, [2, 0, 1])  # [D, N, G]

    dl_dvar = np.sum(dl_dnorm_x * (mean - x), axis=0) * ((var + eps) ** (-1.5))  # [N, G]
    dl_dmean = np.sum(dl_dnorm_x * (-1.0 / np.sqrt(var + eps)), axis=0) \
               - 2 * dl_dvar * np.sum(x - mean, axis=0) / D  # [N, G]

    dx = dl_dnorm_x / np.sqrt(var + eps) + dl_dvar * (x - mean) / D + dl_dmean / D  # [D, N, G]
    dgamma = np.sum(dout * norm_x, axis=(0, 2, 3))  # [C, ]
    dbeta = np.sum(dout, axis=(0, 2, 3))  # [C, ]

    # reshape back to original shapes
    dx = np.transpose(dx, [1, 2, 0])  # [N, G, D]
    dx = np.reshape(dx, [N, C, H, W])  # [N, C, H, W]
    dgamma = dgamma[np.newaxis, :, np.newaxis, np.newaxis]  # [1, C, 1, 1]mma)
    dbeta = dbeta[np.newaxis, :, np.newaxis, np.newaxis]  # [1, C, 1, 1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
