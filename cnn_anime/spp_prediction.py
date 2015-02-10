""" Spatial pyramid pooling like prediction, using arbitrary confidence maps as inputs.
    Designed to work with a global average pooled network.
"""
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_pool

from affine import resize

def spp_predict(fmaps, pyramid):
    """ From input confidence maps, perform "SPP" prediction across a scale pyramid and using
        spatial pruning of labels and confidences.

    Arguments:
        fmaps
            theano symbolic 4D tensor with shape (nb_images, nb_labels, nb_rows, nb_cols)
        pyramid
            python list of average pooling kernel sizes, e.g. [3, 5].
    Returns:
        symbolic (nb_images, nb_labels) tensor of spatially pooled multi-scale predictions.
    """
    # Step 1: average pooling of the confidences across multiple scales, then average pooling
    # of that using spatial information to get multi-scale spatial confidences.
    pooled_maps = fmaps
    nb_images, nb_labels, nb_rows, nb_cols = fmaps.shape
    
    for ws in pyramid:
        pooled_maps += resize(
            dnn_pool(fmaps, (ws, ws), (1, 1), mode='average'),
            (nb_rows, nb_cols)
        )
    pooled_maps /= len(pyramid) + 1
    # Step 2: spatial max-pooling across labels.
    label_conf, label_map = T.max_and_argmax(pooled_maps, axis=1, keepdims=True)
    bcast_labels = T.addbroadcast(T.arange(nb_labels).reshape([1, nb_labels, 1, 1]), 0, 2, 3)
    label_mask = T.eq(bcast_labels, label_map)

    return T.mean(label_mask * label_conf, axis=[2,3])
