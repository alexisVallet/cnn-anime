# -*- coding: utf-8 -*-
""" Theano implementations of images affine transforms.
"""
import numpy as np
import theano
import theano.tensor as T

def resize(images, (out_rows, out_cols)):
    """ Theano symbolic image resizing with nearest neighbor interpolation.

    Arguments:
        images
            theano 4D tensor of shape (nb_images, nb_channels, nb_rows, nb_cols)
        out_rows
            symbolic int32 representing the number of output rows.
        out_cols
            symbolic int32 representing the number of output columns.
    Returns:
        theano 4D tensor of shape (nb_images, nb_channels, out_rows, out_cols)
    """
    nb_images, nb_channels, in_rows, in_cols = images.shape
    # Building a matrix of output image coordinates to transform.
    xs = T.arange(out_rows * out_cols) % out_cols
    ys = T.arange(out_rows * out_cols) // out_cols
    X = T.cast(T.stack(xs, ys), 'float32')
    # Building the inverse transformation matrix to apply (just a diagonal with 1/fx and 1/fy)
    fy = T.cast(in_rows, 'float32') / out_rows
    fx = T.cast(in_cols, 'float32') / out_cols
    M = T.set_subtensor(T.set_subtensor(T.zeros([2,2])[0,0], fx)[1,1], fy)
    # Compute the input coordinates by dot product.
    Xp = T.cast(T.dot(M, X), 'int32')
    # Then get our output images via indexing into the original ones.
    return images[:,:,Xp[1],Xp[0]].reshape([nb_images, nb_channels, out_rows, out_cols])

if __name__ == "__main__":
    img = cv2.imread('data/pixiv-115/images/初音ミク/1000062.jpg')
    s_img = T.tensor4('imgs')
    s_or = T.scalar('or', dtype='int32')
    s_oc = T.scalar('oc', dtype='int32')
    f_resize = theano.function([s_img, s_or, s_oc], resize(s_img, (s_or, s_oc)))
    t_img = np.rollaxis(img, 2, 0).reshape([1, img.shape[2], img.shape[0], img.shape[1]])
    r_img = f_resize(t_img, 512, 484)
    out_img = np.rollaxis(np.reshape(r_img, r_img.shape[1:]), 0, 3) / 255
    print out_img.shape
    cv2.imshow('src', img)
    cv2.imshow('dst', out_img)
    cv2.waitKey(0)
