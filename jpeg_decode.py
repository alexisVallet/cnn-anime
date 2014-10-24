# -*- coding: utf-8 -*-
""" Fast jpeg decompression using libjpeg-turbo.
"""
import ctypes
from numpy.ctypeslib import ndpointer, as_ctypes
import numpy as np
import cv2

lib = ctypes.cdll.LoadLibrary("./c/jpeg_decode.so")
# C function to read a jpeg header
cread_header = lib.read_header
cread_header.restype = None
cread_header.argtypes = [
    ndpointer(np.uint8), # input raw image buffer
    ctypes.c_int,          # size of the buffer
    ctypes.POINTER(ctypes.c_int),  # output number of rows
    ctypes.POINTER(ctypes.c_int),  # output number of cols
]

# C function to decode a jpeg image
cdecode_jpeg = lib.decode_jpeg
cdecode_jpeg.restype = None
cdecode_jpeg.argtypes = [
    ndpointer(np.uint8),                      # Input buffer
    ctypes.c_int,
    ndpointer(np.uint8, flags="C_CONTIGUOUS") # Output image
]

def jpeg_decode(raw_buffer):
    # First read the image dimensions to initialize memory.
    rows, cols = [ctypes.c_int(), ctypes.c_int()]
    cread_header(raw_buffer, raw_buffer.size, ctypes.byref(rows),
                 ctypes.byref(cols))
    # Initialize the image properly.
    image = np.empty(
        [rows.value, cols.value, 3],
        np.uint8
    )

    cdecode_jpeg(raw_buffer, raw_buffer.size, image)

    # Return the image.
    return image

if __name__ == "__main__":
    image_fname = './data/pixiv-1M/images/初音ミク_0.jpg'

    with open(image_fname, 'rb') as image_file:
        raw_buffer = np.fromfile(image_file, np.uint8)
        image = jpeg_decode(raw_buffer)
        cv2.imshow('image', image)
        cv2.waitKey(0)
