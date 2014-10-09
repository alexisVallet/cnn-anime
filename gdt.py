import ctypes
from numpy.ctypeslib import ndpointer, as_ctypes
import numpy as np
import cv2
import sys

lib = ctypes.cdll.LoadLibrary("./c/gdt.so")
cgdt1D = lib.gdt1D
cgdt1D.restype = None
cgdt1D.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ndpointer(ctypes.c_float, flags=("C_CONTIGUOUS",'W')),
    ndpointer(ctypes.c_int, flags=("C_CONTIGUOUS",'W')),
    ctypes.c_int,
    ctypes.c_double
]
cgdt2D = lib.gdt2D
cgdt2D.restype = None
cgdt2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ctypes.c_int,
                   ctypes.c_int,
                   ndpointer(ctypes.c_float, flags=("C_CONTIGUOUS",'W')),
                   ndpointer(ctypes.c_int, flags=("C_CONTIGUOUS",'W')),
                   ndpointer(ctypes.c_int, flags=("C_CONTIGUOUS",'W'))]

def gdt2D_py(d, f, rg=None):
    """ Python version for testing purposes.
    """
    rows, cols = f.shape
    if rg==None:
        rg = max(rows,cols)

    float_f = f.astype(np.float32)
    df = np.empty([rows, cols], dtype=np.float32)
    arg1 = np.empty([rows, cols], dtype=np.int32)
    arg2 = np.empty([rows, cols], dtype=np.int32)
    dx = np.array([d[0], d[2]], dtype=np.float32)
    dy = np.array([d[1], d[3]], dtype=np.float32)
    # Compute 1D gdt on each line separately.
    for i in range(rows):
        cgdt1D(dx, float_f[i], cols, df[i], arg1, 0, rg)
    # Then on each column.
    dft = np.require(df.T, dtype=np.float32, requirements=['C_CONTIGUOUS'])
    outdf = np.empty([cols, rows], dtype=np.float32)
    for j in range(cols):
        cgdt1D(dy, dft[j], rows, outdf[j], arg2, 0, rg)

    return (outdf.T, arg2)
    

def gdt2D(d, f, rg=None, scaling=1.):
    """ Computes the generalized distance transform of a function on a 2D
        grid, with quadratic distance measure.
    
    Arguments:
        d 4-elements vector containing quadratic coefficients (ax,ay,bx,by)
          defining the following distance measure 
             d(dx, dy) = ax*dx + bx*dx^2 + ay*dy + by*dy^2
        f numpy 2D array defining function values at each point of the
          grid.

    Returns:
        (df, args) where:
        - df(p) = min_q(f(q) + d(p - q))
        - args(p) = argmin_q(f(q) + d(p - q))
    """
    # Introduce deformation scaling into the distance function to avoid
    # having to touch the C code.
    cx, cy, cx2, cy2 = d
    d_ = np.array(
        [scaling * cx, scaling * cy, 
         scaling**2 * cx2, scaling**2 * cy2],
        dtype=np.float32
    )
    rows, cols = f.shape
    if rg==None:
        rg = max(rows,cols)
    df = np.empty(f.shape, dtype=np.float32)
    argi = np.empty([rows,cols,1], dtype=np.int32)
    argj = np.empty([rows,cols,1], dtype=np.int32)
    cgdt2D(d_.astype(np.float32), f.astype(np.float32), rows, cols,
           df, argi, argj, rg)

    return (df, np.concatenate((argi,argj), axis=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input an image")
    img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (dfimg, args) = gdt2D(np.array([0,0,1,1]), img)
    cv2.imshow("distance transform", dfimg / 255)
    cv2.waitKey(0)
