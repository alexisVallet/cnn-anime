import cv2
import numpy as np
import os.path
import os
import sys

if __name__ == "__main__":
    for folder in sys.argv[1:]:
        img_fnames = [f for f in os.listdir(folder) if f.endswith('.jpg') and not f.endswith('figure.jpg')]

        for img_fname in img_fnames:
            img = cv2.imread(folder + '/' + img_fname)
            plot = cv2.imread(folder + '/' + img_fname + ".png")
            p_r, p_c = plot.shape[0:2]
            i_r, i_c = img.shape[0:2]
            # Resize image if it is too big:
            if max(i_r, i_c) > 512:
                ratio = 512. / max(i_r, i_c)
                img = cv2.resize(img, None, fx=ratio, fy=ratio)
            i_r, i_c = img.shape[0:2]
            out_r, out_c = (p_r + i_r, max(p_c, i_c))
            out = 255 * np.ones([out_r, out_c, 3], dtype=np.uint8)
            i_offset = (out_c - i_c) // 2
            p_offset = (out_c - p_c) // 2
            out[0:i_r, i_offset:i_offset+i_c] = img
            out[i_r:i_r+p_r, p_offset:p_offset+p_c] = plot
            cv2.imwrite(folder + '/' + img_fname + "-figure.jpg", out)
