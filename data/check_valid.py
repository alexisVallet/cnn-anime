import cv2
import sys

if __name__ == "__main__":
    # Checks that all images in a folder can be opened by opencv.
    if len(sys.argv) < 2:
        raise ValueError("Please input an image folder.")
    foldername = argv[2]
    imagefnames = [fname for fname in dir(foldername) if fname.endswith('.jpg')]
    for imagefname in imagefnames:
        image = cv2.imread(imagefname)
        if image == None:
            print imagefname + " couldn't be loaded properly by opencv!"
