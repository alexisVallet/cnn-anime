import cPickle as pickle
import numpy as np
import cv2
import sys
from random import shuffle

if __name__ == "__main__":
    test_set = pickle.load(open(sys.argv[1], 'rb'))
    nb_check = int(sys.argv[2])
    nb_incorrect = 0
    keys = test_set.keys()
    shuffle(keys)
    rand_fnames = keys[0:nb_check]

    for i, fname in enumerate(rand_fnames):
        print "Ground truth:"
        for l in test_set[fname]:
            print l
        img = cv2.imread('images/' + fname)
        cv2.imshow('img', img)
        print "Is the ground truth correct?"
        answer = ord('a')
        while answer not in [ord('y'), ord('n')]:
            answer = cv2.waitKey(0) % 256
            print answer
        if answer == ord('n'):
            nb_incorrect += 1
        print repr(nb_incorrect) + " incorrect so far out of " + repr(i + 1)
    print repr(nb_incorrect) + " out of " + repr(nb_check) + " total."

