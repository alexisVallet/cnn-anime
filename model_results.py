""" Displays randomly selected images from a test set and the predicted labels according
    to a model.
"""
import cPickle as pickle
import sys
import numpy as np
import cv2
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_pool

from dataset import load_pixiv_1M, LazyIO, IdentityDataset
from spp_prediction import spp_predict

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Please input an output model file, image folder and test file.")
    model_file = sys.argv[1]
    image_folder = sys.argv[2]
    test_file = sys.argv[3]
    classifier = None
    print "Loading model..."
    with open(model_file, 'rb') as model:
        classifier = pickle.load(model)
    print "Loading test data..."
    test_data = load_pixiv_1M(
        image_folder,
        test_file,
        LazyIO
    )
    # shuffling
    test_data.shuffle(np.random.permutation(len(test_data)))
    # Predicting on the 50 first samples.
    samples = []
    labels = test_data.get_labels()[0:50]
    samples_iter = iter(test_data)

    for i in range(50):
        samples.append(samples_iter.next())
    test_subset = IdentityDataset(samples, labels)
    int_to_label = classifier.named_conv.int_to_label

    # Pooling fct
    fmaps = T.tensor4('fmaps')
    pool = lambda ws: theano.function(
        [fmaps],
        dnn_pool(fmaps, (ws, ws), (1,1), mode='average')
    )

    pyramid = [3,5,7,9,11,13,15]
    # SPP prediction fct
    pred = theano.function([fmaps], spp_predict(fmaps, pyramid))
    
    for i, sample in enumerate(test_subset):
        # Run prediction
        # Show confidence maps for top 5 guesses.
        maps = classifier.compute_activations(14, IdentityDataset([sample], labels[i]))
        # Multi-scale average pooling.
        # 1 by 1 average pooling is just the original feature maps.
        pooled_maps = np.array(maps, copy=True)
        mrows, mcols = pooled_maps.shape[2:]
        for wsize in pyramid:
            pmap = pool(wsize)(maps)
            # Resize to original scale via linear interpolation.
            for j in range(115):
                fmap = np.array(pmap[0,j],copy=True)
                resized = cv2.resize(fmap, (mcols, mrows))
                pooled_maps[0,j] += resized
        pooled_maps /= 8.
        label_map = np.argmax(pooled_maps[0], axis=0)
        label_conf = np.max(pooled_maps[0], axis=0)
        spatial_confs = np.zeros([115])
        for j in range(mrows):
            for k in range(mcols):
                spatial_confs[label_map[j,k]] += label_conf[j,k]
        spatial_confs /= mrows * mcols
        remaining_labels = np.unique(label_map)
        print "Ground truth:"
        for l in labels[i]:
            print l
        top_picks = filter(lambda l: l in remaining_labels, np.argsort(spatial_confs)[::-1])
        print "Spatially-corrected predictions:"
        for j in range(min(len(top_picks), 5)):
            print int_to_label[top_picks[j]] + ": " + repr(spatial_confs[top_picks[j]])
        # Pure theano version:
        th_confs = pred(maps)
        print "Theano version:"
        print th_confs.min()
        print th_confs.max()
        for il in np.argsort(th_confs[0])[::-1][0:5]:
            print int_to_label[il] + ": " + repr(th_confs[0,il])
        cv2.namedWindow('label confidence map', cv2.WINDOW_NORMAL)
        cv2.imshow('label confidence map', label_conf / label_conf.max())
        cv2.imshow('image', np.rollaxis(samples[i], 0, 3))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
