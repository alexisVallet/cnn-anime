""" Picks random samples from the confusion data.
"""
import cPickle as pickle
import shutil
import sys
import json
import numpy as np
import os.path
import matplotlib.font_manager as fm
from pylab import *
import operator

def generate_plot(confidences, ground_truth, out_fname):
    prop = fm.FontProperties(fname='./ipag.ttf')
    sorted_confs = sorted(confidences.items(), key=operator.itemgetter(1), reverse=True)
    confs = map(lambda x: x[1], sorted_confs[0:5])[::-1]
    labels = map(lambda x: x[0], sorted_confs[0:5])[::-1]
    colors = map(lambda l: 'r' if l in ground_truth else 'b', labels)
    positions = np.linspace(0,1,5)
    figure(figsize=(2,1))
    frame = gca()
    frame.axes.get_xaxis().set_visible(False)
    barh(positions, confs, height=0.2, color=colors)
    yticks(positions + 0.1, labels)
    for tick in frame.axes.get_yticklabels():
        tick.set_font_properties(prop)
    savefig(out_fname, bbox_inches='tight')

if __name__ == "__main__":
    samples = pickle.load(open(sys.argv[1], 'rb'))
    out = sys.argv[2]
    perm = np.random.permutation(len(samples))
    keys = samples.keys()

    for i in range(50):
        fname = keys[perm[i]]
        conf = samples[fname]['confidence']
        gt = samples[fname]['ground_truth']
        
        shutil.copy('../data/pixiv-115/images/' + fname, out)
        generate_plot(conf, gt, out + '/' + os.path.basename(fname) + '.png')

