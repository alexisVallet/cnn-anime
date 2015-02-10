import cPickle as pickle
import cnn_classifier

if __name__ == "__main__":
    classifier = pickle.load(open('data/pixiv-1M/models/pixiv_1M48.pkl', 'rb'))
    pickle.dump(classifier, open('data/pixiv-1M/models/pixiv_1M_48.pkl', 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

