import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import copy
import cPickle as pickle

from classifier import ClassifierMixin
from dataset import IdentityDataset, mini_batch_split
from spp_prediction import spp_predict
from threshold import learn_threshold

def loadcnn(filepath):
    """ Dumb wrapper around pickle to go around restrictions.
    """
    cnn = None
    with open(filepath, 'rb') as cnnfile:
        cnn = pickle.load(cnnfile)
    return cnn

def dumpcnn(cnn, filepath):
    with open(filepath, 'wb') as cnnfile:
        pickle.dump(cnn, cnnfile, protocol=pickle.HIGHEST_PROTOCOL)

class BaseCNNClassifier:
    """ Image classifier based on a convolutional neural network.
    """
    def __init__(self, architecture, optimizer, input_shape,
                 srng, init='random', cost='mlr', l2_reg=0,
                 preprocessing=[], named_conv=None, lin_thresh=None, verbose=False):
        """ Initializes a convolutional neural network with a specific
            architecture, optimization method for training and 
            initialization procedure.

        Arguments:
            architecture
                list of tuples which should be either:
                - ('conv', {
                    nb_filters: ,
                    rows: ,
                    cols: ,
                    stride_r: ,  defaults to 1
                    stride_c: ,  defaults to 1
                    padding: , defaults to (0,0)
                    init_bias: , defaults to 0
                    non_lin: ', one of 'relu' or None
                  })
                - ('pool', {
                    type: 'max' or 'average',
                    rows: ,
                    cols: ,
                    stride_r: ,
                    stride_c:
                  })
                - ('fc', {
                    nb_units: ,
                    init_std: , defaults to 0.01
                    init_bias:  defaults to 0
                  })
                - ('linear', {
                    nb_outputs: ,
                    init_bias:
                  })
                - ('dropout', dropout_proba)
                - ('spp', pyramid)
                - Pre-trained layers, as instances of the Layer class.
            optimizer
                optimization method to use for training the network. Should be
                an instance of Optimizer.
            input_shape
                shape of input images, in [nb_channels, nb_rows, nb_cols] format.
            srng
                shared random number generator to use for dropout and weights initialization.
            init
                initialization procedure for the network. Can be:
                - 'random' for random initialization of network weights.
                - 'orth' for random initialization of network weights, with orthogonal weights
                  for filters on the same convolutional layer.
            cost
                cost function to use for training. One of:
                - 'mlr' for multinomial logistic regression cost function. Prediction output
                  will be softmaxed probabilities in the single label case, and "generalized"
                  softmaxed likelihoods in the multi label case.
                - 'bpmll' for the back propagation multi label learning (BP-MLL) cost function
                  by Zhang and Zhou, 2006.  Prediction output will be confidence scores in
                  ]-inf; +inf[.
                - 'multi-mlr' for a kind of compromise between the 2.
            l2_reg
                parameter controlling the strength of l2 regularization.
            orth_reg
                parameter controlling the strength of non-orthogonality penalty
                for convolutional layers.
            preprocessing
                pipeline of dataset transformers for training and testing preprocessing.
            verbose
                verbosity.
        """
        self.architecture = architecture
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.srng = srng
        self.init = init
        self.cost = cost
        self.l2_reg = l2_reg
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.named_conv = named_conv
        self.lin_thresh = lin_thresh
        self.init_model()

    def init_model(self):
        if self.verbose:
            print "Initializing weights..."
        # If loaded from a file, use the already initialized weights.
        if self.init in ['random', 'orth']:
            self.model = self.init_random()
            self.architecture = self.model.layers
        else:
            raise ValueError(repr(self.init) + 
                             " is not a valid initialization method.")
        self.compile_prediction()
        
    def __getstate__(self):        
        return (self.architecture, self.optimizer, self.input_shape, self.srng, self.init,
                self.cost, self.l2_reg, self.preprocessing, self.verbose, self.model,
                self.named_conv, self.lin_thresh)

    def __setstate__(self, state):
        (self.architecture, self.optimizer, self.input_shape, self.srng, self.init,
         self.cost, self.l2_reg, self.preprocessing, self.verbose, self.model,
         self.named_conv, self.lin_thresh) = state
        self.init_model()
    
    def compile_prediction(self):
        # Compile the CNN prediction functions.
        if self.verbose:
            print "Compiling prediction functions..."
        test_samples = T.tensor4('test_samples')
        self._predict_proba = theano.function(
            [test_samples],
            self.model.forward_pass(test_samples)
        )
        
    def train(self, dataset, valid_data=None):
        # Use theano's FFT convolutions.
        # Initialize the CNN with the desired method.
        # Run the preprocessing pipeline.
        if self.verbose:
            print "Preprocessing..."
        pp_dataset = dataset
        pp_valid_data = valid_data

        for preproc in self.preprocessing:
            if self.verbose:
                print "Preprocessing step " + repr(preproc) + "..."
            pp_dataset = preproc.train_data_transform(pp_dataset)
            if valid_data != None:
                pp_valid_data = preproc.test_data_transform(pp_valid_data)

        # Run the optimizer on the CNN cost function.
        if self.verbose:
            print "Optimizing the cost function..."
        # Apply the FFT convolution optimization. Although it is applied to the whole
        # graph, FFT convolutions are the only ones where we actually use nnet.conv2D,
        # so only those get compiled away. For other convolutions, we use the specific
        # ops.
        self.optimizer.optimize(
            self,
            pp_dataset,
            pp_valid_data
        )

    def predict_probas(self, images):
        # Run the preprocessing pipeline.
        pp_images = images

        for preproc in self.preprocessing:
            pp_images = preproc.test_data_transform(pp_images)

        # Run prediction.
        probas = self._predict_proba(pp_images.to_array())
        if len(probas.shape) == 4:
            probas = probas.reshape([probas.shape[0], probas.shape[1]])

        # Then convert the probabilities (for instance averageing when
        # samples were duplicated, etc).
        for preproc in self.preprocessing[::-1]:
            probas = preproc.proba_transform(probas)

        return probas

    def predict_probas_batched(self, test_set, batch_size=1):
        splits = mini_batch_split(test_set, batch_size)
        nb_batches = splits.size - 1
        samples_it = iter(test_set)
        probas = None
        offset = 0

        for i in range(nb_batches):
            cur_batch_size = splits[i+1] - splits[i]
            batch = []
            batch_labels = [frozenset([])] * cur_batch_size
            for j in range(cur_batch_size):
                batch.append(samples_it.next())
            batch_probas = self.predict_probas(self.named_conv.test_data_transform(
                IdentityDataset(batch, batch_labels)
            ))
            if probas == None:
                probas = np.empty([len(test_set), batch_probas.shape[1]])
            probas[offset:offset+cur_batch_size] = batch_probas
            offset += cur_batch_size
        return probas
    
    def learn_mlabel_threshold(self, train_set, batch_size=1):
        """ Learns a linear threshold function for multi-label prediction.
        """
        train_labels = map(lambda ls: frozenset(map(lambda l: self.named_conv.label_to_int[l], ls)), train_set.get_labels())
        train_confs = self.predict_probas_batched(train_set, batch_size)
        self.lin_thresh = learn_threshold(train_confs, train_labels)
    
    def predict_labels(self, images, method='top-1', batch_size=1, confidence=False):
        """ Predicts the label sets for a number of images.

        Arguments:
            images
                list of images to predict the labels for.
            method
                method to use to find the labels set:
                - top-1 just returns the one most probable label.
                - ('thresh', t) thresholds the probabilities at t, i.e. all probabilities
                  greater than t will be included in the labels set.
            confidence
                Set to true to return (label, confidence) pairs instead of just labels.
        """
        if method == 'top-1':
            probas = self.predict_probas_batched(images, batch_size=batch_size)
            max_idxs = np.argmax(probas, axis=1)
            labels = []
            nb_samples, nb_classes = probas.shape

            for i in range(nb_samples):
                if not confidence:
                    labels.append(frozenset([max_idxs[i]]))
                else:
                    labels.append(frozenset([(max_idxs[i], probas[i, max_idxs[i]])]))
            return labels
        elif method == 'thresh':
            probas = self.predict_probas_batched(images, batch_size=batch_size)
            nb_samples, nb_classes = probas.shape
            labels = []
            # Compute thresholds
            w, b = self.lin_thresh
            t = np.dot(probas, w) + b
            
            for i in range(nb_samples):
                labels_set = []
                for j in range(nb_classes):
                    if probas[i,j] > t[i]:
                        if not confidence:
                            labels_set.append(j)
                        else:
                            labels_set.append((j, probas[i,j]))
                labels.append(frozenset(labels_set))
            return labels
        else:
            raise ValueError("Invalid prediction method: " + repr(method))

    def init_random(self):
        """ Initializes a convolutional neural network at random given an 
        architecture. Convolutional weights are initialized by sampling a normal 
        distribution with mean 0 and variance 10^-2. Biases are initialized at 0. 
        Fully connected weights are initialized by sampling an uniform distribution
        within -/+ 4 * sqrt(6 / (nb_inputs + nb_outputs)) lower/higher bounds. 
        Softmax weights and biases are initialized at 0.

        Arguments:
            architecture
                see the documentation for BaseCNNClassifier.
            l2_reg
                l2 regularization factor (aka weight decay)
        Returns:
            a randomly initialized convolutional neural network.
        """
        layers = []
        nb_conv_mp = 0
        nb_fc = 0
        current_input_shape = self.input_shape
        i = 0
        
        # Initialize convolutional, max-pooling and FC layers.
        for layer_arch in self.architecture:
            # Pre-trained layers.
            if np.any([isinstance(layer_arch, c) for c in
                       [PoolLayer, ConvLayer, AveragePoolLayer]]):
                layers.append(layer_arch)
                nb_conv_mp += 1
                current_input_shape = layer_arch.output_shape(current_input_shape)
            elif isinstance(layer_arch, Layer):
                layers.append(layer_arch)
                nb_fc += 1
                current_input_shape = layer_arch.output_shape(current_input_shape)
            elif layer_arch == 'avg-pool':
                layer = AveragePoolLayer()
                layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
                nb_fc += 1
            elif layer_arch[0] == 'spp':
                layer = SPPLayer(layer_arch[1])
                layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
            elif layer_arch[0] == 'conv':
                input_dim = current_input_shape[0]
                p = layer_arch[1]
                nb_filters, nb_rows, nb_cols, stride_r, stride_c, std, bias, padding, nonlin = (
                    p['nb_filters'],
                    p['rows'],
                    p['cols'],
                    p['stride_r'] if 'stride_r' in p else 1,
                    p['stride_c'] if 'stride_c' in p else 1,
                    p['init_std'] if 'init_std' in p else 0.01,
                    p['init_bias'] if 'init_bias' in p else 0.,
                    p['padding'] if 'padding' in p else (0,0),
                    p['non_lin'] if 'non_lin' in p else 'relu'
                )
                in_pool_size = 1.
                if 0 < i < len(self.architecture)-1 and isinstance(self.architecture[i-1], tuple):
                    if i > 0 and self.architecture[i-1][0] == 'dropout':
                        in_pool_size = 1. / (1 - self.architecture[i-1][1])
                fan_in = nb_rows * nb_cols * input_dim / in_pool_size
                pool_size = 1.
                if isinstance(self.architecture[i-1], tuple) and isinstance(self.architecture[i+1], tuple):
                    if self.architecture[i+1] in ['avg-pool', 'spp']:
                        pool_size = nb_rows * nb_cols
                    elif self.architecture[i+1][0] == 'pool':
                        pool_size = self.architecture[i+1][1]['stride_r'] * self.architecture[i+1][1]['stride_c']
                    elif self.architecture[i+1][0] == 'dropout':
                        pool_size = 1. / (1 - self.architecture[i+1][1])
                fan_out = float(nb_filters * nb_rows * nb_cols) / pool_size
                filters = None
                biases = bias * np.ones(
                    [1, nb_filters, 1, 1],
                    theano.config.floatX
                )
                if self.init == 'random':
                    filters = np.random.uniform(
                        - np.sqrt(6 / (fan_in + fan_out)),
                        np.sqrt(6 / (fan_in + fan_out)),
                        [nb_filters, input_dim, nb_rows, nb_cols]
                    ).astype(theano.config.floatX)
                    print filters.std()
                elif self.init == 'orth':
                    # Generate a uniformly distributed random orthogonal matrix using
                    # the QR decomposition of a normally-dsitributed one.
                    X = np.random.normal(size=[nb_rows * nb_cols * input_dim, nb_filters])
                    Q, R = np.linalg.qr(X, mode='reduced')
                    filters = (
                        np.sqrt(6 / (fan_in + fan_out)) *
                        np.sqrt(nb_rows*nb_cols*input_dim) *
                        Q.T.reshape([nb_filters, input_dim, nb_rows, nb_cols]) / 2
                    ).astype(theano.config.floatX)
                    print filters.std()
                layer = ConvLayer(
                    'C' + repr(nb_conv_mp),
                    filters,
                    biases,
                    (stride_r, stride_c),
                    padding,
                    nonlin
                )
                layers.append(layer)
                nb_conv_mp += 1
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "Output of C" + repr(nb_conv_mp) + ": " + repr(current_input_shape)
                    print "fan_in: " + repr(fan_in) + ", fan_out: " + repr(fan_out)
            elif layer_arch[0] == 'pool':
                # Max pooling layers leave the input dimension unchanged.
                pool_type, rows, cols, stride_r, stride_c = (
                    layer_arch[1]['type'],
                    layer_arch[1]['rows'],
                    layer_arch[1]['cols'],
                    layer_arch[1]['stride_r'],
                    layer_arch[1]['stride_c']
                )
                layer = PoolLayer(pool_type, (rows, cols), (stride_r, stride_c))
                layers.append(layer)
                nb_conv_mp += 1
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "Output of M" + repr(nb_conv_mp) + ": " + repr(current_input_shape)
            elif layer_arch[0] == 'fc':
                # The inputs will be a flattened array of whatever came before.
                nb_inputs = int(np.prod(current_input_shape))
                p = layer_arch[1]
                nb_units, std, bias = (
                    p['nb_units'],
                    p['init_std'] if 'init_std' in p else 0.01,
                    p['init_bias'] if 'init_bias' in p else 0.
                )
                prob_alive_in = 1.
                prob_alive_out = 1.
                # If dropout after, the fan out is effectively multiplied by the probability of
                # the unit being alive in the first place. Same for the fan in.
                if 0 < i < len(self.architecture)-1:
                    if isinstance(self.architecture[i-1], DropoutLayer):
                        prob_alive_in = 1. - self.architecture[i-1].drop_proba
                    elif isinstance(self.architecture[i-1], tuple) and self.architecture[i-1][0] == 'dropout':
                        prob_alive_in = 1. - self.architecture[i-1][1]
                    if isinstance(self.architecture[i+1], DropoutLayer):
                        prob_alive_out = 1. - self.architecture[i+1].drop_proba
                    elif isinstance(self.architecture[i+1], tuple) and (isinstance(self.architecture[i+1], DropoutLayer)
                    or self.architecture[i+1][0] == 'dropout'):
                        prob_alive_out = 1. - self.architecture[i+1][1]
                weights = np.random.uniform(
                    -np.sqrt(6. / (nb_inputs * prob_alive_in + nb_units * prob_alive_out)),
                    np.sqrt(6. / (nb_inputs * prob_alive_in + nb_units * prob_alive_out)),
                    [nb_inputs, nb_units]
                ).astype(theano.config.floatX)
                biases = bias * np.ones(
                    [nb_units],
                    theano.config.floatX
                )
                layer = FCLayer('F' + repr(nb_fc), weights, biases)
                layers.append(layer)
                nb_fc += 1
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "FC layer " + repr(nb_fc)
                    print "fan in: " + repr(nb_inputs * prob_alive_in) + ", fan out: " + repr(nb_units * prob_alive_out)
            elif layer_arch[0] == 'linear':
                # The inputs will be a flattened array of whatever came before.
                nb_inputs = int(np.prod(current_input_shape))
                p = layer_arch[1]
                nb_outputs, std, bias = (
                    p['nb_outputs'],
                    p['init_std'] if 'init_std' in p else 0.01,
                    p['init_bias'] if 'init_bias' in p else 0.
                )
                prob_alive_in = 1.
                if (isinstance(self.architecture[i-1], NoTraining)
                    and isinstance(self.architecture[i-1].layer, DropoutLayer)):
                    prob_alive_in = 1. - self.architecture[i-1].layer.drop_proba
                elif isinstance(self.architecture[i-1], DropoutLayer):
                    prob_alive_in = 1. - self.architecture[i-1].drop_proba
                elif self.architecture[i-1][0] == 'dropout':
                    prob_alive_in = 1. - self.architecture[i-1][1]
                weights = np.random.uniform(
                    -np.sqrt(6 / (nb_inputs * prob_alive_in + nb_outputs)),
                    np.sqrt(6 / (nb_inputs + nb_outputs)),
                    [nb_inputs, nb_outputs]
                ).astype(theano.config.floatX)
                biases = bias * np.ones(
                    [nb_outputs],
                    theano.config.floatX
                )
                layer = LinearLayer('L', weights, biases)
                layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "Softmax layer " + repr(nb_fc)
                    print "fan in: " + repr(nb_inputs * prob_alive_in) + ", fan out: " + repr(nb_outputs)
            elif layer_arch[0] == 'dropout':
                drop_proba = layer_arch[1]
                layer_srng = RandomStreams(np.random.randint(0, 200000000))
                layers.append(DropoutLayer(
                    drop_proba=drop_proba,
                    srng=layer_srng,
                    input_shape=current_input_shape
                ))
                nb_fc += 1
            else:
                raise ValueError(repr(layer_arch) + 
                                 " is not a valid layer architecture.")
            i += 1
        
        return CNN(
            layers,
            np.prod(current_input_shape),
            self.l2_reg,
            0,
            cost = self.cost
        )

    def compute_activations(self, layer_number, test_set):
        """ Compute the activations of the network after the layer_numberth layer of the
            the network.
        """
        # Run the preprocessing pipeline.
        pp_images = test_set

        for preproc in self.preprocessing:
            pp_images = preproc.test_data_transform(pp_images)
        batch = pp_images.to_array()

        # Compile activation function.
        mode = theano.compile.get_default_mode()
        mode = mode.including('cudnn')
        activations = theano.function(
            [],
            self.model.compute_activations(layer_number, batch, test=True)
        )

        return activations()

    def spp_predict_confs(self, layer_number, test_set, pyramid):
        """ Compute SPP predictions for a CNN. The CNN needs to have confidence maps as
            activations of the specified layer.
        """
        # Run the preprocessing pipeline.
        pp_images = test_set

        for preproc in self.preprocessing:
            pp_images = preproc.test_data_transform(pp_images)
        confs = None
        # Compilation of the prediction function.
        images = T.tensor4('images')
        f_predict = theano.function(
            [images],
            spp_predict(self.model.compute_activations(layer_number, images, test=True), pyramid)
        )
        
        for i, sample in enumerate(pp_images):
            # Compute confidence maps for the sample.
            images = IdentityDataset([sample], [frozenset([0])]).to_array()
            confs_ = f_predict(images)
            nb_labels = confs_.shape[1]
            if confs == None:
                confs = np.zeros([len(test_set), nb_labels])
            confs[i] = confs_[0]
        return confs

class CNNClassifier(BaseCNNClassifier, ClassifierMixin):
    pass

class CNN:
    """ Convolutional neural network.
    """
    
    def __init__(self, layers, nb_classes, l2_reg=0, orth_penalty=0, cost='mlr'):
        """ Initializes a convolutional neural net with a specific architecture.
            Uses ReLU non-linearities, and dropout regularization.

        Arguments:
            layers
                initialized layers of the network.
            l2_reg
                l2 regularization (AKA weight decay) parameter.
            orth_penalty
                non-orthogonality penalty parameter.
            nb_classes
                number of classes this network classifies samples into. Number of
                labels in the multi-label setting.
        """
        self.layers = layers
        self.nb_classes = nb_classes
        self.l2_reg = l2_reg
        self.orth_penalty = orth_penalty
        self.cost = cost

    def parameters(self):
        """ Returns the parameters of the convnet, as a list of shared theano
            variables.
        """
        return reduce(lambda params, l1: params + l1.parameters(),
                      self.layers,
                      [])

    def forward_pass(self, batch, test=False):
        """ The forward pass of a convnet. This outputs a symbolic matrix of
            probabilities for each sample of the batch to belong to each class.

        Arguments:
            batch
                symbolic theano 4D tensor for a batch of images. Should have
                shape (nb_images, nb_channels, nb_rows, nb_cols).
        Returns:
            A symbolic (nb_images, nb_classes) matrix containing the output
            probabilities of the convnet classifier.
        """
        return self.compute_activations(len(self.layers) - 1, batch, test=test)

    def compute_activations(self, layer_number, batch, test=False):
        # First accumulate the convolutional layer forward passes.
        fpass = batch
        
        for i in range(layer_number + 1):
            fpass = self.layers[i].forward_pass(fpass)

        return fpass
    
    def cost_function(self, batch, labels, test=False):
        """ Returns the symbolic cost function of the convolutional neural
            net, using a multi-label generalization of the multinomial logistic
            regression cost function.

        Arguments:
            batch
                symbolic theano 4D tensor for a batch of images. Should have
                shape (nb_images, nb_channels, nb_rows, nb_cols).
            labels
                symbolic theano matrix of shape (nb_images, nb_classes) where
                labels[i,j] = 1 if j is a label of x_i else 0.
        Returns:
            A symbolic scalar representing the cost of the function for the given
            batch. Should be (sub) differentiable with regards to self.parameters().            
        """
        params_norm = 0

        for param in self.parameters():
            params_norm += T.sqr(param).sum()
        reg_term = 0 if self.l2_reg == 0 else self.l2_reg * params_norm / 2

        orth_penalty = 0
        
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                orth_penalty += layer.orth_penalty()
        orth_penalty = 0 if self.orth_penalty == 0 else self.orth_penalty * orth_penalty
        
        if self.cost == 'mlr':
            # The cost function basically sums the log softmaxed probabilities for the
            # correct
            # labels. We average the results to make it insensitive to batch size.
            
            return - T.mean(T.log(
                T.sum(
                    T.nnet.softmax(self.forward_pass(batch, test=test)) * labels,
                    axis=1
                )
            )) + reg_term + orth_penalty
        elif self.cost == 'multi-mlr':
            scores = self.forward_pass(batch, test=test)
            exp_scores = T.exp(scores - scores.max(axis=1, keepdims=True))
            label_scores = labels * exp_scores
            non_label_scores = (1. - labels) * exp_scores
            eps = 1E-10
            
            return - T.mean(T.log(
                (eps + label_scores.sum(axis=1))
                / (eps + non_label_scores.sum(axis=1))
            )) + reg_term + orth_penalty
        elif self.cost == 'bp-mll':
            # BP-MLL is much more complicated. First, we need to count the number
            # of labels and non-labels for each sample, to normalize the error measure
            # for each sample.
            nb_labels = T.sum(labels, axis=1)
            nb_non_labels = self.nb_classes - nb_labels
            err_norm_factor = 1. / (nb_labels * nb_non_labels)
            # Separately, we need to measure the error for each pair of label/non-label
            # for each sample. For simplicity of implementation, we do it in a dense
            # fashion.
            # M is a (batch_size, nb_classes, nb_classes) shaped mask, where M[i,j,k]
            # is 1 if j is a label and k a non-label of sample i, 0 otherwise. We compute
            # it by elementwise multiplication of the stacked label matrix with
            # the transpose of the non-labels matrix.
            _labels = labels.reshape((labels.shape[0], labels.shape[1], 1))
            stack_labels = T.concatenate([_labels] * self.nb_classes, axis=2)
            pairs_mask = stack_labels * (1 - stack_labels.dimshuffle(0, 2, 1))
            # In a similar fashion, we compute the error for all pairs of outputs from the
            # network, this time using an elementwise difference.
            fpass = self.forward_pass(batch, test=test)
            fpass = fpass.reshape((fpass.shape[0], fpass.shape[1], 1))
            stack_outputs = T.concatenate([fpass] * self.nb_classes, axis=2)
            pairs_err = stack_outputs - stack_outputs.dimshuffle(0, 2, 1)
            # Apply the exponential to the masked results.
            exp_pairs_err = T.exp(-pairs_mask * pairs_err)
            # Sum it up for all labels.
            error = T.sum(exp_pairs_err, axis=[1,2])
            # Normalize the error.
            norm_error = err_norm_factor * error
            # Add the regularization term and take the mean over the batch.
            return T.mean(norm_error) + reg_term + orth_penalty
        else:
            raise ValueError(repr(self.cost) + " is not a valid cost function.")

class Layer:
    def forward_pass(self, batch):
        """ Returns symbolic output feature maps from the input symbolic
            feature maps. The output feature maps should represent the computation
            of the layer, and should be automatically differentiable.

        Arguments:
            fmaps
                input theano symbolic feature maps, in a 4D tensor with shape
                (batch_size, in_dim, nb_rows, nb_cols).
        Returns:
            An output theano symbolic tensor.
        """
        raise NotImplementedError()
    
    def parameters(self):
        """ Returns a list of shared theano variables corresponding to the layer parameters
            to optimize.
        """
        raise NotImplementedError()

    def output_shape(self, sample_shape):
        """ Returns the shape of the output of the forward pass on a single sample.
        """
        raise NotImplementedError()

class DropoutLayer(Layer):
    """ Dropout layer. Randomly sets inputs to 0.
    """
    def __init__(self, drop_proba, srng, input_shape):
        """ Initializes the dropout layer with a given dropout probability and a shared
            random number generator.
        """
        self.drop_proba = drop_proba
        self.srng = srng
        self.input_shape = input_shape

    def forward_pass(self, input_tensor, test=False):
        # At test time, no dropout, but multiplying weights with the dropout
        # probability. Note that with ReLU non-linearity, one can simply multiply
        # the outputs.
        if test:
            return (1. - self.drop_proba) * input_tensor
        else:
            # Generate a mask of ones and zeros of the shape of the input tensor.
            mask = self.srng.binomial(
                size=(input_tensor.shape[0], input_tensor.shape[1], 1, 1) if len(self.input_shape) == 2 else input_tensor.shape,
                n=1,
                p=1. - self.drop_proba,
                nstreams=np.prod(self.input_shape),
            )
            if len(self.input_shape) == 2:
                return input_tensor * T.addbroadcast(T.cast(mask, theano.config.floatX), 2, 3)
            else:
                return input_tensor * T.cast(mask, theano.config.floatX)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        return input_shape

class LinearLayer(Layer):
    """ Simple linear layer.
    """
    def __init__(self, name, init_weights, init_biases):
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        assert init_weights.shape[1] == init_biases.shape[0]
        self.nb_inputs, self.nb_outputs = init_weights.shape
        self.weights = theano.shared(init_weights, name=name+'_W')
        self.biases = theano.shared(init_biases, name=name+'_b')
        self.name = name

    def forward_pass(self, input_matrix, test=False):
        return T.dot(T.flatten(input_matrix, outdim=2), self.weights) + self.biases

    def parameters(self):
        return [self.weights, self.biases]

    def output_shape(self, input_shape):
        return [self.nb_outputs]

    def __getstate__(self):
        return (self.nb_inputs, self.nb_outputs, self.weights.get_value(),
                self.biases.get_value(), self.name)

    def __setstate__(self, state):
        self.nb_inputs, self.nb_outputs, weights, biases, self.name = state
        self.weights = theano.shared(weights, name=self.name+'_W')
        self.biases = theano.shared(biases, name=self.name+'_b')

    def to_conv(self):
        """ Converts this layer into a 1x1 convolutional layer with no
            non-linearity that basically applies this linear layer
            pixel-wise. Useful for producing confidence maps at prediction
            time, when using a linear layer implementation over GAP
            layers is best for training.
        """
        return ConvLayer(
            self.name,
            self.weights.get_value().T.reshape([self.nb_outputs, self.nb_inputs, 1, 1]),
            self.biases.get_value().reshape([1, self.nb_outputs, 1, 1]),
            pooling=(1,1),
            padding=(0,0),
            nonlin=None
        )

class FCLayer(Layer):
    """ Fully connected layer with ReLU non-linearity.
    """
    
    def __init__(self, name, init_weights, init_biases):
        """ Initializes the fully-connected layer with a matrix of
            initial weights and a vector of initial biases.

        Arguments:
            name
                a name for the layer.
            init_weights
                matrix of nb_inputs by nb_outputs initial weights.
            init_biases
                vector of nb_outputs initial biases.
        """
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        assert init_weights.shape[1] == init_biases.shape[0]
        self.nb_inputs, self.nb_outputs = init_weights.shape
        self.weights = theano.shared(init_weights, name=name+'_W')
        self.biases = theano.shared(init_biases, name=name+'_b')
        self.name = name

    def forward_pass(self, input_matrix, test=False):
        return T.maximum(T.dot(T.flatten(input_matrix, outdim=2), self.weights) + self.biases, 0)

    def parameters(self):
        return [self.weights, self.biases]

    def output_shape(self, input_shape):
        return [self.nb_outputs]

    def __getstate__(self):
        return (self.nb_inputs, self.nb_outputs, self.weights.get_value(),
                self.biases.get_value(), self.name)

    def __setstate__(self, state):
        self.nb_inputs, self.nb_outputs, weights, biases, self.name = state
        self.weights = theano.shared(weights, name=self.name+'_W')
        self.biases = theano.shared(biases, name=self.name+'_b')

    def to_conv(self):
        """ Converts this layer into a 1x1 convolutional layer with no
            ReLU non-linearity that basically applies this FC layer
            pixel-wise. Useful for producing confidence maps at prediction
            time, when using a linear layer implementation over GAP
            layers is best for training.
        """
        return ConvLayer(
            self.name,
            self.weights.get_value().T.reshape([self.nb_outputs, self.nb_inputs, 1, 1]),
            self.biases.get_value().reshape([1, self.nb_outputs, 1, 1]),
            pooling=(1,1),
            padding=(0,0),
            nonlin='relu'
        )

class ConvLayer(Layer):
    """ Convolutional layer with ReLU non-linearity and max-pooling.
    """
    
    def __init__(self, name, init_filters, init_biases, pooling=(1,1), padding=(0,0),
                 nonlin='relu'):
        """ Initializes a convolutional layer with a set of initial filters and 
            biases.

        Arguments:
            name
                a name for the layer.
            init_filters
                4D numpy array of shape (nb_filters, nb_features, nb_rows, nb_cols).
            init_biases
                numpy vector of nb_filters elements for initial bias values.
            fmap_shape
                shape of the input feature map, optional. Used for internal theano
                optimizations.
        """
        assert len(init_filters.shape) == 4
        assert len(init_biases.shape) == 4
        assert init_filters.shape[0] == init_biases.shape[1]
        # Store the initial filters in a shared variable.
        self.filters = theano.shared(init_filters, name+'_W')
        self.biases = theano.shared(init_biases, name+'_b')
        self.filters_shape = init_filters.shape
        self.pooling = pooling
        self.padding = padding
        self.name = name
        self.nonlin = nonlin
        
    def forward_pass(self, fmaps):
        # Computes the raw convolution output, depending on the desired
        # implementation.
        out_fmaps = dnn_conv(
            fmaps,
            self.filters,
            border_mode=self.padding,
            subsample=self.pooling
        )
        if self.nonlin == 'relu':
            # Add biases and apply ReLU non-linearity.
            relu_fmaps = T.maximum(
                0, 
                out_fmaps + T.addbroadcast(self.biases, 0, 2, 3)
            )
            return relu_fmaps
        elif self.nonlin == None:
            return out_fmaps + T.addbroadcast(self.biases, 0, 2, 3)
        else:
            raise ValueError("Invalid non-linearity: " + repr(self.nonlin))            

    def parameters(self):
        return [self.filters, self.biases]

    def orth_penalty(self):
        flat_filters = T.flatten(self.filters, outdim=2)
        penalties = T.dot(flat_filters, flat_filters.T)
        diag_mask = 1 - T.identity_like(penalties)

        return T.abs_((penalties * diag_mask).sum() / 2)

    def output_shape(self, input_shape):
        pad_r, pad_c = self.padding
        out_shape = [
            input_shape[1] - self.filters_shape[2] + 1 + pad_r * 2,
            input_shape[2] - self.filters_shape[3] + 1 + pad_c * 2
        ]
        
        return [
            self.filters_shape[0],
            out_shape[0] // self.pooling[0],
            out_shape[1] // self.pooling[1]
        ]

    def __getstate__(self):
        return (self.filters.get_value(), self.biases.get_value(), self.filters_shape,
                self.padding, self.pooling, self.name, self.nonlin)

    def __setstate__(self, state):
        filters, biases, self.filters_shape, self.padding, self.pooling, self.name, self.nonlin = state
        self.filters = theano.shared(filters, self.name+'_W')
        self.biases = theano.shared(biases, self.name+'_b')
    
class PoolLayer(Layer):
    """ Non-overlapping max pooling layer.
    """
    def __init__(self, pool_type, pooling_size, stride):
        """ Initialize a max-pooling layer with a given pooling window size.

        Arguments:
            pooling_size
                size of the max pooling window.
        """
        self.pool_type = pool_type
        self.pooling_size = pooling_size
        self.stride = stride

    def forward_pass(self, fmaps):
        return dnn_pool(fmaps, self.pooling_size, self.stride, mode=self.pool_type)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        overlap_r = max(0, self.pooling_size[0] - self.stride[0])
        overlap_c = max(0, self.pooling_size[1] - self.stride[1])

        return [
            input_shape[0],
            (input_shape[1] - overlap_r) // self.stride[0],
            (input_shape[2] - overlap_c) // self.stride[1]
        ]

class AveragePoolLayer(Layer):
    """ Global average pooling layer.
    """
    def __init__(self):
        pass

    def forward_pass(self, fmaps):
        # Computes the mean of each individual feature map.
        return T.mean(fmaps, axis=[2,3], keepdims=True)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        return [
            input_shape[0],
            1,
            1
        ]

class NoTraining(Layer):
    """ Wraps another layer, except that it will not be trained. Useful for supervised
        pre-training.
    """
    def __init__(self, layer):
        self.layer = layer

    def forward_pass(self, batch):
        return self.layer.forward_pass(batch)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        return self.layer.output_shape(input_shape)

class SPPLayer(Layer):
    """ Spatial pyramid pooling + spatial label pruning.
    """
    def __init__(self, pyramid):
        self.pyramid = pyramid

    def forward_pass(self, batch):
        return spp_predict(batch, self.pyramid)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        return [
            input_shape[0],
            1,
            1
        ]
