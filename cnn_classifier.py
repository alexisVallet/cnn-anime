import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool, GpuDnnSoftmax
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn.cluster import KMeans
import copy

from classifier import ClassifierMixin

class BaseCNNClassifier:
    """ Image classifier based on a convolutional neural network.
    """
    def __init__(self, architecture, optimizer, input_shape,
                 srng, init='random', cost='bp-mll', l2_reg=0, preprocessing=[],
                 verbose=False):
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
                    init_std: ,  defaults to 0.01
                    init_bias:  defaults to 0
                  })
                - ('max-pool', {
                    rows: ,
                    cols: ,
                    stride_r: ,
                    stride_c:
                  })
                - 'avg-pool'
                - ('fc', {
                    nb_units: ,
                    init_std: , defaults to 0.01
                    init_bias:  defaults to 0
                  })
                - ('softmax', {
                    nb_outputs; ,
                    init_std: , defaults to 0.01
                    init_bias:  defaults to 0
                  })
                - ('dropout', dropout_proba)
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
            cost
                cost function to use for training. One of:
                - 'log-scaled' for log-scaled 
            l2_reg
                parameter controlling the strength of l2 regularization.
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
        self.l2_reg = l2_reg
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.model = None
        self.named_conv = None
        self.loaded = False

    def __getstate__(self):        
        return (self.architecture, self.optimizer, self.input_shape, self.srng, self.init,
                self.l2_reg, self.preprocessing, self.verbose, self.model, self.named_conv)

    def __setstate__(self, state):
        (self.architecture, self.optimizer, self.input_shape, self.srng, self.init,
         self.l2_reg, self.preprocessing, self.verbose, self.model, self.named_conv) = state
        self.compile_prediction()
        self.loaded = True
    
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
        if self.verbose:
            print "Initializing weights..."
        # If loaded from a file, use the already initialized weights.
        if not self.loaded:
            if self.init == 'random':
                self.model = self.init_random()
            else:
                raise ValueError(repr(self.init) + 
                                " is not a valid initialization method.")
        self.compile_prediction()
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

        # Then convert the probabilities (for instance averageing when
        # samples were duplicated, etc).
        for preproc in self.preprocessing[::-1]:
            probas = preproc.proba_transform(probas)

        return probas

    def predict_labels(self, images, method='top-1'):
        """ Predicts the label sets for a number of images.

        Arguments:
            images
                list of images to predict the labels for.
            method
                method to use to find the labels set:
                - top-1 just returns the one most probable label.
                - ('thresh', t) thresholds the probabilities at t, i.e. all probabilities
                  greater than t will be included in the labels set. If there are no such
                  probabilities, then default to top-1.
        """
        if method == 'top-1':
            return np.argmax(self.predict_probas(images), axis=1)
        elif method[0] == 'thresh':
            t = method[1]
            probas = self.predict_probas(images)
            nb_samples, nb_classes = probas.shape
            labels = []
            
            for i in range(nb_samples):
                labels_set = []
                for j in range(nb_classes):
                    if probas[i,j] > t:
                        labels_set.append(j)
                if labels_set == []:
                    labels_set.append(np.argmax(probas[i]))
                labels.append(frozenset(labels_set))
            return labels

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
                       [MaxPoolLayer, ConvLayer, AveragePoolLayer]]):
                layers.append(layer_arch)
                nb_conv_mp += 1
                current_input_shape = layer_arch.output_shape(current_input_shape)
            elif isinstance(layer_arch, Layer):
                layers.append(layer_arch)
                nb_fc += 1
                current_input_shape = layer_arch.output_shape(current_input_shape)
            elif layer_arch == 'avg-pool':
                layer = AveragePoolLayer()
                current_input_shape = layer.output_shape(current_input_shape)
                layers.append(layer)
                nb_conv_mp += 1
            elif layer_arch[0] == 'conv':
                input_dim = current_input_shape[0]
                p = layer_arch[1]
                nb_filters, nb_rows, nb_cols, stride_r, stride_c, std, bias = (
                    p['nb_filters'],
                    p['rows'],
                    p['cols'],
                    p['stride_r'] if 'stride_r' in p else 1,
                    p['stride_c'] if 'stride_c' in p else 1,
                    p['init_std'] if 'init_std' in p else 0.01,
                    p['init_bias'] if 'init_bias' in p else 0.
                )
                fan_in = nb_rows * nb_cols * input_dim
                pool_size = None
                if self.architecture[i+1] == 'avg-pool':
                    pool_size = nb_rows * nb_cols
                elif self.architecture[i+1][0] == 'max-pool':
                    pool_size = self.architecture[i+1][1]['stride_r'] * self.architecture[i+1][1]['stride_c']
                else:
                    pool_size = 1
                fan_out = float(nb_filters * nb_rows * nb_cols) / pool_size
                filters = np.random.uniform(
                    - np.sqrt(6 / (fan_in + fan_out)),
                    np.sqrt(6 / (fan_in + fan_out)),
                    [nb_filters, input_dim, nb_rows, nb_cols]
                ).astype(theano.config.floatX)
                biases = bias * np.ones(
                    [1, nb_filters, 1, 1],
                    theano.config.floatX
                )
                layer = ConvLayer('C' + repr(nb_conv_mp), filters, biases, (stride_r, stride_c))
                layers.append(layer)
                nb_conv_mp += 1
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "Output of C" + repr(nb_conv_mp) + ": " + repr(current_input_shape)
                    print "fan_in: " + repr(fan_in) + ", fan_out: " + repr(fan_out)
            elif layer_arch[0] == 'max-pool':
                # Max pooling layers leave the input dimension unchanged.
                rows, cols, stride_r, stride_c = (
                    layer_arch[1]['rows'],
                    layer_arch[1]['cols'],
                    layer_arch[1]['stride_r'],
                    layer_arch[1]['stride_c']
                )
                layer = MaxPoolLayer((rows, cols), (stride_r, stride_c))
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
                if isinstance(self.architecture[i-1], DropoutLayer):
                    prob_alive_in = 1. - self.architecture[i-1].drop_proba
                elif self.architecture[i-1][0] == 'dropout':
                    prob_alive_in = 1. - self.architecture[i-1][1]
                if isinstance(self.architecture[i+1], DropoutLayer):
                    prob_alive_out = 1. - self.architecture[i+1].drop_proba
                elif (isinstance(self.architecture[i+1], DropoutLayer)
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
            elif layer_arch[0] == 'softmax':
                # The inputs will be a flattened array of whatever came before.
                nb_inputs = int(np.prod(current_input_shape))
                p = layer_arch[1]
                nb_outputs, std, bias = (
                    p['nb_outputs'],
                    p['init_std'] if 'init_std' in p else 0.01,
                    p['init_bias'] if 'init_bias' in p else 0.
                )
                prob_alive_in = 1.
                if isinstance(self.architecture[i-1], DropoutLayer):
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
                layer = SoftmaxLayer('S', weights, biases)
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

        if len(layers) != nb_conv_mp + nb_fc + 1:
            raise ValueError(
                "The architecture should finish with exactly one Softmax layer."
            )
        
        return CNN(
            layers[0:nb_conv_mp], 
            layers[nb_conv_mp:nb_conv_mp+nb_fc],
            layers[-1],
            self.l2_reg
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

class CNNClassifier(BaseCNNClassifier, ClassifierMixin):
    pass

class CNN:
    """ Convolutional neural network.
    """
    
    def __init__(self, conv_mp_layers, fc_layers, softmax_layer, l2_reg=0):
        """ Initializes a convolutional neural net with a specific architecture.
            Uses ReLU non-linearities, and dropout regularization.

        Arguments:
            conv_layers
                list of layer objects specifying the architecture of the
                convolution and max-pooling layers.
            fc_layers
                list of ints specifying the width of each final fully
                connected layer.
            conv_init
                initialization procedure for convolutional filters.
            fc_init
                initialization procedure for fully-connected layer weights.
        """
        assert np.all([isinstance(l, ConvLayer) or isinstance(l, MaxPoolLayer)
                       or isinstance(l, AveragePoolLayer)
                       for l in conv_mp_layers])
        assert np.all([isinstance(l, FCLayer) or isinstance (l, DropoutLayer) for l in fc_layers])
        assert isinstance(softmax_layer, SoftmaxLayer)
        self.conv_mp_layers = conv_mp_layers
        self.fc_layers = fc_layers
        self.softmax_layer = softmax_layer
        self.l2_reg = l2_reg
        self.nb_classes = softmax_layer.nb_outputs

    def parameters(self):
        """ Returns the parameters of the convnet, as a list of shared theano
            variables.
        """
        return reduce(lambda params, l1: params + l1.parameters(),
                      self.conv_mp_layers + self.fc_layers + [self.softmax_layer],
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
        # First accumulate the convolutional layer forward passes.
        fpass = batch

        for conv_mp_layer in self.conv_mp_layers:
            fpass = conv_mp_layer.forward_pass(fpass)

        # Reshape the output (batch_size, nb_features, nb_rows, nb_cols)
        # tensor into a data matrix (batch_size, nb_features * nb_rows * nb_cols)
        # suitable to be fed to the FC layers.
        fpass = T.flatten(fpass, outdim=2)

        # Then accumulate the FC layer forward passes.
        for fc_layer in self.fc_layers:
            fpass = fc_layer.forward_pass(fpass, test=test)

        # Finally, apply the final softmax layer.
        fpass = self.softmax_layer.forward_pass(fpass)

        return fpass

    def compute_activations(self, layer_number, batch, test=False):
        # First accumulate the convolutional layer forward passes.
        nb_layers = 0
        fpass = batch

        for conv_mp_layer in self.conv_mp_layers:
            fpass = conv_mp_layer.forward_pass(fpass)
            nb_layers += 1
            if nb_layers >= layer_number:
                return fpass

        # Reshape the output (batch_size, nb_features, nb_rows, nb_cols)
        # tensor into a data matrix (batch_size, nb_features * nb_rows * nb_cols)
        # suitable to be fed to the FC layers.
        fpass = T.flatten(fpass, outdim=2)

        # Then accumulate the FC layer forward passes.
        for fc_layer in self.fc_layers:
            fpass = fc_layer.forward_pass(fpass, test=test)
            nb_layers += 1
            if nb_layers >= layer_number:
                return fpass

        # Finally, apply the final softmax layer.
        fpass = self.softmax_layer.forward_pass(fpass)

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
        # The cost function basically sums the log softmaxed probabilities for the
        # correct
        # labels. We average the results to make it insensitive to batch size.
        params_norm = 0

        for param in self.parameters():
            params_norm += T.dot(param.flatten(), param.flatten())

        params_norm = T.sqrt(params_norm)

        cost = - T.mean(T.log(
            T.sum(
                self.forward_pass(batch, test=test) * labels,
                axis=1
            )
        )) #+ self.l2_reg * params_norm / 2

        return cost

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
                size=input_tensor.shape,
                n=1,
                p=1. - self.drop_proba,
                nstreams=np.prod(self.input_shape),
            )
        
            return input_tensor * T.cast(mask, theano.config.floatX)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        return input_shape
    
class SoftmaxLayer(Layer):
    """ Softmax layer. Essentially multinomial logistic regression.
    """

    def __init__(self, name, init_weights, init_biases):
        """ Initializes the softmax layer with a matrix of initial
            weights and a vector of initial biases.

        Arguments:
            name
                name for the layer.
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

    def forward_pass(self, input_matrix):
        return T.nnet.softmax(T.dot(input_matrix, self.weights) + self.biases)

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
        return T.maximum(T.dot(input_matrix, self.weights) + self.biases, 0)

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

class ConvLayer(Layer):
    """ Convolutional layer with ReLU non-linearity and max-pooling.
    """
    
    def __init__(self, name, init_filters, init_biases, pooling=(1,1)):
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
        self.name = name
        
    def forward_pass(self, fmaps):
        # Computes the raw convolution output, depending on the desired
        # implementation.
        out_fmaps = dnn_conv(
            fmaps,
            self.filters,
            border_mode='valid',
            subsample=self.pooling
        )
        nb_filters = self.filters_shape[0]
        # Add biases and apply ReLU non-linearity.
        relu_fmaps = T.maximum(
            0, 
            out_fmaps +  T.addbroadcast(self.biases, 0, 2, 3)
        )
        return relu_fmaps

            
    def parameters(self):
        return [self.filters, self.biases]

    def output_shape(self, input_shape):
        out_shape = [
            input_shape[1] - self.filters_shape[2] + 1,
            input_shape[2] - self.filters_shape[3] + 1
        ]
        
        return [
            self.filters_shape[0],
            out_shape[0] // self.pooling[0],
            out_shape[1] // self.pooling[1]
        ]

    def __getstate__(self):
        return (self.filters.get_value(), self.biases.get_value(), self.filters_shape,
                self.pooling, self.name)

    def __setstate__(self, state):
        filters, biases, self.filters_shape, self.pooling, self.name = state
        self.filters = theano.shared(filters, self.name+'_W')
        self.biases = theano.shared(biases, self.name+'_b')
    
class MaxPoolLayer(Layer):
    """ Non-overlapping max pooling layer.
    """
    def __init__(self, pooling_size, stride):
        """ Initialize a max-pooling layer with a given pooling window size.

        Arguments:
            pooling_size
                size of the max pooling window.
        """
        self.pooling_size = pooling_size
        self.stride = stride

    def forward_pass(self, fmaps):        
        return dnn_pool(fmaps, self.pooling_size, self.stride)

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
        # Computes the mean of each individual feature map, effectively making
        # them into 1 by 1 feature maps.
        return T.mean(fmaps, axis=[2,3], keepdims=True)

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        return [
            input_shape[0],
            1,
            1
        ]
