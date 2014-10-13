import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
from sklearn.cluster import KMeans

from classifier import ClassifierMixin

class BaseCNNClassifier:
    """ Image classifier based on a convolutional neural network.
    """
    def __init__(self, architecture, optimizer, input_shape, init='random',
                 l2_reg=0, verbose=False):
        """ Initializes a convolutional neural network with a specific
            architecture, optimization method for training and 
            initialization procedure.

        Arguments:
            architecture
                list of tuples which should be either:
                - ('conv', nb_filters, nb_rows, nb_cols)
                - ('max-pool', win_size)
                - ('fc', nb_outputs)
                - ('softmax', nb_outputs)
            optimizer
                optimization method to use for training the network. Should be
                an instance of Optimizer.
            input_shape
                shape of input images, in [nb_channels, nb_rows, nb_cols] format.
            init
                initialization procedure for the network. Can be:
                - 'random' for random initialization of network weights.
                - ('unsupervised', batch_size, nb_subwins) for greedy, layer-wise 
                  unsupervised pre-training.
            l2_reg
                parameter controlling the strength of l2 regularization.
        """
        self.architecture = architecture
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.init = init
        self.l2_reg = l2_reg
        self.verbose = verbose

    def train(self, images, labels):
        # Use theano's FFT convolutions.
        mode = theano.compile.get_default_mode()
        mode = mode.including('conv_fft_valid', 'conv_fft_full')
        # Initialize the CNN with the desired method.
        cnn = None
        if self.init == 'random':
            cnn = self.init_random()
        elif self.init[0] == 'unsupervised':
            cnn = self.init_unsupervised(images, self.init[1], self.init[2], mode)
        else:
            raise ValueError(repr(self.init) + 
                             " is not a valid initialization method.")
        # Compile the CNN prediction functions.
        test_samples = T.tensor4('test_samples')
        self._predict_proba = theano.function(
            [test_samples],
            cnn.forward_pass(test_samples)
        )
        self._predict_label = theano.function(
            [test_samples],
            T.argmax(cnn.forward_pass(test_samples), axis=1)
        )

        # Run the optimizer on the CNN cost function.        
        self.optimizer.optimize(
            images,
            labels,
            cnn.cost_function,
            cnn.parameters(),
            compile_mode=mode
        )

    def predict_proba(self, images):
        return self._predict_proba(images.to_array())

    def predict_label(self, images):
        return self._predict_proba(images.to_array())

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
        std = 0.01
        current_input_shape = self.input_shape
        
        # Initialize convolutional, max-pooling and FC layers.
        for layer_arch in self.architecture:
            if layer_arch[0] == 'conv':
                input_dim = current_input_shape[0]
                nb_filters, nb_rows, nb_cols = layer_arch[1:]
                filters = std * np.random.standard_normal(
                    [nb_filters, input_dim, nb_rows, nb_cols]
                ).astype(theano.config.floatX)
                biases = np.ones(
                    [nb_filters],
                    theano.config.floatX
                )
                layer = ConvLayer(filters, biases)
                layers.append(layer)
                nb_conv_mp += 1
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "Output of C" + repr(nb_conv_mp) + ": " + repr(current_input_shape)
            elif layer_arch[0] == 'max-pool':
                # Max pooling layers leave the input dimension unchanged.
                pooling_size = layer_arch[1]
                layer = MaxPoolLayer(pooling_size)
                layers.append(layer)
                nb_conv_mp += 1
                current_input_shape = layer.output_shape(current_input_shape)
                if self.verbose:
                    print "Output of M" + repr(nb_conv_mp) + ": " + repr(current_input_shape)
            elif layer_arch[0] == 'fc':
                # The inputs will be a flattened array of whatever came before.
                nb_inputs = int(np.prod(current_input_shape))
                nb_neurons = layer_arch[1]
                weights = std * np.random.standard_normal(
                    [nb_inputs, nb_neurons]
                ).astype(theano.config.floatX)
                # w_bound = 4. * np.sqrt(6. / (nb_inputs + nb_neurons))
                # weights =  np.random.uniform(
                #     -w_bound, 
                #     w_bound, 
                #     [nb_inputs, nb_neurons]
                # ).astype(theano.config.floatX)
                biases = np.ones(
                    [nb_neurons],
                    theano.config.floatX
                )
                layer = FCLayer(weights, biases)
                layers.append(layer)
                nb_fc += 1
                current_input_shape = layer.output_shape(current_input_shape)
            elif layer_arch[0] == 'softmax':
                # The inputs will be a flattened array of whatever came before.
                nb_inputs = int(np.prod(current_input_shape))
                nb_outputs = layer_arch[1]
                weights = np.zeros(
                    [nb_inputs, nb_outputs],
                    theano.config.floatX
                )
                biases = np.zeros(
                    [nb_outputs],
                    theano.config.floatX
                )
                layer = SoftmaxLayer(weights, biases)
                layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
            else:
                raise ValueError(repr(layer_arch) + 
                                 " is not a valid layer architecture.")

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

    def init_unsupervised(self, images, batch_size, nb_subwins, compile_mode):
        """ Initializes the weights of the neural network using greedy layer-wise
            pre training.

        Arguments:
            images
               training data to use for initialization.
            batch_size
               number of random training images to use for the procedure. Should
               be big enough to be representative of the whole dataset, but small
               enough to fit in memory.
            nb_subwins
               number of random subwindows to sample from the training images
               to use for the procedure.

        Returns:
            an initialized convolutional neural network.
        """
        # Get the random batch of training images.
        batch = np.empty(
            [batch_size] + images.sample_shape,
            theano.config.floatX
        )
        images.shuffle(np.random.permutation(len(images)))
        images_it = iter(images)

        for i in range(batch_size):
            batch[i] = images_it.next()

        # Then, layer by layer, initialize the convolutional filters using
        # K-means centers from randomly samples subwindows of the input. Then
        # run a forward pass of the input on the initialized layer to get the
        # input from the next one.
        current_input = batch
        current_input_shape = self.input_shape
        conv_mp_layers = []
        fc_layers = []
        softmax_layer = None
        
        for layer_arch in self.architecture:
            if layer_arch[0] == 'conv':
                if self.verbose:
                    print "Unsupervised pre-training for conv layer..."
                nb_filters, nb_rows, nb_cols = layer_arch[1:]
                nb_input_dim, inp_rows, inp_cols = current_input_shape
                filter_shape = [nb_input_dim, nb_rows, nb_cols]
                # Pick random subwindows out of the batch, run KMeans on them.
                X = np.empty(
                    [nb_subwins, np.prod(filter_shape)]
                )
                for i in range(nb_subwins):
                    rand_sample = np.random.randint(batch_size)
                    rand_row = np.random.randint(inp_rows - nb_rows - 1)
                    rand_col = np.random.randint(inp_cols - nb_cols - 1)
                    X[i] = batch[rand_sample][:, 
                                              rand_row:rand_row+nb_rows,
                                              rand_col:rand_col+nb_cols].flatten('C')
                km = KMeans(nb_filters)
                km.fit(X)
                filters = km.cluster_centers_.reshape(
                    [nb_filters] + filter_shape
                ).astype(theano.config.floatX)
                biases = np.zeros([nb_filters], theano.config.floatX)
                layer = ConvLayer(filters, biases)
                conv_mp_layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
                # Run a forward pass to get the new batch.
                batch = theano.function(
                    [],
                    layer.forward_pass(batch),
                    mode=compile_mode
                )()
            if layer_arch[0] == 'max-pool':
                if self.verbose:
                    print "Unsupervised pre-training for max-pooling layer..."
                pooling_size = layer_arch[1]
                layer = MaxPoolLayer(pooling_size)
                conv_mp_layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
                # Run a forward pass to get the new batch.
                batch = theano.function(
                    [],
                    layer.forward_pass(batch),
                    mode=compile_mode
                )()
            if layer_arch[0] == 'fc':
                if self.verbose:
                    print "Unsupervised pre-training for FC layer..."
                # For FC layers, run KMeans on the whole batch, and retrieve
                # as many centers as output neurons
                nb_outputs = layer_arch[1]
                nb_inputs = np.prod(current_input_shape)
                km = KMeans(nb_outputs)
                km.fit(batch.reshape([batch_size, nb_inputs]))
                weights = km.cluster_centers_.astype(theano.config.floatX).T
                biases = np.zeros([nb_outputs], theano.config.floatX)
                layer = FCLayer(weights, biases)
                fc_layers.append(layer)
                current_input_shape = layer.output_shape(current_input_shape)
                # Run a forward pass to get the new batch.
                batch = theano.function(
                    [],
                    layer.forward_pass(batch.reshape([batch_size, nb_inputs])),
                    mode=compile_mode
                )()
            if layer_arch[0] == 'softmax':
                if self.verbose:
                    print "Unsupervised pre-training for softmax layer..."
                # Same as FC layer.
                nb_outputs = layer_arch[1]
                nb_inputs = np.prod(current_input_shape)
                km = KMeans(nb_outputs)
                km.fit(batch.reshape([batch_size, nb_inputs]))
                weights = km.cluster_centers_.astype(theano.config.floatX).T
                biases = np.zeros([nb_outputs], theano.config.floatX)
                softmax_layer = SoftmaxLayer(weights, biases)
                current_input_shape = softmax_layer.output_shape(current_input_shape)
                # Run a forward pass to get the new batch.
                batch = theano.function(
                    [],
                    softmax_layer.forward_pass(
                        batch.reshape([batch_size, nb_inputs])
                    ),
                    mode=compile_mode
                )()
        return CNN(conv_mp_layers, fc_layers, softmax_layer, self.l2_reg)

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
                       for l in conv_mp_layers])
        assert np.all([isinstance(l, FCLayer) for l in fc_layers])
        assert isinstance(softmax_layer, SoftmaxLayer)
        self.conv_mp_layers = conv_mp_layers
        self.fc_layers = fc_layers
        self.softmax_layer = softmax_layer
        self.l2_reg = l2_reg

    def parameters(self):
        """ Returns the parameters of the convnet, as a list of shared theano
            variables.
        """
        return reduce(lambda params, l1: params + l1.parameters(),
                      self.conv_mp_layers + self.fc_layers + [self.softmax_layer],
                      [])

    def forward_pass(self, batch):
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
            fpass = fc_layer.forward_pass(fpass)

        # Finally, apply the final softmax layer.
        fpass = self.softmax_layer.forward_pass(fpass)

        return fpass
    
    def cost_function(self, batch, labels):
        """ Returns the symbolic cost function of the convolutional neural
            net.

        Arguments:
            batch
                symbolic theano 4D tensor for a batch of images. Should have
                shape (nb_images, nb_channels, nb_rows, nb_cols).
            labels
                symbolic theano 1D int tensor for corresponding image labels.
                Should have shape (nb_images,) . Labels should be in the {0, nb_classes-1}
                set.
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

        cost = - T.mean(
            T.log(self.forward_pass(batch)[T.arange(batch.shape[0]),
                                           labels])
        ) + self.l2_reg * params_norm / 2

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
    
class SoftmaxLayer(Layer):
    """ Softmax layer. Essentially multinomial logistic regression.
    """

    def __init__(self, init_weights, init_biases):
        """ Initializes the softmax layer with a matrix of initial
            weights and a vector of initial biases.

        Arguments:
            init_weights
                matrix of nb_inputs by nb_outputs initial weights.
            init_biases
                vector of nb_outputs initial biases.
        """
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        assert init_weights.shape[1] == init_biases.shape[0]
        self.nb_inputs, self.nb_outputs = init_weights.shape
        self.weights = theano.shared(init_weights)
        self.biases = theano.shared(init_biases)

    def forward_pass(self, input_matrix):
        return T.nnet.softmax(T.dot(input_matrix, self.weights) + self.biases)

    def parameters(self):
        return [self.weights, self.biases]

    def output_shape(self, input_shape):
        return [self.nb_outputs]
    
class FCLayer(Layer):
    """ Fully connected layer with ReLU non-linearity.
    """
    
    def __init__(self, init_weights, init_biases):
        """ Initializes the fully-connected layer with a matrix of
            initial weights and a vector of initial biases.

        Arguments:
            init_weights
                matrix of nb_inputs by nb_outputs initial weights.
            init_biases
                vector of nb_outputs initial biases.
        """
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        assert init_weights.shape[1] == init_biases.shape[0]
        self.nb_inputs, self.nb_outputs = init_weights.shape
        self.weights = theano.shared(init_weights)
        self.biases = theano.shared(init_biases)

    def forward_pass(self, input_matrix):
        return T.maximum(T.dot(input_matrix, self.weights) + self.biases, 0)

    def parameters(self):
        return [self.weights, self.biases]

    def output_shape(self, input_shape):
        return [self.nb_outputs]

class ConvLayer(Layer):
    """ Convolutional layer with ReLU non-linearity.
    """
    
    def __init__(self, init_filters, init_biases):
        """ Initializes a convolutional layer with a set of initial filters and 
            biases.

        Arguments:
            init_filters
                4D numpy array of shape (nb_filters, nb_features, nb_rows, nb_cols).
            init_biases
                numpy vector of nb_filters elements for initial bias values.
            fmap_shape
                shape of the input feature map, optional. Used for internal theano
                optimizations.
        """
        assert len(init_filters.shape) == 4
        assert len(init_biases.shape) == 1
        assert init_filters.shape[0] == init_biases.shape[0]
        # Store the initial filters in a shared variable.
        self.filters = theano.shared(init_filters)
        self.biases = theano.shared(init_biases)
        self.filters_shape = init_filters.shape
        
    def forward_pass(self, fmaps):
        # Computes the raw convolution output.
        out_fmaps = T.nnet.conv.conv2d(
            fmaps,
            self.filters,
            border_mode='valid'
        )
        nb_filters = self.filters_shape[0]
        # Add biases and apply ReLU non-linearity.
        relu_fmaps = T.maximum(
            0, 
            out_fmaps +  self.biases.reshape([1, nb_filters, 1, 1])
        )
        return relu_fmaps

            
    def parameters(self):
        return [self.filters, self.biases]

    def output_shape(self, input_shape):
        return [
            self.filters_shape[0],
            input_shape[1] - self.filters_shape[2] + 1,
            input_shape[2] - self.filters_shape[3] + 1
        ]

class MaxPoolLayer(Layer):
    """ Non-overlapping max pooling layer.
    """
    def __init__(self, pooling_size):
        """ Initialize a max-pooling layer with a given pooling window size.

        Arguments:
            pooling_size
                size of the max pooling window.
        """
        self.pooling_size = pooling_size

    def forward_pass(self, fmaps):
        return max_pool_2d(fmaps, (self.pooling_size, self.pooling_size))

    def parameters(self):
        return []

    def output_shape(self, input_shape):
        roffset = 0 if input_shape[1] % self.pooling_size == 0 else 1
        coffset = 0 if input_shape[2] % self.pooling_size == 0 else 1

        return [
            input_shape[0],
            input_shape[1] // self.pooling_size + roffset,
            input_shape[2] // self.pooling_size + coffset
        ]
