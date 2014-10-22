import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.dnn import dnn_conv
import numpy as np
from sklearn.cluster import KMeans

from classifier import ClassifierMixin

class BaseCNNClassifier:
    """ Image classifier based on a convolutional neural network.
    """
    def __init__(self, architecture, optimizer, input_shape, init='random',
                 l2_reg=0, preprocessing=[], verbose=False):
        """ Initializes a convolutional neural network with a specific
            architecture, optimization method for training and 
            initialization procedure.

        Arguments:
            architecture
                list of tuples which should be either:
                - ('conv', nb_filters, nb_rows, nb_cols, pool_r, pool_c)
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
        self.init = init
        self.l2_reg = l2_reg
        self.preprocessing = preprocessing
        self.verbose = verbose

    def train(self, dataset, valid_data):
        # Use theano's FFT convolutions.
        # Initialize the CNN with the desired method.
        if self.verbose:
            print "Initializing weights..."
        cnn = None
        if self.init == 'random':
            cnn = self.init_random()
        else:
            raise ValueError(repr(self.init) + 
                             " is not a valid initialization method.")
        # Compile the CNN prediction functions.
        if self.verbose:
            print "Compiling prediction functions..."
        test_samples = T.tensor4('test_samples')
        self._predict_proba = theano.function(
            [test_samples],
            cnn.forward_pass(test_samples)
        )
        self._predict_labels = theano.function(
            [test_samples],
            T.argmax(cnn.forward_pass(test_samples), axis=1)
        )

        # Run the preprocessing pipeline.
        if self.verbose:
            print "Preprocessing..."
        pp_dataset = dataset
        pp_valid_data = valid_data

        for preproc in self.preprocessing:
            if self.verbose:
                print "Preprocessing step " + repr(preproc) + "..."
            pp_dataset = preproc.train_data_transform(pp_dataset)
            pp_valid_data = preproc.test_data_transform(pp_valid_data)

        # Run the optimizer on the CNN cost function.
        if self.verbose:
            print "Optimizing the cost function..."
        # Apply the FFT convolution optimization. Although it is applied to the whole
        # graph, FFT convolutions are the only ones where we actually use nnet.conv2D,
        # so only those get compiled away. For other convolutions, we use the specific
        # ops.
        mode = theano.compile.get_default_mode()
        mode = mode.including('cudnn')
        self.optimizer.optimize(
            cnn,
            pp_dataset,
            pp_valid_data,
            compile_mode=mode
        )

    def predict_probas(self, images):
        # Run the preprocessing pipeline.
        pp_images = images

        for preproc in self.preprocessing:
            pp_images = preproc.test_data_transform(pp_images)
        
        return self._predict_proba(pp_images.to_array())

    def predict_labels(self, images):
        # Run the preprocessing pipeline.
        pp_images = images

        for preproc in self.preprocessing:
            pp_images = preproc.test_data_transform(pp_images)
        
        return self._predict_labels(pp_images.to_array())

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
                nb_filters, nb_rows, nb_cols, pool_r, pool_c = layer_arch[1:]
                filters = std * np.random.standard_normal(
                    [nb_filters, input_dim, nb_rows, nb_cols]
                ).astype(theano.config.floatX)
                biases = np.ones(
                    [1, nb_filters, 1, 1],
                    theano.config.floatX
                )
                layer = ConvLayer(filters, biases, (pool_r, pool_c))
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
    """ Convolutional layer with ReLU non-linearity and max-pooling.
    """
    
    def __init__(self, init_filters, init_biases, pooling=(1,1)):
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
        assert len(init_biases.shape) == 4
        assert init_filters.shape[0] == init_biases.shape[1]
        # Store the initial filters in a shared variable.
        self.filters = theano.shared(init_filters)
        self.biases = theano.shared(init_biases)
        self.filters_shape = init_filters.shape
        self.pooling = pooling
        
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
        roffset = 0 if out_shape[0] % self.pooling[0] == 0 else 1
        coffset = 0 if out_shape[1] % self.pooling[1] == 0 else 1

        return [
            self.filters_shape[0],
            out_shape[0] // self.pooling[0] + roffset,
            out_shape[1] // self.pooling[1] + coffset
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
