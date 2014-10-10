import theano
import theano.tensor as T
import numpy

class Optimizer:
    """ Abstract optimizer class for optimizing cost functions over datasets.
    """

    def optimize(self, samples, labels, cost, parameters, compile_mode=None):
        """ Optimize a symbolic cost function, given symbolic variables for
            parameters to optimize for.

        Arguments:
            samples
                an arbitrary dataset.
            labels
                numpy vector of integer labels corresponding to the samples.
            cost_function
                python function taking as input two symbolic tensors (or constant numpy
                arrays) for samples and labels respectively, and returns a symbolic theano
                scalar which evaluates the cost.
            parameters
                shared theano variable for parameters to optimize the cost function for.
                Assumed to have been initialized prior to the optimization.
            compile_mode
                optional theano compilation mode, for cost-function specific optimizations.
        """
        raise NotImplementedError()

class GD:
    """ Implementation of fixed learning rate gradient descent.
    """
    def __init__(self, learning_rate, nb_iter, nb_samples, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.nb_iter = nb_iter
        self.nb_samples = nb_samples

    def optmize(self, samples, labels, cost_function, parameters, compile_mode=None):
        if self.verbose:
            print "Compiling cost and gradient functions..."
        # Store all the samples in a numpy array.
        samples_array = samples.to_array()
        cost = cost_function(samples_array, labels)
        # Theano function to run a full GD iteration.
        
        run_iteration = theano.function(
            [],
            [cost, T.grad(cost).norm(2)],
            updates=[ # Might not work, could need to update each param separately.
                (parameters, parameters - self.learning_rate * T.grad(cost, parameters))
            ],
            mode=compile_mode
        )
        for t in range(1, self.nb_iter + 1):
            cost_val, grad_norm = run_iteration()
            if self.verbose:
                print "Epoch " + repr(t)
                print "Cost: " + repr(cost_val)
                print "Gradient norm: " + repr(grad_norm)
