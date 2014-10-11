import theano
import theano.tensor as T
import numpy

class GD:
    """ Implementation of fixed learning rate gradient descent.
    """
    def __init__(self, learning_rate, nb_iter, nb_samples, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.nb_iter = nb_iter
        self.nb_samples = nb_samples

    def optimize(self, samples, labels, cost_function, parameters, compile_mode=None):
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
        if self.verbose:
            print "Compiling cost and gradient functions..."
        # Store all the samples in a numpy array.
        samples_array = samples.to_array()
        cost = cost_function(samples_array, labels)
        # Theano function to run a full GD iteration.
        updates = []

        for param in parameters:
            updates.append(
                (param, param - self.learning_rate * T.grad(cost, param))
            )

        run_iteration = theano.function(
            [],
            [cost],
            updates=updates,
            mode=compile_mode
        )
        for t in range(1, self.nb_iter + 1):
            cost_val = run_iteration()
            if self.verbose:
                print "Epoch " + repr(t)
                print "Cost: " + repr(cost_val)
