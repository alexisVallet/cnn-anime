import theano
import theano.tensor as T
import numpy

class Optimizer:
    """ Abstract optimizer class.
    """

    def optimize(self, cost, parameters, compile_mode=None):
        """ Optimize a symbolic cost function, given symbolic variables for
            parameters to optimize for.

        Arguments:
            cost_function
                symbolic theano scalar which evaluates the cost.
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
    def __init__(self, learning_rate, nb_iter, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.nb_iter = nb_iter

    def optmize(self, cost, parameters, compile_mode=None):
        # Theano function to run a full GD iteration.
        if self.verbose:
            print "Compiling cost and gradient functions..."
        run_iteration = theano.function(
            [],
            [cost, T.grad(cost).norm(2)],
            updates=[
                (parameters, parameters - self.learning_rate * T.grad(cost))
            ],
            mode=compile_mode
        )
        for t in range(1, self.nb_iter + 1):
            cost_val, grad_norm = run_iteration()
            if self.verbose:
                print "Epoch " + repr(t)
                print "Cost: " + repr(cost_val)
                print "Gradient norm: " + repr(grad_norm)
