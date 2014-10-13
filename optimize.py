import theano
import theano.tensor as T
import numpy as np

class GD:
    """ Implementation of fixed learning rate full batch gradient descent.
    """
    def __init__(self, learning_rate, nb_iter, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.nb_iter = nb_iter

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

class SGD:
    """ Implementation of stochastic gradient descent.
    """
    def __init__(self, batch_size, init_rate, nb_epochs, learning_schedule='fixed',
                 update_rule='simple', eps=10E-5, verbose=False):
        """ Initialized the optimization method.
        
        Arguments:
            batch_size
                approximate number of samples for each mini-batch. The actual
                number will vary slightly to divide the dataset cleanly.
            init_rate
                initial learning rate. Will be updated according to the
                update rule.
            nb_epochs
                number of epochs, or number of iterations over the entire
                training set, to run before stopping.
            learning_schedule
                rule to update the learning rate. Can be:
                - 'fixed' for constant learning rate fixed to the initial rate.
                - ('decaying', decay) for learning rate decaying by a decay
                  factor for every epoch.
            update_rule
                rule to update the parameters. Can be:
                - 'simple' for simply w(t) = w(t-1) - alpha * grad(t)
                - ('momentum', mtm) for the momentum method:
                  w(t) = w(t-1) - alpha * dw(t)
                  dw(t) = mtm * dw(t-1) + grad(t)
                - ('rprop', inc_rate, dec_rate) for the rprop method,
                  should be used with large mini-batches only.
            eps
                precision for the convergence criteria.
            verbose
                True for regular printed messages.
        """
        self.batch_size = batch_size
        self.init_rate = init_rate
        self.nb_epochs = nb_epochs
        self.learning_schedule = learning_schedule
        self.update_rule = update_rule
        self.eps = eps
        self.verbose = verbose

    def optimize(self, samples, labels, cost_function, parameters, 
                 compile_mode=None):
        # Determine batches by picking the number of batch which,
        # when used to divide the number of samples, best approximates
        # the desired batch size.
        flt_nb_samples = float(len(samples))
        ideal_nb_batches = flt_nb_samples / self.batch_size
        lower_nb_batches = np.floor(ideal_nb_batches)
        upper_nb_batches = np.ceil(ideal_nb_batches)
        lower_error = abs(flt_nb_samples / lower_nb_batches - self.batch_size)
        upper_error = abs(flt_nb_samples / upper_nb_batches - self.batch_size)
        nb_batches = (int(lower_nb_batches) if lower_error < upper_error 
                      else int(upper_nb_batches))
        # Split the dataset into that number of batches in roughly equal-sized
        # batches.
        splits = np.round(np.linspace(
            0, len(samples), 
            num=nb_batches+1)
        ).astype(np.int32)
        
        # Store mini-batches into a shared variable. Since theano shared variables
        # must have constant storage space, we'll initialize to the shape of the
        # largest batch.
        largest_batch_size = 0
        for i in range(nb_batches):
            batch_size = splits[i+1] - splits[i]
            largest_batch_size = max(batch_size, largest_batch_size)
        batch = theano.shared(
            np.empty(
                [largest_batch_size] + samples.sample_shape,
                theano.config.floatX
            ),
            name='batch'
        )
        # Similarly for the batch labels.
        batch_labels = theano.shared(
            np.empty(
                [largest_batch_size],
                np.int32
            ),
            name='batch_labels'
        )

        # Compile the theano function to run a full SGD iteration.
        cost = cost_function(batch, batch_labels)
        updates = []
        
        learning_rate = theano.shared(
            np.float32(self.init_rate)
        )

        # Update rule.
        if self.update_rule == 'simple':
            for param in parameters:
                updates.append(
                    (param, param - learning_rate * T.grad(cost, param))
                )
        elif self.update_rule[0] == 'momentum':
            mtm = self.update_rule[1]
            # Keep track of the update at t-1.
            prev_updates = []
            for param in parameters:
                param_shape = param.get_value().shape
                prev_updates.append(theano.shared(
                    np.zeros(param_shape, theano.config.floatX)
                ))
            # And update the weights by taking into account a momentum
            # from t-1.
            for i in range(len(parameters)):
                cur_update = (
                    mtm * prev_updates[i] + learning_rate * 
                    T.grad(cost, parameters[i])
                )
                updates += [
                    (parameters[i], parameters[i] - cur_update),
                    (prev_updates[i], cur_update)
                ]
        else:
            raise ValueError("Invalid update rule!")

        run_iteration = theano.function(
            [],
            [cost],
            updates=updates,
            mode=compile_mode
        )

        # Run the actual iterations, shuffling the dataset at each epoch.
        for t in range(1, self.nb_epochs + 1):
            permutation = np.random.permutation(len(samples))
            samples.shuffle(permutation)
            samples_iterator = iter(samples)
            avg_cost = np.array([0], theano.config.floatX)

            for i in range(nb_batches):
                # Select the batch.
                batch_size = splits[i+1] - splits[i]
                batch_labels.set_value(labels[permutation[splits[i]:splits[i+1]]])
                new_batch = np.empty(
                    [batch_size] + samples.sample_shape,
                    theano.config.floatX
                )
                
                for j in range(batch_size):
                    new_batch[j] = samples_iterator.next()
                batch.set_value(new_batch)
                
                # Run the iteration.
                cost_val = run_iteration()
                avg_cost += cost_val
            if self.verbose:
                print "Epoch " + repr(t)
                print "Cost: " + repr(avg_cost / nb_batches)
                for i in range(len(parameters)):
                    print ("Param " + repr(i) + " mean mag " 
                           + repr(np.mean(np.abs(parameters[i].get_value()))))
