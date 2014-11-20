from batch_producer import batch_producer
import multiprocessing as mp
from multiprocessing.sharedctypes import RawValue, RawArray
import theano
import theano.tensor as T
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import time
from collections import deque
import matplotlib.pyplot as plt
import cPickle as pickle

from dataset import mini_batch_split

class SGD:
    """ Implementation of stochastic gradient descent.
    """
    def __init__(self, batch_size, init_rate, nb_epochs, learning_schedule='fixed',
                 update_rule='simple', accuracy_measure='sample', pickle_schedule=None,
                 verbose=0):
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
                - 'constant' for constant learning rate fixed to the initial rate.
                - ('decay', decay_factor, patience) for 
                  learning rate decaying when validation error does not decrease 
                  after patience epochs.
            update_rule
                rule to update the parameters. Can be:
                - 'simple' for simply w(t) = w(t-1) - alpha * grad(t)
                - ('momentum', mtm) for the momentum method:
                  w(t) = w(t-1) - alpha * dw(t)
                  dw(t) = mtm * dw(t-1) + grad(t)
                - ('rmsprop', msqr_fact) for the rmsprop method:
                  w(t) = w(-1) - alpha * grad(w, t) / sqrt(msqr(w, t))
                  msqr(w, 0) = 1.
                  msqr(w, t) = msqr_fact * msqr(w, t-1) + (1 - msqr_fact) * grad(w, t)^2
            accuracy_measure
                measure of accuracy to use on the validation set:
                - 'sample' for the regular, number of samples gotten right
                  measure.
            pickle_schedule
                None for no model pickling, or a (n, name) so that every n epochs the model
                gets pickled to name_e.pkl where e is the current epoch number.
            verbose
                verbosity level: 0 for no messages, 1 for messages every epoch, 2
                for messages every iteration.
        """
        self.batch_size = batch_size
        self.init_rate = init_rate
        self.nb_epochs = nb_epochs
        self.learning_schedule = learning_schedule
        self.update_rule = update_rule
        self.accuracy_measure = accuracy_measure
        self.pickle_schedule = pickle_schedule
        self.verbose = verbose

    def optimize(self, classifier, samples, valid_data=None, compile_mode=None):
        model = classifier.model
        splits = mini_batch_split(samples, self.batch_size)
        nb_batches = splits.size - 1
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
                [largest_batch_size, model.nb_classes],
                theano.config.floatX
            ),
            name='batch_labels'
        )

        # Compile the theano function to run a full SGD iteration.
        cost = model.cost_function(batch, batch_labels)
        updates = []
        
        learning_rate = theano.shared(
            np.float32(self.init_rate)
        )

        parameters = model.parameters()

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
        elif self.update_rule[0] == 'rmsprop':
            msqr_fact, max_rate = self.update_rule[1:]
            min_norm = self.init_rate / max_rate

            # Keep track of a mean squared magnitude of gradient for each weight.
            msqr = []
            
            for param in parameters:
                param_shape = param.get_value().shape
                msqr.append(theano.shared(
                    np.ones(param_shape, theano.config.floatX)
                ))
            # Update the weights, dividing by the mean magnitude.
            for i in range(len(parameters)):
                grad = T.grad(cost, parameters[i])
                new_msqr = msqr_fact * msqr[i] + (1 - msqr_fact) * grad * grad
                # We require a minimum norm because:
                # - It effectively bounds the learning_rate, which is beneficial for
                #   the same reasons as in RPROP.
                # - It avoids numerics issues when the mean squared norm becomes too
                #   small.
                clip_mnorm = T.maximum(T.sqrt(new_msqr), min_norm)
                updates += [
                    (parameters[i], parameters[i]
                     - self.init_rate * grad * (1. / clip_mnorm)),
                     (msqr[i], new_msqr)
                ]
        else:
            raise ValueError("Invalid update rule!")
        
        run_iteration = theano.function(
            [],
            cost,
            updates=updates,
            mode=compile_mode
        )
        compute_cost = None
        prev_err = None
        prev_dec = 0
        stats = None
        since_last_pickle = 1

        # Run the actual iterations, shuffling the dataset at each epoch.
        for t in range(1, self.nb_epochs + 1):
            permutation = np.random.permutation(len(samples))
            samples.shuffle(permutation)
            samples_iterator = iter(samples)
            train_labels = samples.get_labels()
            last_100_costs = deque()
            avg_cost = 0.

            # Initialize the shared memory stuff for the batch producer/consumer.
            available = RawValue('B', 0)
            shared_samples = RawArray(
                'f',
                np.prod([largest_batch_size] + samples.sample_shape)
            )
            shared_labels = RawArray(
                'f',
                np.prod([largest_batch_size, model.nb_classes])
            )
            # Convenient numpy wrappers for these two.
            np_shared_samples = np.ctypeslib.as_array(
                shared_samples
            )
            np_shared_samples.shape = [largest_batch_size] + samples.sample_shape
            np_shared_labels = np.ctypeslib.as_array(
                shared_labels
            )
            np_shared_labels.shape = [largest_batch_size, model.nb_classes]
            condition = mp.Condition()
            producer_process = mp.Process(
                target=batch_producer,
                args=(samples, splits, largest_batch_size, model.nb_classes,
                      shared_samples, shared_labels, available, condition)
            )
            producer_process.start()
            
            for i in range(nb_batches):
                # Run the iteration.
                iter_start = time.time()
                condition.acquire()
                print "Waiting for batch..."
                while available.value != 1:
                    condition.wait()
                print "Got the batch."
                # When the batch is available, put it in the GPU.
                batch.set_value(np_shared_samples[0:batch_size])
                batch_labels.set_value(np_shared_labels[0:batch_size])
                available.value = 0
                condition.notify()
                condition.release()
                print "Running iteration..."
                cost_val = run_iteration()
                print "Finished iteration."
                iter_end = time.time()

                avg_cost += cost_val
                last_100_costs.append(cost_val)
                if len(last_100_costs) > 100:
                    last_100_costs.popleft()
                
                if self.verbose == 2:
                    print "Batch " + repr(i+1) + " out of " + repr(nb_batches)
                    print "Cost running average: " + repr(np.mean(last_100_costs))
                    print "Processed in " + repr(iter_end - iter_start) + " seconds."
            # Learning schedule.
            if self.learning_schedule == 'constant':
                pass
            elif self.learning_schedule[0] == 'decay':
                # Compute a validation error rate, decay the learning rate
                # if it didn't decrease since last epoch. We simply use the cost
                # function for that.
                if compute_cost == None:
                    compute_cost = theano.function(
                        [],
                        model.cost_function(batch, batch_labels, test=True),
                        mode=compile_mode
                    )
                decay, delay = self.learning_schedule[1:]
                # Compute prediction by splitting the validation set into mini-batches.
                tofrozenset = lambda l: l if isinstance(l, frozenset) else frozenset([l])
                valid_labels = map(tofrozenset, valid_data.get_labels())
                predicted_labels = []
                valid_splits = mini_batch_split(valid_data, self.batch_size)
                nb_valid_batches = valid_splits.size - 1
                valid_iter = iter(valid_data)
                avg_valid_cost = 0.

                for i in range(nb_valid_batches):
                    cur_batch_size = valid_splits[i+1] - valid_splits[i]
                    cur_labels = valid_labels[valid_splits[i]:valid_splits[i+1]]
                    valid_samples_batch = np.empty(
                        [cur_batch_size] + valid_data.sample_shape,
                        theano.config.floatX
                    )
                    valid_labels_batch = np.zeros(
                        [cur_batch_size, model.nb_classes],
                        theano.config.floatX
                    )
                    for j in range(cur_batch_size):
                        valid_samples_batch[j] = valid_iter.next()
                        for label in cur_labels[j]:
                            valid_labels_batch[j,label] = 1
                    batch.set_value(valid_samples_batch)
                    batch_labels.set_value(valid_labels_batch)
                    avg_valid_cost += compute_cost()
                current_err = avg_valid_cost / nb_valid_batches

                if prev_err != None and prev_err < current_err:
                    if prev_dec == delay:
                        if self.verbose >= 1:
                            print "Validation error not decreasing, decaying."
                        learning_rate.set_value(
                            np.float32(learning_rate.get_value() * decay)
                        )
                        prev_dec = 0
                    else:
                        prev_dec += 1
                else:
                    prev_dec = 0
                prev_err = current_err
            else:
                raise ValueError(repr(self.learning_schedule) 
                                 + " is not a valid learning schedule!")
            # Epoch end info messages.
            if self.verbose >= 1:
                print "Epoch " + repr(t)
                print "Training error: " + repr(avg_cost / nb_batches)
                if prev_err != None:
                    print "Validation error: " + repr(prev_err)
                if stats == None:
                    stats = {'train_err': [], 'valid_err': []}
                stats['train_err'].append(avg_cost / nb_batches)
                stats['valid_err'].append(prev_err)
                for param in parameters:
                    if param.name not in stats:
                        stats[param.name] = {
                            'mean': [],
                            'std': []
                        }
                    stats[param.name]['mean'].append(np.mean(param.get_value()))
                    stats[param.name]['std'].append(np.std(param.get_value()))
                        # Pickle the model according to the schedule.
            if self.pickle_schedule != None:
                period, name = self.pickle_schedule
                if since_last_pickle >= period:
                    if self.verbose >= 1:
                        print "Pickling current model..."
                    with open(name + repr(t) + '.pkl', 'wb') as outfile:
                        pickle.dump(
                            classifier,
                            outfile,
                            protocol=pickle.HIGHEST_PROTOCOL
                        )
                    if stats != None:
                        with open(name + "_log_" + repr(t) + '.pkl', 'wb') as outfile:
                            pickle.dump(stats, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                    since_last_pickle = 1
                else:
                    since_last_pickle += 1

