""" Function for pre-processing batches in a separate process. Purposefully avoids importing
    theano to avoid problems with devices and all. Allows the program to compute the gradient
    and update the network weights in parallel to the preprocessing of the next batch.
"""
import numpy as np
import multiprocessing as mp

def batch_producer(train_set, splits, largest_batch_size, nb_classes,
                   shared_samples, shared_labels, available, condition):
    """ Producer process function to prepare batches of images for optimization.
        Writes the output to a raw array in shared memory, synchronized by a
        multiprocessing condition object.

    Arguments:
        train_set
            Training dataset. Can be any kind of dataset, but for this to make sense in
            a parallel setting, it should include most expensive pre-processing, i.e.
            loading the image from disk, decompressing the jpeg data, etc.
        splits
            array of size nb_batches + 1 where {splits[i], ..., splits[i+1]-1} are the
            indices of samples in batch i .
        out_samples
            output RawArray in shared memory where the batches will be written to.
        out_shape
            shape of output RawArray, also expected to be in shared memory. Should
            be in (batch_size, nb_channels, rows, cols) format.
        out_labels
            output RawArray in shared memory where the batch labels will be written to.
        available
            shared boolean indicating when a batch is available for consumption (true)
            or has been consumed (false).
        condition
            multiprocessing condition used for synchronizing processes.
    """
    nb_batches = splits.size - 1
    train_labels = train_set.get_labels()
    img_shape = train_set.sample_shape
    np_samples = np.ctypeslib.as_array(
        shared_samples
    )
    np_samples.shape = [largest_batch_size] + img_shape
    np_labels = np.ctypeslib.as_array(
        shared_labels
    )
    np_labels.shape = [largest_batch_size, nb_classes]
    samples_iterator = iter(train_set)
    
    for i in range(nb_batches):
        # Select the batch.
        batch_size = splits[i+1] - splits[i]
        new_batch = np.empty(
            [batch_size] + train_set.sample_shape,
            np.float32
        )
        new_labels = np.zeros(
            [batch_size, nb_classes],
            np.float32
        )
        cur_labels = train_labels[splits[i]:splits[i+1]]
        
        for j in range(batch_size):
            new_sample = samples_iterator.next()
            new_batch[j] = new_sample
            for label in cur_labels[j]:
                new_labels[j,label] = 1
        # Copy the batch to shared memory, synchronizing with consumer process.
        condition.acquire()
        # Wait for the previous batch to be consumed.
        while available.value != 0:
            condition.wait()
        np_samples[0:batch_size] = new_batch
        np_labels[0:batch_size] = new_labels
        available.value = 1
        condition.notify()
        condition.release()
