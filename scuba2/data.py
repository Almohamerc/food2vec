import cPickle
import gzip
import os
import urllib
import numpy
import theano
from theano import tensor as T

__author__ = 'henryzlo'
DATA_DIR = '/home/si28b/data'


def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(
        numpy.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        numpy.asarray(data_y, dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, 'int32')


def load_mnist():
    path = os.path.join(DATA_DIR, 'mnist.pkl.gz')
    if not os.path.isfile(path):
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/' +
                           'deep/data/mnist/mnist.pkl.gz', path)
    output = []
    with gzip.open(path, 'rb') as f:
        for dataset in cPickle.load(f):
            (x, y) = shared_dataset(dataset)
            output.append((x, y))
    return output


def load_cifar10():
    path = os.path.join(DATA_DIR, 'cifar-10-python.tar.gz')
    if not os.path.isfile(path):
        urllib.urlretrieve('http://www.cs.toronto.edu/~kriz/' +
                           'cifar-10-python.tar.gz', path)

    def load_cifar_batch(path):
        with open(path, 'rb') as f:
            d = cPickle.load(f)
        data = d['data'].astype(theano.config.floatX)
        # row-normalize!
        data = data / data.max(1)[:, numpy.newaxis]
        labels = numpy.asarray(d['labels'], dtype='int32')
        return [data, labels]

    folder = os.path.join(DATA_DIR, 'cifar-10-batches-py')
    train_data = None
    train_labels = None
    for fname in os.listdir(folder):
        the_file = os.path.join(folder, fname)
        if 'data_batch' in fname:
            [data, labels] = load_cifar_batch(the_file)
            if train_data is not None:
                train_data = numpy.vstack((train_data, data))
                train_labels = numpy.append(train_labels, labels)
            else:
                train_data = data
                train_labels = labels
        elif fname == 'test_batch':
            [test_data, test_labels] = load_cifar_batch(the_file)

    test_set_x, test_set_y = shared_dataset((test_data, test_labels))
    train_set_x, train_set_y = shared_dataset((train_data, train_labels))

    return [(train_set_x, train_set_y), (test_set_x, test_set_y)]
