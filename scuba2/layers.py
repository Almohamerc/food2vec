import numpy
import theano
import theano.tensor as T


_floatX = theano.config.floatX
_rng = numpy.random.RandomState(1234)


def set_seed(seed=12345):
    _rng = numpy.random.RandomState(seed)


class Layer(object):
    def __init__(self, function=lambda x: x):
        self.params = []
        self._input = None
        self._function = function
        self.output = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value
        self.output = self._function(self._input)


class Linear(Layer):
    # Glorot initialization
    # Sigmoid should *= 4 for W
    # Softmax should be 0s everywhere
    def __init__(self, n_in, n_out):
        sd = numpy.sqrt(6. / (n_in + n_out))
        w_values = _rng.uniform(-sd, sd, size=(n_in, n_out)).astype(_floatX)
        b_values = numpy.zeros((n_out,), dtype=_floatX)
        self.W = theano.shared(value=w_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        super(Linear, self).__init__(lambda x: T.dot(x, self.W) + self.b)
        self.params = [self.W, self.b]

    def __repr__(self):
        shape = self.W.shape.eval()
        return "[Linear (%d x %d)]" % (shape[0], shape[1])


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__(T.tanh)

    def __repr__(self):
        return "[Tanh]"


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__(T.nnet.sigmoid)

    def __repr__(self):
        return "[Sigmoid]"


class Relu(Layer):
    def __init__(self):
        super(Relu, self).__init__(lambda x: x * (x > 0.))

    def __repr__(self):
        return "[Relu]"


class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__(T.nnet.softmax)

    def NLL(self, y):
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

    @Layer.input.setter
    def input(self, value):
        Layer.input.fset(self, value)
        self.y_pred = T.argmax(self.output, axis=1)

    def __repr__(self):
        return "[Softmax]"
