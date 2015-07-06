from layers import *


class MLP(object):
    def __init__(self, input):
        self.layers = []
        self.input = input
        self.output = input

    def add(self, layer):
        assert isinstance(layer, Layer)
        layer.input = self.output
        self.output = layer.output
        self.layers.append(layer)

        if isinstance(layer, Softmax):
            self.NLL = layer.NLL
            self.errors = layer.errors

    @property
    def params(self):
        out = []
        for l in self.layers:
            out.extend(l.params)
        return out

    def __repr__(self):
        layers = ' -> '.join([repr(l) for l in self.layers])
        return layers