import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin='relu'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        x = [x] if isinstance(x, Value) else x
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.nonlin is None:
            return act
        elif self.nonlin == 'relu':
            return act.relu()
        elif self.nonlin == 'tanh':
            return act.tanh()
        elif self.nonlin == 'sigmoid':
            return act.sigmoid()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'Linear' if self.nonlin is None else self.nonlin}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, nonlin):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=nonlin if i < len(nouts)-1 else None) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
