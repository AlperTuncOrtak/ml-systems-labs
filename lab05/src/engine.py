import math


class Value:
    """A scalar node in a computation graph.
    Stores data, gradient, parent nodes and its local derivative rule."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._op = op
        self._backward = lambda: None   # local derivative rule

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():                 # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():                 # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant powers"
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():                 # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward():                 # 1 if a > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")
        def _backward():                 # s (1 - s)
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _backward():                 # 1 / a
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # built from the operations above, so no new derivative rules needed
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1

    def backward(self):
        # 1) topological sort: every node comes after its parents
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        # 2) reset, then set the output gradient to 1
        for v in order:
            v.grad = 0.0
        self.grad = 1.0
        # 3) visit nodes in reverse order; rules use +=, so gradients accumulate
        for v in reversed(order):
            v._backward()
        return order


def vsum(values):
    """Sum a list of Values as a graph."""
    total = Value(0.0)
    for v in values:
        total = total + v
    return total
