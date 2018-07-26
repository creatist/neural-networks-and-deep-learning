"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        """
        得到一个 30行 1列 和 一个 10行 1列的list 数组
        即， len(self.biases) = 2, len(self.biases[0]) = 30, len(self.biases[1]) = 1
        len(self.biases[1])= 10 , len(self.biases[1][0]) = 1
        """
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        """
        得到 一个 30 行 784 列 和一个 10 行 30 列的  list 数组
        即， len(self.weights) = 2, len(self.weights[0]) = 30 len(self.weights[0][0])=784
        len(self.weights[1])=10, len(self.weights[1][0])=30
        就是说 weights[0] 表示第 1 层和第 2 层之间的所有权重，weights[0][0]是第 1 层和第 2 层第一个神经元之间的权重，依次类推.
        对于每一个神经元的输入权重都是一个784维的行向量，即长度为784的list
        公式 BP4 中的 Wjk ，j刚好对应行列式的行，k对应行列式的列
        """
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        """
        epoch 的含义：每一个epoch都要对所有数据重新训练一次。
        """
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            """
            每更新一次权重和偏移量就计算一次识别正确率
            """
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate.
        每个mini_batch 求一次权重偏导和偏移量偏导的平均值，更新一次权重和偏移量
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            """
            对每个数据进行一次前向传播和反向传播
            """
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            """
            将每个训练数据得到的偏导加在一起
            """
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
            """ 根据方程（47）——简写为：ΔC≈∂C/∂w *Δw.  其中 nw 即为∂C/∂w 。
                只要使 ΔC 为负值，则 网络的总代价C就会减小。
                下面的代码中 Δw = -(eta/len(mini_batch))*nw = -(eta/len(mini_batch))* ∂C/∂w  ~ - ∂C/∂w
                所以 ΔC≈∂C/∂w *Δw = -(eta/len(mini_batch))* ∂C/∂w * ∂C/∂w  <= 0
                根据公式（32）或者 BP4 得，∂C/∂w=ain * δout , 当 δout为0，这虽然是我们期望的，实际却达不到；
                    如果所有的ain = 0 网络的训练就没有了意义, 所以整体上 ain > 0
                    所以 ∂C/∂w 总是 大于 0
                所以 ΔC 总是 小于 0 ，以此达到减小消耗的目的。
                同样对于 Δb, ΔC≈∂C/∂b *Δb = -(eta/len(mini_batch))*nb = -(eta/len(mini_batch)) * ∂C/∂b * ∂C/∂b
                ∂C/∂b = δ > 0
                所以  ΔC≈∂C/∂b *Δb < 0
                这两个计算公式来源于 2.6节，简单说就是取 Δw = -步长 * 偏导, 
                至于为什么除以 len(mini_batch)，
                    这里的nabla_w 和 nabla_b是mini_batch中所有输入得到的权重和及偏置的和， 所以要除以len(mini_batch)求平均值
            """
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        """
        输入是 784 行 1 列 的列向量
        activations 中每一个 list数组的维数是不相同的，其中包含了多个activation,
        比如第一个activation，即输入 x ,是 784 行 1 列， 第二个 activation 是 30 行 1 列， 第三个是 10 行 1 列 
        """
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
        """
        这里 num_layers=3, 所以 xrange(2,3) 生成 [2] 
        上面求出的 delta是倒数第 1 层和倒数第 2 层间的 delta，
        下面求出的 delta是倒数第 2 层和倒数第 3 层间的 delta ，
        所以用 weights[-l+1] = weights[1] 即，第 2 层 和第 3 层，倒数第 1 层和倒数第 2 层的 weight 来求
        """
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            """
            delta 由 公式 BP2: δl=((wl+1)Tδl+1)⊙σ′(zl) 得出， 和 biases 一样是列向量
            """
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            """
            nabla_w 是一个多行多列的矩阵，delta是一个列向量，activations[i]也是一个列向量，从向量的运算法则来说需要将activations转置
            四个基本方程 BP1 ~ BP4并不都是矩阵形式，非矩阵形式的方程运算的时候要根据结果的行数列数 对乘积变量调整顺序或进行转置
            """
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
