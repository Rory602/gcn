# layers
```python
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        return tf.nn.relu(output,name="tf_relu")
        # return self.act(output)
```
代码解析:
```python
 # convolve
supports = list()
for i in range(len(self.support)):
    if not self.featureless:
        pre_sup = dot(x, self.vars['weights_' + str(i)],
                      sparse=self.sparse_inputs)
    else:
        pre_sup = self.vars['weights_' + str(i)]
    support = dot(self.support[i], pre_sup, sparse=True)
    supports.append(support)
output = tf.add_n(supports)

```
以上代码是如下公式的实现:
<a href="https://www.codecogs.com/eqnedit.php?latex=g_{\theta^{\prime}}&space;\star&space;x&space;\approx&space;\sum_{k=0}^{K}&space;\theta_{k}^{\prime}&space;T_{k}(\tilde{L})&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_{\theta^{\prime}}&space;\star&space;x&space;\approx&space;\sum_{k=0}^{K}&space;\theta_{k}^{\prime}&space;T_{k}(\tilde{L})&space;x" title="g_{\theta^{\prime}} \star x \approx \sum_{k=0}^{K} \theta_{k}^{\prime} T_{k}(\tilde{L}) x" /></a>
参考文献  
Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)


