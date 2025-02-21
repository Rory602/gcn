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
$$
g_{\theta^{\prime}} \star x\approx \sum_{k=0}^K\theta_k^{\prime}T_k(\tilde{L})x
$$

传统的神经网络都是输入变量乘以参数,图卷积神经网络在此基础上加入图结构信息,即$T_k(\tilde{L})$,其中K表示传播的深度.即support的个数.  



```python
class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., 		                            sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
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
        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
        # bias
        if self.bias:
            output += self.vars['bias']
        return self.act(output)

```

由` output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)`看出，只是`class Dense`只是应用欧式空间特征矩阵进行前向传播计算，没有应用非欧式图结构信息。

##### 参考文献  
Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)


