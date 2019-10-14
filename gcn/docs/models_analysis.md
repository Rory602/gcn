# models
```python

   
def build(self):
    """ Wrapper for _build() """
    # 构建多层layers
    with tf.variable_scope(self.name):
        self._build()

    # Build sequential layer model
    self.activations.append(self.inputs)
    for layer in self.layers:
        hidden = layer(self.activations[-1])
        self.activations.append(hidden)
    self.outputs = self.activations[-1]

    # Store model variables for easy access
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    self.vars = {var.name: var for var in variables}

    # Build metrics
    self._loss()
    self._accuracy()

    self.opt_op = self.optimizer.minimize(self.loss)
```
**解析**  
从`self.outputs = self.activations[-1]`代码看出,选取最后一层的进行为输出预测值,进行评估优化  
