# metrics
```python

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking.先从correct_prediction中过滤出mask包含的正确答案个数,除以mask中个数"""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1),name="accuracy")
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    # 另一种计算方式如下:
    # accuracy_all_filter = tf.multiply(accuracy_all,mask)
    # debug = tf.divide(tf.reduce_mean(accuracy_all_filter),tf.reduce_mean(mask))
    mask = tf.divide(mask, tf.reduce_mean(mask, name="reduce_mean"), name="divide")
    accuracy_all = tf.multiply(accuracy_all, mask,name="multiply")
    return tf.reduce_mean(accuracy_all)
```

```python
mask = tf.divide(mask, tf.reduce_mean(mask, name="reduce_mean"), name="divide")   
accuracy_all = tf.multiply(accuracy_all, mask,name="multiply")   
```
先从correct_prediction中过滤出mask包含的正确答案个数,除以mask中个数,可以替换为以下写法:   
```python
accuracy_all_filter = tf.multiply(accuracy_all,mask)
result = tf.divide(tf.reduce_mean(accuracy_all_filter),tf.reduce_mean(mask))
```
