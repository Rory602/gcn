import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask = tf.divide(mask, tf.reduce_mean(mask))
    loss = tf.multiply(loss, mask)
    return tf.reduce_mean(loss)


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
