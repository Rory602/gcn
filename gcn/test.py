# -*- coding: utf-8 -*- 
"""
@Time : 2019/10/3 下午8:54 
@Author : Wang Zhiyuan 
@Email : wangzhiyuan@geotmt.com
@File : test.py 
"""# coding:utf-8
import tensorflow as tf
from tensorflow.python import debug as tf_debug
###############################
"""
# 一个样本
###############################
logits = tf.constant([2,7,5],dtype=tf.float32)
labels = [0,1,0]
#对logits使用softmax,[0.00589975 0.8756006  0.11849965]
res1 = tf.nn.softmax(logits)
# 交叉熵损失中的各个对数部分,[-5.1328454  -0.13284525 -2.1328452 ]
res2 = tf.log(res1)
# 交叉熵损失,0.13284527
res3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)

with tf.Session() as sess:
	res1,res2,res3 = sess.run([res1,res2,res3])
	print(res1)
	print(res2)
	print(res3)
print('====================================================')
###############################
"""
# 多个样本
###############################
logits = tf.constant([[2,7,5],[6,3,4]],dtype=tf.float32)
labels = [[0,1,0],[1,0,0]]
#对logits使用softmax,[[0.00589975 0.8756006  0.11849965] [0.8437947  0.04201007 0.11419519]]
res1 = tf.nn.softmax(logits)
# 交叉熵损失中的各个对数部分,[[-5.1328454  -0.13284525 -2.1328452 ] [-0.16984606 -3.169846   -2.169846  ]]
res2 = tf.log(res1)
# 交叉熵损失,[0.13284527 0.16984604]
res3 = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name="test")
# 求出交叉熵损失后再对各个样本的交叉熵损失取平均,0.15134566
res4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
res1,res2,res3,res4 = sess.run([res1,res2,res3,res4])
print(res1)
print(res2)
print(res3)
print(res4)
sess.close()
