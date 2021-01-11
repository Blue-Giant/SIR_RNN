# tensorflow 实现递归神经网络

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# 输入图片是28*28
n_inputs = 28                                     # 输入一行，一行有28个数据
max_time = 28                                     # 一共28行
lstm_size = 200                                   # 隐层单元
n_class = 10                                      # 分类个数
batch_size = 50                                   # 每个批次样本大小
n_batch = mnist.train.num_examples // batch_size  # 批次个数

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_class], stddev=0.1))
biase = tf.Variable(tf.constant(0.1, shape=[n_class]))


# 定义RNN网络
def RNN(X, weights, biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL
    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


def LSTM(X, weights, biase):
    # inputs format : [batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本cell
    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biase)
    return results


# 返回结果
prediction = LSTM(x, weights, biase)  # RNN
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 计算准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(61):
        for batch in range(batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
