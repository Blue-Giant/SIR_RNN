import tensorflow as tf
import numpy as np
import RNN_data

'''
    TensorFlow中的RNN的API主要包括以下两个路径:
        1) tf.nn.rnn_cell(主要定义RNN的几种常见的cell)
        2) tf.nn(RNN中的辅助操作)
'''
# 一 RNN中的cell
# 基类(最顶级的父类): tf.nn.rnn_cell.RNNCell()
# 最基础的RNN的实现: tf.nn.rnn_cell.BasicRNNCell()
# 简单的LSTM cell实现: tf.nn.rnn_cell.BasicLSTMCell()
# 最常用的LSTM实现: tf.nn.rnn_cell.LSTMCell()
# RGU cell实现: tf.nn.rnn_cell.GRUCell()
# 多层RNN结构网络的实现: tf.nn.rnn_cell.MultiRNNCell()

# -----------------------  RNN 模型 ---------------------------------------
# TensorFlow中实现RNN的基本单元，每个RNNCell都有一个call方法，使用方式是：(output, next_state) = call(input, state)。
# 也就是说，每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”，这就是RNNCell的基本功能。
# 在BasicRNNCell中，state_size永远等于output_size
# 除了call方法外，对于RNNCell，还有两个类属性比较重要：state_size 和 output_size
# 前者是隐层的大小，后者是输出的大小。比如我们通常是将一个batch送入模型计算，设输入数据的形状为(batch_size, input_size)，
# 那么计算时得到的隐层状态就是(batch_size, state_size)，输出就是(batch_size, output_size)。


# x_inputs: 是进入RNN的变量，形状为(batch_size, in_size)
# h_init: 是进入rnn的隐藏状态，形状为(batch_size, num_units)
# num2hidden: RNN隐藏节点的个数
# output = new_state = act(W * input + U * h_state + B)
# tensorflow 1.14 不用调用rnncell.call()，直接热rnncell()即可.
# 不然会出现错误 'BasicLSTMCell' object has no attribute '_kernel'
def single_rnn(x_inputs, h_init=0.0, num2hidden=100, W_in=None, B_in=None, W_out=None, B_out=None,
               act2name='sigmoid', opt2RNN_cell='rnn', scope2rnn=None):
    if act2name == 'relu':
        RNN_activation = tf.nn.relu
    elif act2name == 'leaky_relu':
        RNN_activation = tf.nn.leaky_relu(0.2)
    elif act2name == 'elu':
        RNN_activation = tf.nn.elu
    elif act2name == 'tanh':
        RNN_activation = tf.nn.tanh
    elif act2name == 'sigmod':
        RNN_activation = tf.nn.sigmoid

    with tf.variable_scope(scope2rnn, reuse=tf.AUTO_REUSE):
        shape2W_in = W_in.get_shape().as_list()
        shape2W_out = W_out.get_shape().as_list()
        assert shape2W_in[-1] == num2hidden
        assert shape2W_out[0] == num2hidden
        in2rnn = tf.add(tf.matmul(x_inputs, W_in), B_in)
        in2rnn = RNN_activation(in2rnn)
        if 'rnn' == str.lower(opt2RNN_cell):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num2hidden, activation=RNN_activation)
        elif 'lstm' == str.lower(opt2RNN_cell):
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=num2hidden, activation=RNN_activation, forget_bias=0.5)
        elif 'gru' == str.lower(opt2RNN_cell):
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=num2hidden, activation=RNN_activation)
        out2rnn, h_out = rnn_cell(in2rnn, h_init)
        out2rnn = tf.add(tf.matmul(out2rnn, W_out), B_out)
        out2rnn = RNN_activation(out2rnn)
        return out2rnn, h_out


#  num2hiddens: 隐藏单元的数目列表，记录了每一层的隐节点个数(可以不一样)
# x_inputs: 是进入RNN的变量，形状为(batch_size, in_size)
# h_init: 是进入rnn的隐藏状态，形状为(batch_size, num_units)
# output = new_state = act(W * input + U * h_state + B)
def multi_rnn(x_inputs, h_init=None, units2hiddens=None, Ws=None, Bs=None, act2name=tf.nn.sigmoid,
              opt2RNN_cell='rnn', scope2rnn=None):
    with tf.variable_scope(scope2rnn, reuse=tf.AUTO_REUSE):
        W_in = Ws[0]
        B_in = Bs[0]
        W_out = Ws[-1]
        B_out = Bs[-1]
        shape2W_in = W_in.get_shape().as_list()
        shape2W_out = W_out.get_shape().as_list()
        assert units2hiddens[0] == shape2W_in[-1]
        assert units2hiddens[-1] == shape2W_out[0]

        shape2x = x_inputs.get_shape().as_list()
        if 2 == len(shape2x):
            x_inputs = tf.expand_dims(x_inputs, axis=0)
        in2rnn = tf.add(tf.matmul(x_inputs, W_in), B_in)
        in2rnn = act2name(in2rnn)

        layers2rnn = len(units2hiddens)
        keep_prob = 0.5
        rnn_cells = []                # 包含所有层的列表
        for i in range(layers2rnn):
            # 构建一个基本rnn单元(一层)
            if 'rnn' == str.lower(opt2RNN_cell):
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=units2hiddens[i], activation=act2name)
            elif 'lstm' == str.lower(opt2RNN_cell):
                rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=units2hiddens[i], activation=act2name, forget_bias=0.5)
            elif 'gru' == str.lower(opt2RNN_cell):
                rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=units2hiddens[i], activation=act2name)
            # 可以添加dropout
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
            rnn_cells.append(drop_cell)

        # 堆叠多个LSTM单元
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)

        # tf.nn.dynamic_rnn() 要求传入的数据集的维度是三维（batch_size, squence_length, num_features)
        # tf.nn.dynamic_rnn的返回值有两个：outputs和state
        # outputs.outputs是一个tensor.如果time_major == True，outputs形状为[max_time, batch_size, cell.output_size]
        # （要求rnn输入与rnn输出形状保持一致）如果time_major == False（默认），outputs形状为[batch_size, max_time, cell.output_size]
        # state.state是一个tensor。state是最终的状态，也就是序列中最后一个cell输出的状态。一般情况下state的形状为
        # [batch_size, cell.output_size]，但当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size]，
        # 其中2也对应着LSTM中的cell state和hidden state
        out2rnn, h_out = tf.nn.dynamic_rnn(multi_layer_cell, in2rnn, dtype=tf.float32)

        out2rnn = tf.add(tf.matmul(out2rnn, W_out), B_out)
        out2rnn = tf.squeeze(act2name(out2rnn), axis=0)
        return out2rnn, h_out


def test():
    # 初始化权和偏置
    W_in = tf.Variable(tf.truncated_normal([1, 128], stddev=0.1))
    B_in = tf.Variable(tf.constant(0., shape=[128]))

    W_out = tf.Variable(tf.truncated_normal([1, 128], stddev=0.1))
    B_out = tf.Variable(tf.constant(0., shape=[128]))

    # 定义LSTM cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)

    # shape=[4, 64]表示每次输入4个样本, 每个样本有64个特征
    x_inputs = tf.placeholder(tf.float32, shape=[1, 1])

    # 给定初始状态
    # initial_state = lstm_cell.zero_state(batch_size, dtype=)
    # 返回[batch_size, 2*len(cells)],或者[batch_size, num_units] 这个函数只是用来生成初始化值的
    init2rnn = lstm_cell.zero_state(1, dtype=tf.float32)

    H = tf.add(tf.matmul(x_inputs, W_in), B_in)
    # 对于t=1时刻传入输入和state0,获取结果值
    output, H1 = lstm_cell(H, init2rnn)
    print(output.get_shape())
    print(H1.h.get_shape())
    print(H1.c.get_shape())


def test_sRNN():
    cell_opt = 'lstm'
    batch_size = 2
    in_dim = 1
    out_im = 1
    hidden_unit = 128

    # 初始化权和偏置
    w_in = tf.Variable(tf.truncated_normal([in_dim, hidden_unit], stddev=0.1))
    b_in = tf.Variable(tf.constant(0., shape=[hidden_unit]))

    w_out = tf.Variable(tf.truncated_normal([hidden_unit, out_im], stddev=0.1))
    b_out = tf.Variable(tf.constant(0., shape=[out_im]))
    with tf.variable_scope('srnn_scope', reuse=tf.AUTO_REUSE):
        X_it = tf.placeholder(tf.float32, name='X_it', shape=[batch_size, in_dim])
        tf_zero = tf.zeros([batch_size, hidden_unit])
        output, H1 = single_rnn(X_it, h_init=tf_zero, num2hidden=hidden_unit, W_in=w_in, B_in=b_in, W_out=w_out,
                                B_out=b_out, opt2RNN_cell=cell_opt)

    print(output.get_shape())
    print(H1.get_shape())


def test_mRNN():
    cell_opt = 'lstm'
    batch_size = 10
    in_dim = 1
    out_im = 1
    hiddens2unit = (128, 64, 32, 16, 8, 4)
    W_list, B_list= [], []

    # 初始化权和偏置
    w_in = tf.Variable(tf.truncated_normal([in_dim, hiddens2unit[0]], stddev=0.1))
    b_in = tf.Variable(tf.constant(0., shape=[hiddens2unit[0]]))
    W_list.append(w_in)
    B_list.append(b_in)

    w_out = tf.Variable(tf.truncated_normal([hiddens2unit[-1], out_im], stddev=0.1))
    b_out = tf.Variable(tf.constant(0., shape=[out_im]))
    W_list.append(w_out)
    B_list.append(b_out)

    with tf.variable_scope('mrnn_scope', reuse=tf.AUTO_REUSE):
        X_it = tf.placeholder(tf.float32, name='X_it', shape=[batch_size, in_dim])
        tf_zero = tf.zeros([batch_size, hiddens2unit[0]])
        output, H1 = multi_rnn(X_it, h_init=tf_zero, units2hiddens=hiddens2unit, Ws=W_list, Bs=B_list,
                                   opt2RNN_cell=cell_opt, scope2rnn='mrnn')
        print(output.get_shape())
        for ih in range(len(H1)):
            print(H1[ih])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i_epoch in range(10):
            train_set = np.random.random([batch_size, in_dim])
            out, hout = sess.run([output, H1], feed_dict={X_it: train_set})
            print(out)
            print('\n')
            print(hout[0].c)
            print(hout[0].h)


if __name__ == "__main__":
    test_mRNN()
