import tensorflow as tf
import numpy as np


# ---------------------------------------------- my activations -----------------------------------------------
def mysin(x):
    return tf.sin(2*np.pi*x)


def srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)


def asrelu(x):   # abs srelu
    return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))


def s2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)
    # return 1.5*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)
    # return 1.25*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)


def s3relu(x):
    # return 0.5*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # return 0.21*tf.nn.relu(1-x)*tf.nn.relu(1+x)*tf.sin(2*np.pi*x)
    # return tf.nn.relu(1 - x) * tf.nn.relu(x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x))   # (work不好)
    # return tf.nn.relu(1 - x) * tf.nn.relu(1 + x) * (tf.sin(2 * np.pi * x) + tf.cos(2 * np.pi * x)) #（不work）
    return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*tf.abs(x))  # work 不如 s2relu
    # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(2*np.pi*x)            # work 不如 s2relu
    # return 1.5*tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.sin(np.pi*x)
    # return tf.nn.relu(1 - x) * tf.nn.relu(x+0.5) * tf.sin(2 * np.pi * x)


def csrelu(x):
    # return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.cos(np.pi*x)
    return 1.5*tf.nn.relu(1 - x) * tf.nn.relu(x) * tf.cos(np.pi * x)
    # return tf.nn.relu(1-tf.abs(x))*tf.nn.relu(tf.abs(x))*tf.cos(np.pi*x)


def stanh(x):
    # return tf.tanh(x)*tf.sin(2*np.pi*x)
    return tf.sin(2*np.pi*tf.tanh(x))


def gauss(x):
    # return 0.2*tf.exp(-4*x*x)
    # return 0.25*tf.exp(-4 * x * x)
    return 0.75 * tf.exp(-2 * x * x)
    # return 0.25*tf.exp(-7.5*(x-0.5)*(x-0.5))


def mexican(x):
    return (1-x*x)*tf.exp(-0.5*x*x)


def modify_mexican(x):
    # return 1.25*x*tf.exp(-0.25*x*x)
    # return x * tf.exp(-0.125 * x * x)
    return x * tf.exp(-0.075*x * x)
    # return -1.25*x*tf.exp(-0.25*x*x)


def sm_mexican(x):
    # return tf.sin(np.pi*x) * x * tf.exp(-0.075*x * x)
    # return tf.sin(np.pi*x) * x * tf.exp(-0.125*x * x)
    return 2.0*tf.sin(np.pi*x) * x * tf.exp(-0.5*x * x)


def singauss(x):
    # return 0.6 * tf.exp(-4 * x * x) * tf.sin(np.pi * x)
    # return 0.6 * tf.exp(-5 * x * x) * tf.sin(np.pi * x)
    # return 0.75*tf.exp(-5*x*x)*tf.sin(2*np.pi*x)
    # return tf.exp(-(x-0.5) * (x - 0.5)) * tf.sin(np.pi * x)
    # return 0.25 * tf.exp(-3.5 * x * x) * tf.sin(2 * np.pi * x)
    # return 0.225*tf.exp(-2.5 * (x - 0.5) * (x - 0.5)) * tf.sin(2*np.pi * x)
    return 0.225 * tf.exp(-2 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
    # return 0.4 * tf.exp(-10 * (x - 0.5) * (x - 0.5)) * tf.sin(2 * np.pi * x)
    # return 0.45 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(np.pi * x)
    # return 0.3 * tf.exp(-5 * (x - 1.0) * (x - 1.0)) * tf.sin(2 * np.pi * x)
    # return tf.sin(2*np.pi*tf.exp(-0.5*x*x))


def powsin_srelu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(2*np.pi*x)*tf.sin(2*np.pi*x)


def sin2_srelu(x):
    return 2.0*tf.nn.relu(1-x)*tf.nn.relu(x)*tf.sin(4*np.pi*x)*tf.sin(2*np.pi*x)


def slrelu(x):
    return tf.nn.leaky_relu(1-x)*tf.nn.leaky_relu(x)


def pow2relu(x):
    return tf.nn.relu(1-x)*tf.nn.relu(x)*tf.nn.relu(x)


def selu(x):
    return tf.nn.elu(1-x)*tf.nn.elu(x)


def wave(x):
    return tf.nn.relu(x) - 2*tf.nn.relu(x-1/4) + \
           2*tf.nn.relu(x-3/4) - tf.nn.relu(x-1)


def phi(x):
    return tf.nn.relu(x) * tf.nn.relu(x)-3*tf.nn.relu(x-1)*tf.nn.relu(x-1) + 3*tf.nn.relu(x-2)*tf.nn.relu(x-2) \
           - tf.nn.relu(x-3)*tf.nn.relu(x-3)*tf.nn.relu(x-3)


#  ------------------------------------------------  初始化权重和偏置 --------------------------------------------
# 生成DNN的权重和偏置
# tf.random_normal(): 用于从服从指定正太分布的数值中取出随机数
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# hape: 输出张量的形状，必选.--- mean: 正态分布的均值，默认为0.----stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32 ----seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样---name: 操作的名称
def Initial_DNN2different_hidden(in_size, out_size, hidden_layers, Flag):
    n_hiddens = len(hidden_layers)
    Weights = []  # 权重列表，用于存储隐藏层的权重
    Biases = []  # 偏置列表，用于存储隐藏层的偏置
    # 隐藏层：第一层的权重和偏置，对输入数据做变换
    W = tf.Variable(0.1 * tf.random.normal([in_size, hidden_layers[0]]), dtype='float32',
                    name='W_transInput' + str(Flag))
    B = tf.Variable(0.1 * tf.random.uniform([1, hidden_layers[0]]), dtype='float32',
                    name='B_transInput' + str(Flag))
    Weights.append(W)
    Biases.append(B)
    # 隐藏层：第二至倒数第二层的权重和偏置
    for i_layer in range(n_hiddens - 1):
        W = tf.Variable(0.1 * tf.random.normal([hidden_layers[i_layer], hidden_layers[i_layer+1]]), dtype='float32',
                        name='W_hidden' + str(i_layer + 1) + str(Flag))
        B = tf.Variable(0.1 * tf.random.uniform([1, hidden_layers[i_layer+1]]), dtype='float32',
                        name='B_hidden' + str(i_layer + 1) + str(Flag))
        Weights.append(W)
        Biases.append(B)

    # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
    W = tf.Variable(0.1 * tf.random.normal([hidden_layers[-1], out_size]), dtype='float32',
                    name='W_outTrans' + str(Flag))
    B = tf.Variable(0.1 * tf.random.uniform([1, out_size]), dtype='float32',
                    name='B_outTrans' + str(Flag))
    Weights.append(W)
    Biases.append(B)

    return Weights, Biases


# tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，
# 均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，
# 那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
# truncated_normal(
#     shape,
#     mean=0.0,
#     stddev=1.0,
#     dtype=tf.float32,
#     seed=None,
#     name=None)
def truncated_normal_init(in_dim, out_dim, scale_coef=1.0, weight_name='weight'):
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.Variable(scale_coef*tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32, name=weight_name)
    return V


# tf.random_uniform()
# 默认是在 0 到 1 之间产生随机数，也可以通过 minval 和 maxval 指定上下界
def uniform_init(in_dim, out_dim, weight_name='weight'):
    V = tf.Variable(tf.random_uniform([in_dim, out_dim], dtype=tf.float32), dtype=tf.float32, name=weight_name)
    return V


# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# 从正态分布中输出随机值。
# 参数:
#     shape: 一维的张量，也是输出的张量。
#     mean: 正态分布的均值。
#     stddev: 正态分布的标准差。
#     dtype: 输出的类型。
#     seed: 一个整数，当设置之后，每次生成的随机数都一样。
#     name: 操作的名字。
def normal_init(in_dim, out_dim, scale_coef=1.0, weight_name='weight'):
    stddev2normal = np.sqrt(2.0/(in_dim + out_dim))
    # 尺度因子防止初始化的数值太小或者太大
    V = tf.Variable(scale_coef*tf.random_normal([in_dim, out_dim], mean=0, stddev=stddev2normal, dtype=tf.float32),
                    dtype=tf.float32, name=weight_name)
    return V


# tf.zeros(
#     shape,
#     dtype=tf.float32,
#     name=None
# )
# shape代表形状，也就是1纬的还是2纬的还是n纬的数组
def zeros_init(in_dim, out_dim, weight_name='weight'):
    V = tf.Variable(tf.zeros([in_dim, out_dim], dtype=tf.float32), dtype=tf.float32, name=weight_name)
    return V


def initialize_NN_xavier(in_size, out_size, hidden_layers, Flag):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        scale = 5.0
        n_hiddens = len(hidden_layers)
        Weights = []                  # 权重列表，用于存储隐藏层的权重
        Biases = []                   # 偏置列表，用于存储隐藏层的偏置

        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        W = truncated_normal_init(in_size, hidden_layers[0], scale_coef=scale, weight_name='W-transInput' + str(Flag))
        B = uniform_init(1, hidden_layers[0], weight_name='B-transInput' + str(Flag))
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            W = truncated_normal_init(hidden_layers[i_layer], hidden_layers[i_layer + 1], scale_coef=scale,
                                      weight_name='W-hidden' + str(i_layer + 1) + str(Flag))
            B = uniform_init(1, hidden_layers[i_layer + 1], weight_name='B-hidden' + str(i_layer + 1) + str(Flag))
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        W = truncated_normal_init(hidden_layers[-1], out_size, scale_coef=scale, weight_name='W-outTrans' + str(Flag))
        B = uniform_init(1, out_size, weight_name='B-outTrans' + str(Flag))
        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def initialize_NN_random_normal(in_size, out_size, hidden_layers, Flag, varcoe=0.5):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag), shape=(1, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.get_variable(
                name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            B = tf.get_variable(
                name='B' + str(i_layer + 1) + str(Flag), shape=(1, hidden_layers[i_layer + 1]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(
            name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.get_variable(
            name='B-outTrans' + str(Flag), shape=(1, out_size),
            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def initialize_NN_random_normal2(in_size, out_size, hidden_layers, Flag, varcoe=0.5):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)
        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            W = tf.get_variable(
                name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def initialize_NN_random_normal2_CS(in_size, out_size, hidden_layers, Flag, varcoe=0.5):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        n_hiddens = len(hidden_layers)
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)

        for i_layer in range(0, n_hiddens - 1):
            stddev_WB = (2.0 / (hidden_layers[i_layer] + hidden_layers[i_layer + 1])) ** varcoe
            if 0 == i_layer:
                W = tf.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer]*2, hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            else:
                W = tf.get_variable(
                    name='W' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer], hidden_layers[i_layer + 1]),
                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
                B = tf.get_variable(name='B' + str(i_layer + 1) + str(Flag), shape=(hidden_layers[i_layer + 1],),
                                    initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
            Weights.append(W)
            Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


def initialize_NN_random_normal2RNN(in_size, out_size, hidden_layers, Flag, varcoe=0.5):
    with tf.variable_scope('WB_scope', reuse=tf.AUTO_REUSE):
        Weights = []  # 权重列表，用于存储隐藏层的权重
        Biases = []  # 偏置列表，用于存储隐藏层的偏置
        # 隐藏层：第一层的权重和偏置，对输入数据做变换
        stddev_WB = (2.0 / (in_size + hidden_layers[0])) ** varcoe
        W = tf.get_variable(name='W-transInput' + str(Flag), shape=(in_size, hidden_layers[0]),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        B = tf.get_variable(name='B-transInput' + str(Flag), shape=(hidden_layers[0],),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB),
                            dtype=tf.float32)
        Weights.append(W)
        Biases.append(B)

        # 输出层：最后一层的权重和偏置。将最后的结果变换到输出维度
        stddev_WB = (2.0 / (hidden_layers[-1] + out_size)) ** varcoe
        W = tf.get_variable(name='W-outTrans' + str(Flag), shape=(hidden_layers[-1], out_size),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)
        B = tf.get_variable(name='B-outTrans' + str(Flag), shape=(out_size,),
                            initializer=tf.random_normal_initializer(stddev=stddev_WB), dtype=tf.float32)

        Weights.append(W)
        Biases.append(B)
        return Weights, Biases


# ----------------------------------- 正则化 -----------------------------------------------
def regular_weights_biases_L1(weights, biases):
    # L1正则化权重和偏置
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.abs(weights[i_layer1]), keep_dims=False)
        regular_b = regular_b + tf.reduce_sum(tf.abs(biases[i_layer1]), keep_dims=False)
    return regular_w + regular_b


# L2正则化权重和偏置
def regular_weights_biases_L2(weights, biases):
    layers = len(weights)
    regular_w = 0
    regular_b = 0
    for i_layer1 in range(layers):
        regular_w = regular_w + tf.reduce_sum(tf.square(weights[i_layer1]), keep_dims=False)
        regular_b = regular_b + tf.reduce_sum(tf.square(biases[i_layer1]), keep_dims=False)
    return regular_w + regular_b


# -------------------- DNN 网络模型 -------------------------------------
def normal_DNN(variable_input, Weights=None, Biases=None, hiddens=None, activate_name=None):
    if activate_name == 'relu':
        DNN_activation = tf.nn.relu
    elif activate_name == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu(0.2)
    elif activate_name == 'elu':
        DNN_activation = tf.nn.elu
    elif activate_name == 'tanh':
        DNN_activation = tf.nn.tanh
    elif activate_name == 'sin':
        DNN_activation = mysin
    elif activate_name == 'srelu':
        DNN_activation = srelu
    elif activate_name == 'powsin_srelu':
        DNN_activation = powsin_srelu
    elif activate_name == 's2relu':
        DNN_activation = s2relu
    elif activate_name == 'sin2_srelu':
        DNN_activation = sin2_srelu
    elif activate_name == 'slrelu':
        DNN_activation = slrelu
    elif activate_name == 'selu':
        DNN_activation = selu
    elif activate_name == 'phi':
        DNN_activation = phi

    layers = len(Weights)                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    hidden_record = 0
    for k in range(layers-1):
        H_pre = H
        W = Weights[k]
        B = Biases[k]
        H = DNN_activation(tf.add(tf.matmul(H, W), B))
        if hiddens[k] == hidden_record:
            H = H+H_pre
        hidden_record = hiddens[k]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    # output = tf.nn.tanh(output)
    return output


# Cos_Sin 代表 cos concatenate sin according to row（i.e. the number of sampling points）
def DNN_Cos_Sin_Base(variable_input, Weights=None, Biases=None, hiddens=None, activate_name=None):
    if str.lower(activate_name) == 'relu':
        DNN_activation = tf.nn.relu
    elif str.lower(activate_name) == 'leaky_relu':
        DNN_activation = tf.nn.leaky_relu
    elif str.lower(activate_name) == 'srelu':
        DNN_activation = srelu
    elif str.lower(activate_name) == 's2relu':
        DNN_activation = s2relu
    elif str.lower(activate_name) == 's3relu':
        DNN_activation = s3relu
    elif str.lower(activate_name) == 'csrelu':
        DNN_activation = csrelu
    elif str.lower(activate_name) == 'sin2_srelu':
        DNN_activation = sin2_srelu
    elif str.lower(activate_name) == 'powsin_srelu':
        DNN_activation = powsin_srelu
    elif str.lower(activate_name) == 'slrelu':
        DNN_activation = slrelu
    elif str.lower(activate_name) == 'elu':
        DNN_activation = tf.nn.elu
    elif str.lower(activate_name) == 'selu':
        DNN_activation = selu
    elif str.lower(activate_name) == 'sin':
        DNN_activation = mysin
    elif str.lower(activate_name) == 'tanh':
        DNN_activation = tf.nn.tanh
    elif str.lower(activate_name) == 'sintanh':
        DNN_activation = stanh
    elif str.lower(activate_name) == 'gauss':
        DNN_activation = gauss
    elif str.lower(activate_name) == 'singauss':
        DNN_activation = singauss
    elif str.lower(activate_name) == 'mexican':
        DNN_activation = mexican
    elif str.lower(activate_name) == 'modify_mexican':
        DNN_activation = modify_mexican
    elif str.lower(activate_name) == 'sin_modify_mexican':
        DNN_activation = sm_mexican
    elif str.lower(activate_name) == 'phi':
        DNN_activation = phi

    layers = len(hiddens) + 1                   # 得到输入到输出的层数，即隐藏层层数
    H = variable_input                      # 代表输入数据，即输入层
    W_in = Weights[0]
    H = tf.matmul(H, W_in)
    # H = tf.concat([tf.cos(H), tf.sin(H)], axis=1)
    H = 0.5 * (tf.concat([tf.cos(H), tf.sin(H)], axis=1))  # 这个效果好
    # H = 0.75*(tf.concat([tf.cos(H), tf.sin(H)], axis=1))
    # H = 0.5*(tf.concat([tf.cos(np.pi * H), tf.sin(np.pi * H)], axis=1))
    # H = tf.concat([tf.cos(2 * np.pi * H), tf.sin(2 * np.pi * H)], axis=1)

    hiddens_record = hiddens[0]
    for k in range(layers-2):
        H_pre = H
        W = Weights[k+1]
        B = Biases[k+1]
        W_shape = W.get_shape().as_list()
        H = DNN_activation(tf.add(tf.matmul(H, W), B))
        if (hiddens[k+1] == hiddens_record) and (W_shape[0] == hiddens_record):
            H = H + H_pre
        hiddens_record = hiddens[k+1]

    W_out = Weights[-1]
    B_out = Biases[-1]
    output = tf.add(tf.matmul(H, W_out), B_out)
    # 下面这个是输出层
    output = tf.nn.tanh(output)
    return output


# -----------------------  RNN 模型 ---------------------------------------
# TensorFlow中实现RNN的基本单元，每个RNNCell都有一个call方法，使用方式是：(output, next_state) = call(input, state)。
# 也就是说，每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”，这就是RNNCell的基本功能。
# 在BasicRNNCell中，state_size永远等于output_size
# 除了call方法外，对于RNNCell，还有两个类属性比较重要：state_size 和 output_size
# 前者是隐层的大小，后者是输出的大小。比如我们通常是将一个batch送入模型计算，设输入数据的形状为(batch_size, input_size)，
# 那么计算时得到的隐层状态就是(batch_size, state_size)，输出就是(batch_size, output_size)。

# x_inputs: 是进入single RNN的变量，形状为(batch_size, in_size)
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
    elif act2name == 'sin':
        RNN_activation = mysin
    elif act2name == 'srelu':
        RNN_activation = srelu
    elif act2name == 'powsin_srelu':
        RNN_activation = powsin_srelu
    elif act2name == 's2relu':
        RNN_activation = s2relu
    elif act2name == 'sin2_srelu':
        RNN_activation = sin2_srelu
    elif act2name == 'slrelu':
        RNN_activation = slrelu
    elif act2name == 'selu':
        RNN_activation = selu
    elif act2name == 'phi':
        RNN_activation = phi
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
def multi_rnn(x_inputs, h_init=None, units2hiddens=None, Ws=None, Bs=None, act2name='sigmoid',
              opt2RNN_cell='rnn', scope2rnn=None):
    if act2name == 'relu':
        RNN_activation = tf.nn.relu
    elif act2name == 'leaky_relu':
        RNN_activation = tf.nn.leaky_relu(0.2)
    elif act2name == 'elu':
        RNN_activation = tf.nn.elu
    elif act2name == 'tanh':
        RNN_activation = tf.nn.tanh
    elif act2name == 'sin':
        RNN_activation = mysin
    elif act2name == 'srelu':
        RNN_activation = srelu
    elif act2name == 'powsin_srelu':
        RNN_activation = powsin_srelu
    elif act2name == 's2relu':
        RNN_activation = s2relu
    elif act2name == 'sin2_srelu':
        RNN_activation = sin2_srelu
    elif act2name == 'slrelu':
        RNN_activation = slrelu
    elif act2name == 'selu':
        RNN_activation = selu
    elif act2name == 'phi':
        RNN_activation = phi
    elif act2name == 'sigmod':
        RNN_activation = tf.nn.sigmoid

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
        in2rnn = RNN_activation(in2rnn)

        layers2rnn = len(units2hiddens)
        keep_prob = 0.5
        rnn_cells = []  # 包含所有层的列表
        for i in range(layers2rnn):
            # 构建一个基本rnn单元(一层)
            if 'rnn' == str.lower(opt2RNN_cell):
                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=units2hiddens[i], activation=RNN_activation)
            elif 'lstm' == str.lower(opt2RNN_cell):
                rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=units2hiddens[i], activation=RNN_activation, forget_bias=0.5)
            elif 'gru' == str.lower(opt2RNN_cell):
                rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=units2hiddens[i], activation=RNN_activation)
            # 可以添加dropout
            drop_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
            rnn_cells.append(drop_cell)

        # 堆叠多个LSTM单元
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)

        # tf.nn.dynamic_rnn() 要求传入的数据集的维度是三维（batch_size, squence_length, num_features)
        # tf.nn.dynamic_rnn的返回值有两个：outputs和state
        # 函数原型如下:
        # tf.nn.dynamic_rnn
        # dynamic_rnn(
        # cell,
        # inputs,
        # sequence_length=None,
        # initial_state=None,
        # dtype=None,
        # parallel_iterations=None,
        # swap_memory=False,
        # time_major=False,
        # scope=None
        # )
        # 重要参数：
        # cell:需要传入tf.nn.rnn_cell空间下的某一类rnn的实例。在本实践中我们传入了tf.nn.rnn_cell.BasicRNNCell(hidden_num),一个包含hidden_num个隐层单元的基本RNN单元。
        # inputs:输入数据
        #
        # 如果 time_major == False (default)
        # input的形状必须为 [batch_size, sequence_length, frame_size]
        #
        # 如果 time_major == True
        # input输入的形状必须为 [sequence_length, batch_size, frame_size]
        #
        # 其中batch_size是批大小，sequence_length是每个序列的大小，而frame_size是序列里面每个分量的大小（可以是shape，例如[3,4]）
        #
        # sequence_length:序列的长度，如果指定了这个参数，那么tf会对长度不够的输入在尾部填0。
        #
        # initial_state:
        # 初始化状态 (可选)， 需要是一个[batch_size, cell.state_size]形状的tensor.一般初始化为0。
        #
        # 函数返回:
        # 一个 outputs, state的tuple:
        # 其中：
        # outputs: RNN输出tensor:
        #
        # 如果 time_major == False (default)
        # outputs的形状为：[batch_size, sequence_length, cell.output_size]
        #
        # 如果 time_major == True
        # outputs的形状： [sequence_length, batch_size, cell.output_size].
        # 其中cell是刚刚传入的第一个参数的cell.
        #
        # state: 最后的状态. 是[batch_size, cell.state_size]. 的形状。state.state是一个tensor。
        # state是最终的状态，也就是序列中最后一个cell输出的状态。一般情况下state的形状为
        # [batch_size, cell.output_size]，但当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size]，
        # 其中2也对应着LSTM中的cell state和hidden state
        # out2rnn, h_out = tf.nn.dynamic_rnn(multi_layer_cell, in2rnn, initial_state=h_init, dtype=tf.float32)
        out2rnn, h_out = tf.nn.dynamic_rnn(multi_layer_cell, in2rnn, dtype=tf.float32)

        out2rnn = tf.add(tf.matmul(out2rnn, W_out), B_out)
        out2rnn = tf.squeeze(RNN_activation(out2rnn), axis=0)
        return out2rnn, h_out


# f(x+eps) = f(x) + lambda*f'(x)*eps     lambda 控制增减量的大小，预设为一个超参数
# f'(x)*eps: 用一个DNN代替，这个部分样式只有x和eps，但是f'(x)eps 也与当前的f(x)相关，所以DNN输入应该含有 x, eps, f(x)
# 这里的DNN模型使用层层残渣连接形式
def myRNN_DNN(init_base, init_day, day_input, incre_rate=0.001, model2DNN=None,
              weigths=None, bias=None, hidden2dnn=None, act2DNN=None):
    Hout_list = []
    H_base = init_base
    day_base = init_day

    len2day = len(day_input)
    for i2day in range(len2day):
        day = day_input[i2day]
        day_incre = day - day_base
        input_v = tf.concat([day, day_incre, H_base], axis=-1)
        if model2DNN == 'normal_DNN':
            H_incre = normal_DNN(input_v, Weights=weigths, Biases=bias, hiddens=hidden2dnn, activate_name=act2DNN)
        elif model2DNN == 'DNN_Cos_Sin_Base':
            H_incre = DNN_Cos_Sin_Base(input_v, Weights=weigths, Biases=bias, hiddens=hidden2dnn, activate_name=act2DNN)
        H = H_base + incre_rate * H_incre
        Hout_list.append(H)

        H_base = H
        day_base = day

    concat_outlist = tf.concat(Hout_list, axis=-1)
    return concat_outlist


# f(x+eps) = f(x) + lambda*f'(x)*eps     lambda 控制增减量的大小，预设为一个超参数
# f'(x)*eps: 用一个DNN代替，这个部分样式只有x和eps，但是f'(x)eps 也与当前的f(x)相关，所以DNN输入应该含有 x, eps, f(x)
# 这里的DNN模型使用层层残渣连接形式
def myRNN_DNN0(day_input, size2day=10, incre_rate=0.001, model2DNN=None, weigths=None, bias=None, hidden2dnn=None,
               act2DNN=None):
    day_base = tf.reshape(day_input[0], shape=[1, -1])
    H_base = tf.zeros_like(day_base)

    Hout_list = []
    day = tf.reshape(day_input[0], shape=[1, -1])
    day_incre = day - day_base
    input_v = tf.concat([day, day_incre, H_base], axis=-1)
    if model2DNN == 'normal_DNN':
        H_base = normal_DNN(input_v, Weights=weigths, Biases=bias, hiddens=hidden2dnn, activate_name=act2DNN)
    elif model2DNN == 'DNN_Cos_Sin_Base':
        H_base = DNN_Cos_Sin_Base(input_v, Weights=weigths, Biases=bias, hiddens=hidden2dnn, activate_name=act2DNN)

    Hout_list.append(H_base)

    len2day = size2day-1
    for i2day in range(len2day):
        day = tf.reshape(day_input[i2day+1], shape=[1, -1])
        day_incre = day - day_base
        input_v = tf.concat([day, day_incre, H_base], axis=-1)
        if model2DNN == 'normal_DNN':
            H_incre = normal_DNN(input_v, Weights=weigths, Biases=bias, hiddens=hidden2dnn, activate_name=act2DNN)
        elif model2DNN == 'DNN_Cos_Sin_Base':
            H_incre = DNN_Cos_Sin_Base(input_v, Weights=weigths, Biases=bias, hiddens=hidden2dnn, activate_name=act2DNN)
        H = H_base + incre_rate * H_incre
        Hout_list.append(H)

        H_base = H
        day_base = day

    concat_outlist = tf.concat(Hout_list, axis=0)
    return concat_outlist
