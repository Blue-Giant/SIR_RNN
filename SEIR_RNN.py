"""
@author: LXA
Benchmark Code of SEIR model
2020-11-13
"""
import os
import sys
import tensorflow as tf
import numpy as np
import time
import platform
import shutil
import DNN_base
import RNN_tools
import RNN_data
import plotData
import saveData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    RNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['eqs_name']), log_fileout)
    RNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    RNN_tools.log_string('activate function: %s\n' % str(R_dic['act_name']), log_fileout)
    RNN_tools.log_string('hidden layers: %s\n' % str(R_dic['hidden_layers']), log_fileout)
    RNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)
    RNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['lr_decay']), log_fileout)
    RNN_tools.log_string('The type for Loss function: %s\n' % str(R_dic['loss_function']), log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        RNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        RNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                             log_fileout)

    if R_dic['activate_stop'] != 0:
        RNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        RNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']),
                             log_fileout)

    RNN_tools.log_string(
        'Initial penalty for difference of predict and true: %s\n' % str(R_dic['init_penalty2predict_true']),
        log_fileout)

    RNN_tools.log_string('The model of regular weights and biases: %s\n' % str(R_dic['regular_weight_model']),
                         log_fileout)

    RNN_tools.log_string('Regularization parameter for weights and biases: %s\n' % str(R_dic['regular_weight']),
                         log_fileout)

    RNN_tools.log_string('Size 2 training set: %s\n' % str(R_dic['size2train']), log_fileout)

    RNN_tools.log_string('Batch-size 2 training: %s\n' % str(R_dic['batch_size2train']), log_fileout)

    RNN_tools.log_string('Batch-size 2 testing: %s\n' % str(R_dic['batch_size2test']), log_fileout)


def print_and_log2train(i_epoch, run_time, tmp_lr, temp_penalty_nt, penalty_wb2s, penalty_wb2e, penalty_wb2i, penalty_wb2r,
                        loss_s, loss_e, loss_i, loss_r, loss_n, log_out=None):
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('penalty for difference of predict and true : %f' % temp_penalty_nt)
    print('penalty weights and biases for S: %f' % penalty_wb2s)
    print('penalty weights and biases for S: %f' % penalty_wb2e)
    print('penalty weights and biases for I: %f' % penalty_wb2i)
    print('penalty weights and biases for R: %f' % penalty_wb2r)
    print('loss for S: %.10f' % loss_s)
    print('loss for E: %.10f' % loss_e)
    print('loss for I: %.10f' % loss_i)
    print('loss for R: %.10f' % loss_r)
    print('total loss: %.10f\n' % loss_n)

    RNN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    RNN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    RNN_tools.log_string('penalty for difference of predict and true : %f' % temp_penalty_nt, log_out)
    RNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2s, log_out)
    RNN_tools.log_string('penalty weights and biases for S: %f' % penalty_wb2e, log_out)
    RNN_tools.log_string('penalty weights and biases for I: %f' % penalty_wb2i, log_out)
    RNN_tools.log_string('penalty weights and biases for R: %f' % penalty_wb2r, log_out)
    RNN_tools.log_string('loss for S: %.10f' % loss_s, log_out)
    RNN_tools.log_string('loss for E: %.10f' % loss_e, log_out)
    RNN_tools.log_string('loss for I: %.10f' % loss_i, log_out)
    RNN_tools.log_string('loss for R: %.10f' % loss_r, log_out)
    RNN_tools.log_string('total loss: %.10f' % loss_n, log_out)


def solve_SEIR2COVID(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    log_fileout = open(os.path.join(log_out_path, 'log_train.txt'), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

    trainSet_szie = R['size2train']
    train_size2batch = R['batch_size2train']
    test_size2batch = R['batch_size2test']
    pt_penalty_init = R['init_penalty2predict_true']  # Regularization parameter for difference of predict and true
    wb_penalty = R['regular_weight']                  # Regularization parameter for weights
    lr_decay = R['lr_decay']
    learning_rate = R['learning_rate']
    act_func = R['act_name']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    flag2S = 'WB2S'
    flag2E = 'WB2E'
    flag2I = 'WB2I'
    flag2R = 'WB2R'
    hidden_layers = R['hidden_layers']
    Weight2S, Bias2S = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2S)
    Weight2E, Bias2E = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2E)
    Weight2I, Bias2I = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2I)
    Weight2R, Bias2R = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2R)

    flag2alpha = 'WB2alpha'
    flag2beta = 'WB2beta'
    flag2gamma = 'WB2gamma'
    Weight2alpha, Bias2alpha = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2alpha)
    Weight2beta, Bias2beta = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2beta)
    Weight2gamma, Bias2gamma = DNN_base.initialize_NN_random_normal2(input_dim, out_dim, hidden_layers, flag2gamma)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            T_it = tf.placeholder(tf.float32, name='T_it', shape=[None, out_dim])
            I_observe = tf.placeholder(tf.float32, name='I_observe', shape=[None, out_dim])
            N_observe = tf.placeholder(tf.float32, name='N_observe', shape=[None, out_dim])
            predict_true_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            # in_alpha = tf.placeholder_with_default(input=1e-5, shape=[], name='beta')
            # in_beta = tf.placeholder_with_default(input=1e-5, shape=[], name='beta')
            # in_gamma = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')
            if 'PDE_DNN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN(T_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func)
                in_alpha = DNN_base.PDE_DNN(T_it, Weight2alpha, Bias2alpha, hidden_layers, activate_name=act_func)
                in_beta = DNN_base.PDE_DNN(T_it, Weight2beta, Bias2beta, hidden_layers, activate_name=act_func)
                in_gamma = DNN_base.PDE_DNN(T_it, Weight2gamma, Bias2gamma, hidden_layers, activate_name=act_func)
            elif 'PDE_DNN_Fourier' == R['model']:
                S_NN = DNN_base.PDE_DNN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN(T_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func)
                in_alpha = DNN_base.PDE_DNN(T_it, Weight2alpha, Bias2alpha, hidden_layers, activate_name=act_func)
                in_beta = DNN_base.PDE_DNN(T_it, Weight2beta, Bias2beta, hidden_layers, activate_name=act_func)
                in_gamma = DNN_base.PDE_DNN(T_it, Weight2gamma, Bias2gamma, hidden_layers, activate_name=act_func)
            elif 'PDE_DNN_BN' == str.upper(R['model']):
                S_NN = DNN_base.PDE_DNN_BN(T_it, Weight2S, Bias2S, hidden_layers, activate_name=act_func, is_training=train_opt)
                E_NN = DNN_base.PDE_DNN_BN(T_it, Weight2E, Bias2E, hidden_layers, activate_name=act_func, is_training=train_opt)
                I_NN = DNN_base.PDE_DNN_BN(T_it, Weight2I, Bias2I, hidden_layers, activate_name=act_func, is_training=train_opt)
                R_NN = DNN_base.PDE_DNN_BN(T_it, Weight2R, Bias2R, hidden_layers, activate_name=act_func, is_training=train_opt)
                in_alpha = DNN_base.PDE_DNN_BN(T_it, Weight2alpha, Bias2alpha, hidden_layers, activate_name=act_func, is_training=train_opt)
                in_beta = DNN_base.PDE_DNN_BN(T_it, Weight2beta, Bias2beta, hidden_layers, activate_name=act_func, is_training=train_opt)
                in_gamma = DNN_base.PDE_DNN_BN(T_it, Weight2gamma, Bias2gamma, hidden_layers, activate_name=act_func, is_training=train_opt)
            elif 'PDE_DNN_SCALEOUT' == str.upper(R['model']):
                freq = np.concatenate(([1], np.arange(1, 20)*10), axis=0)
                S_NN = DNN_base.PDE_DNN_scaleOut(T_it, Weight2S, Bias2S, hidden_layers, freq, activate_name=act_func)
                E_NN = DNN_base.PDE_DNN_scaleOut(T_it, Weight2E, Bias2E, hidden_layers, freq, activate_name=act_func)
                I_NN = DNN_base.PDE_DNN_scaleOut(T_it, Weight2I, Bias2I, hidden_layers, freq, activate_name=act_func)
                R_NN = DNN_base.PDE_DNN_scaleOut(T_it, Weight2R, Bias2R, hidden_layers, freq, activate_name=act_func)
                in_alpha = DNN_base.PDE_DNN_scaleOut(T_it, Weight2alpha, Bias2alpha, hidden_layers, freq, activate_name=act_func)
                in_beta = DNN_base.PDE_DNN_scaleOut(T_it, Weight2beta, Bias2beta, hidden_layers, freq, activate_name=act_func)
                in_gamma = DNN_base.PDE_DNN_scaleOut(T_it, Weight2gamma, Bias2gamma, hidden_layers, freq, activate_name=act_func)

            alpha = tf.exp(in_alpha)
            beta = tf.exp(in_beta)
            gamma = tf.exp(in_gamma)

            S_NN = DNN_base.gauss(S_NN)
            E_NN = DNN_base.gauss(E_NN)
            I_NN = DNN_base.gauss(I_NN)
            R_NN = DNN_base.gauss(R_NN)

            N_NN = S_NN + E_NN + I_NN + R_NN

            dS_NN2t = tf.gradients(S_NN, T_it)[0]
            dE_NN2t = tf.gradients(E_NN, T_it)[0]
            dI_NN2t = tf.gradients(I_NN, T_it)[0]
            dR_NN2t = tf.gradients(R_NN, T_it)[0]
            dN_NN2t = tf.gradients(N_NN, T_it)[0]

            temp_snn2t = -(beta * tf.multiply(S_NN, I_NN))/N_NN    # / :无论多少维的tensor，都是对于最终的每个元素都除的
            temp_enn2t = beta*tf.multiply(S_NN, I_NN) - alpha * E_NN
            temp_inn2t = alpha * E_NN - gamma * I_NN
            temp_rnn2t = gamma * I_NN

            if str.lower(R['loss_function']) == 'l2_loss':
                # LossS_Net_obs = tf.reduce_mean(tf.square(S_NN - S_observe))
                # LossE_Net_obs = tf.reduce_mean(tf.square(E_NN - E_observe))
                LossI_Net_obs = tf.reduce_mean(tf.square(I_NN - I_observe))
                # LossR_Net_obs = tf.reduce_mean(tf.square(R_NN - R_observe))
                LossN_Net_obs = tf.reduce_mean(tf.square(N_NN - N_observe))

                Loss2dS = tf.reduce_mean(tf.square(dS_NN2t - temp_snn2t))
                Loss2dE = tf.reduce_mean(tf.square(dE_NN2t - temp_enn2t))
                Loss2dI = tf.reduce_mean(tf.square(dI_NN2t - temp_inn2t))
                Loss2dR = tf.reduce_mean(tf.square(dR_NN2t - temp_rnn2t))
                Loss2dN = tf.reduce_mean(tf.square(dN_NN2t))
            elif str.lower(R['loss_function']) == 'lncosh_loss':
                # LossS_Net_obs = tf.reduce_mean(tf.ln(tf.cosh(S_NN - S_observe)))
                #  LossE_Net_obs = tf.reduce_mean(tf.log(tf.cosh(E_NN - E_observe)))
                LossI_Net_obs = tf.reduce_mean(tf.log(tf.cosh(I_NN - I_observe)))
                # LossR_Net_obs = tf.reduce_mean(tf.log(tf.cosh(R_NN - R_observe)))
                LossN_Net_obs = tf.reduce_mean(tf.log(tf.cosh(N_NN - N_observe)))

                Loss2dS = tf.reduce_mean(tf.log(tf.cosh(dS_NN2t - temp_snn2t)))
                Loss2dE = tf.reduce_mean(tf.log(tf.cosh(dE_NN2t - temp_enn2t)))
                Loss2dI = tf.reduce_mean(tf.log(tf.cosh(dI_NN2t - temp_inn2t)))
                Loss2dR = tf.reduce_mean(tf.log(tf.cosh(dR_NN2t - temp_rnn2t)))
                Loss2dN = tf.reduce_mean(tf.log(tf.cosh(dN_NN2t)))

            if R['regular_weight_model'] == 'L1':
                regular_WB2S = DNN_base.regular_weights_biases_L1(Weight2S, Bias2S)
                regular_WB2E = DNN_base.regular_weights_biases_L1(Weight2E, Bias2E)
                regular_WB2I = DNN_base.regular_weights_biases_L1(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L1(Weight2R, Bias2R)
            elif R['regular_weight_model'] == 'L2':
                regular_WB2S = DNN_base.regular_weights_biases_L2(Weight2S, Bias2S)
                regular_WB2E = DNN_base.regular_weights_biases_L2(Weight2E, Bias2E)
                regular_WB2I = DNN_base.regular_weights_biases_L2(Weight2I, Bias2I)
                regular_WB2R = DNN_base.regular_weights_biases_L2(Weight2R, Bias2R)
            else:
                regular_WB2S = tf.constant(0.0)
                regular_WB2E = tf.constant(0.0)
                regular_WB2I = tf.constant(0.0)
                regular_WB2R = tf.constant(0.0)

            PWB2E = wb_penalty * regular_WB2E
            PWB2S = wb_penalty * regular_WB2S
            PWB2I = wb_penalty * regular_WB2I
            PWB2R = wb_penalty * regular_WB2R

            Loss2S = Loss2dS + PWB2S
            Loss2E = Loss2dE + PWB2E
            Loss2I = predict_true_penalty * LossI_Net_obs + Loss2dI + PWB2I
            Loss2R = Loss2dR + PWB2R
            Loss2All = LossN_Net_obs + Loss2dN

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_Loss2S = my_optimizer.minimize(Loss2S, global_step=global_steps)
            train_Loss2E = my_optimizer.minimize(Loss2E, global_step=global_steps)
            train_Loss2I = my_optimizer.minimize(Loss2I, global_step=global_steps)
            train_Loss2R = my_optimizer.minimize(Loss2R, global_step=global_steps)
            train_Loss2All = my_optimizer.minimize(Loss2All, global_step=global_steps)
            train_Loss = tf.group(train_Loss2S, train_Loss2E, train_Loss2I, train_Loss2R, train_Loss2All)

    t0 = time.time()
    loss_s_all, loss_e_all, loss_i_all, loss_r_all, loss_n_all = [], [], [], [], []
    test_epoch = []
    test_mse2I_all, test_rel2I_all = [], []

    # filename = 'data2csv/Italia_data.csv'
    filename = 'data2csv/Korea_data.csv'
    date, data = RNN_data.load_csvData(filename)
    assert (trainSet_szie + test_size2batch <= len(data))

    train_date, train_data, test_date, test_data = \
        RNN_data.split_csvData2train_test(date, data, size2train=trainSet_szie, normalFactor=R['total_population'])

    if R['total_population'] != 1:
        Have_normal = True
        NormalFactor = 1.0
    else:
        Have_normal = False
        NormalFactor = R['total_population']

    if R['total_population'] == 1:
        ndata2train = np.ones(train_size2batch, dtype=np.float32) * float(9776000)
    else:
        ndata2train = np.ones(train_size2batch, dtype=np.float32)

    # 对于时间数据来说，验证模型的合理性，要用连续的时间数据验证
    test_t_bach = RNN_data.sample_testDays_serially(test_date, test_size2batch)
    i_obs_test = RNN_data.sample_testData_serially(test_data, test_size2batch, NormalFactor)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate
        for i_epoch in range(R['max_epoch'] + 1):
            t_batch, i_obs = RNN_data.randSample_Normalize_existData(train_date, train_data, batchsize=train_size2batch,
                                                                     normalFactor=NormalFactor)
            n_obs = ndata2train.reshape(train_size2batch, 1)
            tmp_lr = tmp_lr * (1 - lr_decay)
            train_option = True
            if R['activate_stage_penalty'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_pt = 10 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_pt = 50 * pt_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_pt = 100 * pt_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_pt = 200 * pt_penalty_init
                else:
                    temp_penalty_pt = 500 * pt_penalty_init
            elif R['activate_stage_penalty'] == 2:
                if i_epoch < int(R['max_epoch'] / 3):
                    temp_penalty_pt = pt_penalty_init
                elif i_epoch < 2 * int(R['max_epoch'] / 3):
                    temp_penalty_pt = 10 * pt_penalty_init
                else:
                    temp_penalty_pt = 50 * pt_penalty_init
            else:
                temp_penalty_pt = pt_penalty_init

            _, loss_s, loss_e, loss_i, loss_r, loss_n, pwb2s, pwb2e, pwb2i, pwb2r = sess.run(
                [train_Loss, Loss2S, Loss2E, Loss2I, Loss2R, Loss2All, PWB2S, PWB2E, PWB2I, PWB2R],
                feed_dict={T_it: t_batch, I_observe: i_obs, N_observe: n_obs, in_learning_rate: tmp_lr,
                           train_opt: train_option, predict_true_penalty: temp_penalty_pt})

            loss_s_all.append(loss_s)
            loss_e_all.append(loss_e)
            loss_i_all.append(loss_i)
            loss_r_all.append(loss_r)
            loss_n_all.append(loss_n)

            if i_epoch % 1000 == 0:
                print_and_log2train(i_epoch, time.time() - t0, tmp_lr, temp_penalty_pt, pwb2s, pwb2e, pwb2i, pwb2r,
                                    loss_s, loss_e, loss_i, loss_r, loss_n, log_out=log_fileout)
                test_epoch.append(i_epoch / 1000)
                train_option = False
                s_nn2test, e_nn2test, i_nn2test, r_nn2test, alpha_test, beta_test, gamma_test = sess.run(
                    [S_NN, E_NN, I_NN, R_NN, alpha, beta, gamma], feed_dict={T_it: test_t_bach, train_opt: train_option})
                point_ERR2I = np.square(i_nn2test - i_obs_test)
                test_mse2I = np.mean(point_ERR2I)
                test_mse2I_all.append(test_mse2I)
                test_rel2I = test_mse2I / np.mean(np.square(i_obs_test))
                test_rel2I_all.append(test_rel2I)

        saveData.save_SEIR_trainLoss2mat_Covid(loss_s_all, loss_e_all, loss_i_all, loss_r_all, loss_n_all,
                                               actName=act_func, outPath=R['FolderName'])

        plotData.plotTrain_loss_1act_func(loss_s_all, lossType='loss2s', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_e_all, lossType='loss2e', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_i_all, lossType='loss2i', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_r_all, lossType='loss2r', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)
        plotData.plotTrain_loss_1act_func(loss_n_all, lossType='loss2n', seedNo=R['seed'], outPath=R['FolderName'],
                                          yaxis_scale=True)

        saveData.save_testMSE_REL2mat(test_mse2I_all, test_rel2I_all, actName='Infected', outPath=R['FolderName'])
        plotData.plotTest_MSE_REL(test_mse2I_all, test_rel2I_all, test_epoch, actName='Infected', seedNo=R['seed'],
                                  outPath=R['FolderName'], yaxis_scale=True)
        saveData.save_SEIR_testSolus2mat_Covid(s_nn2test, e_nn2test, i_nn2test, r_nn2test, name2solus1='snn2test',
                                               name2solus2='enn2test', name2solus3='inn2test', name2solus4='rnn2test',
                                               outPath=R['FolderName'])
        saveData.save_SEIR_testParas2mat_Covid(alpha_test, beta_test, gamma_test, name2para1='alpha2test',
                                               name2para2='beta2test', name2para3='gamma2test',
                                               outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0  # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）

    # 文件保存路径设置
    store_file = 'SEIR2covid'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['eqs_name'] = 'SEIR'
    R['input_dim'] = 1                      # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1                     # 输出维数

    # ------------------------------------  神经网络的设置  ----------------------------------------
    R['size2train'] = 70                    # 训练集的大小
    R['batch_size2train'] = 20              # 训练数据的批大小
    R['batch_size2test'] = 10               # 训练数据的批大小

    R['init_penalty2predict_true'] = 50     # Regularization parameter for boundary conditions
    R['activate_stage_penalty'] = 1         # 是否开启阶段调整边界惩罚项
    if R['activate_stage_penalty'] == 1 or R['activate_stage_penalty'] == 2:
        R['init_penalty2predict_true'] = 1

    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'      # The model of regular weights and biases
    # R['regular_weight'] = 0.000         # Regularization parameter for weights
    R['regular_weight'] = 0.001           # Regularization parameter for weights

    if 50000 < R['max_epoch']:
        R['learning_rate'] = 2e-4         # 学习率
        R['lr_decay'] = 5e-5              # 学习率 decay
    elif (20000 < R['max_epoch'] and 50000 >= R['max_epoch']):
        R['learning_rate'] = 1e-4         # 学习率
        R['lr_decay'] = 5e-5              # 学习率 decay
    else:
        R['learning_rate'] = 5e-5         # 学习率
        R['lr_decay'] = 1e-5              # 学习率 decay
    R['optimizer_name'] = 'Adam'          # 优化器
    # R['loss_function'] = 'L2_loss'
    R['loss_function'] = 'lncosh_loss'

    # R['hidden_layers'] = (10, 10, 8, 6, 6, 3)  # it is used to debug our work
    R['hidden_layers'] = (80, 80, 60, 40, 40, 20)
    # R['hidden_layers'] = (100, 100, 80, 60, 60, 40)
    # R['hidden_layers'] = (200, 100, 100, 80, 50, 50)
    # R['hidden_layers'] = (300, 200, 200, 100, 80, 80)
    # R['hidden_layers'] = (400, 300, 300, 200, 100, 100)
    # R['hidden_layers'] = (500, 400, 300, 200, 200, 100, 100)
    # R['hidden_layers'] = (600, 400, 400, 300, 200, 200, 100)
    # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)

    # 网络模型的选择
    # R['model'] = 'PDE_DNN'
    # R['model'] = 'PDE_DNN_BN'
    # R['model'] = 'PDE_DNN_scaleOut'
    R['model'] = 'PDE_DNN_Fourier'

    # 激活函数的选择
    # R['act_name'] = 'relu'
    # R['act_name'] = 'tanh'
    # R['act_name'] = 'leaky_relu'
    # R['act_name'] = 'srelu'
    R['act_name'] = 's2relu'
    # R['act_name'] = 'slrelu'
    # R['act_name'] = 'elu'
    # R['act_name'] = 'selu'
    # R['act_name'] = 'phi'

    R['total_population'] = 9776000
    # R['total_population'] = 1

    solve_SEIR2COVID(R)
