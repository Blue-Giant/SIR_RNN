import scipy.io as scio


def true_value2convid(trueArray, name2Array=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, str.upper(name2Array))
    key2mat = name2Array
    scio.savemat(outFile2data, {key2mat: trueArray})


def save_SIR_trainLoss2mat_Covid(loss_sArray, loss_iArray, loss_rArray, loss_nArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_i'
    key2mat_3 = 'loss_r'
    key2mat_4 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_iArray, key2mat_3: loss_rArray,
                                key2mat_4: loss_nArray})


def save_SIRD_trainLoss2mat_Covid(loss_sArray, loss_iArray, loss_rArray, loss_dArray, loss_nArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_i'
    key2mat_3 = 'loss_r'
    key2mat_4 = 'loss_d'
    key2mat_5 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_iArray, key2mat_3: loss_rArray,
                                key2mat_4: loss_dArray, key2mat_5: loss_nArray})


def save_SEIR_trainLoss2mat_Covid(loss_sArray, loss_eArray, loss_iArray, loss_rArray, loss_nArray, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_e'
    key2mat_3 = 'loss_i'
    key2mat_4 = 'loss_r'
    key2mat_5 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss_sArray, key2mat_2: loss_eArray, key2mat_3: loss_iArray,
                                key2mat_4: loss_rArray, key2mat_5: loss_nArray})


def save_SEIRD_trainLoss2mat_Covid(loss2s_arr, loss2e_arr, loss2i_arr, loss2r_arr, loss2d_arr, loss2n_arr,
                                   actName=None, outPath=None):
    outFile2data = '%s/LossSEIRD2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_s'
    key2mat_2 = 'loss_e'
    key2mat_3 = 'loss_i'
    key2mat_4 = 'loss_r'
    key2mat_5 = 'loss_d'
    key2mat_6 = 'loss_n'
    scio.savemat(outFile2data, {key2mat_1: loss2s_arr, key2mat_2: loss2e_arr, key2mat_3: loss2i_arr,
                                key2mat_4: loss2r_arr, key2mat_5: loss2d_arr, key2mat_6: loss2n_arr})


def save_test_paras2mat_Covid(para_array, name2para=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, name2para)
    key2mat_1 = str(name2para)
    scio.savemat(outFile2data, {key2mat_1: para_array})


def save_testSolus2mat_Covid(solus_array, name2solus=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, name2solus)
    key2mat_1 = str(name2solus)
    scio.savemat(outFile2data, {key2mat_1: solus_array})


def save_SIR_testSolus2mat_Covid(solu1_array, solu2_array, solu3_array, name2solus1=None,
                                 name2solus2=None, name2solus3=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solus2test')
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array})


def save_SIR_testParas2mat_Covid(para1_array, para2_array, name2para1=None, name2para2=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array})


def save_SEIR_testSolus2mat_Covid(solu1_array, solu2_array, solu3_array, solu4_array, name2solus1=None,
                                  name2solus2=None, name2solus3=None, name2solus4=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solus2test')
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    key2mat_4 = str(name2solus4)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array,
                                key2mat_4: solu4_array})


def save_SEIR_testParas2mat_Covid(para1_array, para2_array, para3_array, name2para1=None, name2para2=None,
                                  name2para3=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    key2mat_3 = str(name2para3)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array, key2mat_3: para3_array})


def save_SEIRD_testSolus2mat_Covid(solu1_array, solu2_array, solu3_array, solu4_array, solu5_array, name2solus1=None,
                                 name2solus2=None, name2solus3=None, name2solus4=None, name2solus5=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'solus2test')
    key2mat_1 = str(name2solus1)
    key2mat_2 = str(name2solus2)
    key2mat_3 = str(name2solus3)
    key2mat_4 = str(name2solus4)
    key2mat_5 = str(name2solus5)
    scio.savemat(outFile2data, {key2mat_1: solu1_array, key2mat_2: solu2_array, key2mat_3: solu3_array,
                                key2mat_4: solu4_array, key2mat_5: solu5_array})


def save_SEIRD_testParas2mat_Covid(para1_array, para2_array, para3_array, para4_array, para5_array, para6_array,
                                   name2para1=None, name2para2=None, name2para3=None, name2para4=None, name2para5=None,
                                   name2para6=None, outPath=None):
    outFile2data = '%s/%s.mat' % (outPath, 'paras2test')
    key2mat_1 = str(name2para1)
    key2mat_2 = str(name2para2)
    key2mat_3 = str(name2para3)
    key2mat_4 = str(name2para4)
    key2mat_5 = str(name2para5)
    key2mat_6 = str(name2para6)
    scio.savemat(outFile2data, {key2mat_1: para1_array, key2mat_2: para2_array, key2mat_3: para3_array,
                                key2mat_4: para4_array, key2mat_5: para5_array, key2mat_6: para6_array})


#######################################################################################################################
def save_trainLoss2mat_1actFunc(loss_it, loss_bd, loss, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd'
    key2mat_3 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_3: loss})


def save_trainLoss2mat_1act_Func(loss_it, loss_bd, loss_bdd, loss, actName=None, outPath=None):
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd'
    key2mat_3 = 'loss_bdd'
    key2mat_4 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_3: loss_bdd, key2mat_4: loss})


def save_trainLoss2mat_1actFunc_Dirichlet(loss_it, loss_bd, loss_bd2, loss_all, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/Loss2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/Loss2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/Loss2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd0'
    key2mat_4 = 'loss_bd2'
    key2mat_5 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_4: loss_bd2, key2mat_5: loss_all})


def save_trainLoss2mat_1actFunc_Navier(loss_U, loss_bd, loss_Psi, loss_bdd, loss, actName=None, outPath=None):
    # print('actName:', actName)
    # print('id of loss_it:', id(loss_U))
    # print('values of loss_it:', loss_U)
    if str.lower(actName) == 's2relu':
        outFile2data = '%s/Loss_s2ReLU.mat' % (outPath)
    elif str.lower(actName) == 'srelu':
        outFile2data = '%s/Loss_sReLU.mat' % (outPath)
    elif str.lower(actName) == 'relu':
        outFile2data = '%s/Loss_ReLU.mat' % (outPath)
    else:
        outFile2data = '%s/Loss_%s.mat' % (outPath, str(actName))

    # print('outFile2data:', outFile2data)

    key2mat_0 = 'lossU_%s' % (str(actName))
    key2mat_1 = 'lossBD_%s' % (str(actName))
    key2mat_2 = 'lossPsi_%s' % (str(actName))
    key2mat_3 = 'lossBDD_%s' % (str(actName))
    key2mat_4 = 'loss_%s' % (str(actName))
    scio.savemat(outFile2data, {key2mat_0: loss_U, key2mat_1: loss_bd, key2mat_2: loss_Psi, key2mat_3: loss_bdd, key2mat_4: loss})


def save_train_MSE_REL2mat(Mse_data, Rel_data, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/train_Err2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/train_Err2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/train_Err2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


# 一个mat文件保存一种数据
def save_testData_or_solus2mat(data, dataName=None, outPath=None):
    if str.lower(dataName) == 'testxy':
        outFile2data = '%s/testData2XY.mat' % (outPath)
        key2mat = 'Points2XY'
    elif str.lower(dataName) == 'testxyz':
        outFile2data = '%s/testData2XYZ.mat' % (outPath)
        key2mat = 'Points2XYZ'
    elif str.lower(dataName) == 'testxyzs':
        outFile2data = '%s/testData2XYZS.mat' % (outPath)
        key2mat = 'Points2XYZS'
    elif str.lower(dataName) == 'testxyzst':
        outFile2data = '%s/testData2XYZST.mat' % (outPath)
        key2mat = 'Points2XYZST'
    elif str.lower(dataName) == 'utrue':
        outFile2data = '%s/Utrue.mat' % (outPath)
        key2mat = 'Utrue'
    else:
        outFile2data = '%s/U%s.mat' % (outPath, dataName)
        key2mat = 'U%s' % (str.upper(dataName))

    scio.savemat(outFile2data, {key2mat: data})


# 合并保存数据
def save_2testSolus2mat(exact_solution, dnn_solution, actName=None, actName1=None, outPath=None):
    outFile2data = '%s/test_solus.mat' % (outPath)
    if str.lower(actName) == 'utrue':
        key2mat_1 = 'Utrue'
    key2mat_2 = 'U%s' % (actName1)
    scio.savemat(outFile2data, {key2mat_1: exact_solution, key2mat_2: dnn_solution})


# 合并保存数据
def save_3testSolus2mat(exact_solution, solution2act1, solution2act2, actName='Utrue', actName1=None, actName2=None,
                        outPath=None):
    outFile2data = '%s/solutions.mat' % (outPath)
    if str.lower(actName) == 'utrue':
        key2mat_1 = 'Utrue'
    key2mat_3 = 'U%s' % (actName1)
    key2mat_4 = 'U%s' % (actName2)
    scio.savemat(outFile2data, {key2mat_1: exact_solution, key2mat_3: solution2act1, key2mat_4: solution2act2})


# 合并保存数据
def save_4testSolus2mat(exact_solution, solution2act1, solution2act2, solution2act3, actName='Utrue', actName1=None,
                        actName2=None, actName3=None, outPath=None):
    outFile2data = '%s/solutions.mat' % (outPath)
    if str.lower(actName) == 'utrue':
        key2mat_1 = 'Utrue'
    key2mat_2 = 'U%s' % (actName1)
    key2mat_3 = 'U%s' % (actName2)
    key2mat_4 = 'U%s' % (actName3)
    scio.savemat(outFile2data, {key2mat_1: exact_solution, key2mat_2: solution2act1, key2mat_3: solution2act2,
                                key2mat_4: solution2act3})


def save_testLoss2mat_1act_Func(loss_it, loss_bd, loss_bd2, loss, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/Loss2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/Loss2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/Loss2ReLU.mat' % (outPath)
    outFile2data = '%s/Loss2%s.mat' % (outPath, actName)
    key2mat_1 = 'loss_it'
    key2mat_2 = 'loss_bd0'
    key2mat_3 = 'loss_bd2'
    key2mat_4 = 'loss'
    scio.savemat(outFile2data, {key2mat_1: loss_it, key2mat_2: loss_bd, key2mat_3: loss_bd2, key2mat_4: loss})


def save_testMSE_REL2mat(Mse_data, Rel_data, actName=None, outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/test_Err2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/test_Err2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/test_Err2ReLU.mat' % (outPath)
    outFile2data = '%s/test_Err2%s.mat' % (outPath, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


def save_testMSE_REL2mat_1type(Mse_data, Rel_data, dataType= None, actName=None,  outPath=None):
    # if actName == 's2ReLU':
    #     outFile2data = '%s/test_Err2s2ReLU.mat' % (outPath)
    # if actName == 'sReLU':
    #     outFile2data = '%s/test_Err2sReLU.mat' % (outPath)
    # if actName == 'ReLU':
    #     outFile2data = '%s/test_Err2ReLU.mat' % (outPath)
    outFile2data = '%s/%s_Err2%s.mat' % (outPath, dataType, actName)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


# 按误差类别保存，MSE和REL
def save_testErrors2mat(err_sReLU, err_s2ReLU, errName=None, outPath=None):
    if str.upper(errName) == 'MSE':
        outFile2data = '%s/MSE.mat' % (outPath)
        key2mat_1 = 'mse2sReLU'
        key2mat_2 = 'mse2s2ReLU'
        scio.savemat(outFile2data, {key2mat_1: err_sReLU, key2mat_2: err_s2ReLU})
    elif str.upper(errName) == 'REL':
        outFile2data = '%s/REL.mat' % (outPath)
        key2mat_1 = 'rel2sReLU'
        key2mat_2 = 'rel2s2ReLU'
        scio.savemat(outFile2data, {key2mat_1: err_sReLU, key2mat_2: err_s2ReLU})


def save_test_point_wise_err2mat(data2point_wise_err, actName=None, outPath=None):
    if str.lower(actName) == 'srelu':
        outFile2data = '%s/pERR2sReLU.mat' % (outPath)
        key2mat = 'pERR2sReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 's2relu':
        outFile2data = '%s/pERR2s2ReLU.mat' % (outPath)
        key2mat = 'pERR2s2ReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'relu':
        outFile2data = '%s/pERR2ReLU.mat' % (outPath)
        key2mat = 'pERR2ReLU'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'sin':
        outFile2data = '%s/pERR2Sin.mat' % (outPath)
        key2mat = 'pERR2Sin'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'powsin_srelu':
        outFile2data = '%s/pERR2p2SinSrelu.mat' % (outPath)
        key2mat = 'pERR2p2SinSrelu'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'tanh':
        outFile2data = '%s/pERR2tanh.mat' % (outPath)
        key2mat = 'pERR2tanh'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})
    elif str.lower(actName) == 'elu':
        outFile2data = '%s/pERR2elu.mat' % (outPath)
        key2mat = 'pERR2elu'
        scio.savemat(outFile2data, {key2mat: data2point_wise_err})