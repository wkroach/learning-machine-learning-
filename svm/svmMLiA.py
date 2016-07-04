# -*- coding: utf-8 -*-
"""
Created on Tue May 31 00:19:19 2016

SVM learned from MLiA and updated into python3
@author: wkroach
"""

from numpy import*


def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []
    labelMat = []
    for currline in fr.readlines():
        dataline = currline.strip().split('\t')
        dataMat.append([float(dataline[0]), float(dataline[1])])
        labelMat.append(float(dataline[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    '''
    随机选择与i配对的j
    ----------------
    返回与标号i配对的下标j
    ----------------
    i：需要被匹配的拉格朗日参数alphas的下标
    m: 参数alphas的下标范围
    '''
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, L, H):
    '''
    限定所计算的alphas参数aj在L，H之间，使之满足kkt条件
    --------------------------
    返回处理后的参数aj
    --------------------------
    aj: 待处理的alphas参数
    L: 下限
    H：上限
    '''
    if aj < L:
        return L
    elif aj > H:
        return H
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简单版smo算法，通过随机选择要优化的参数对进行优化并进行迭代直到收敛或迭代次数达到上限
    --------------------------
    返回计算出来的参数b，与alphas
    --------------------------
    dataMatIn: 二维的列表，存放读入数据特征向量
    classLabels: 读入（训练）数据的已知类别，只有1，-1两种情况
    C: 参数C，决定离群点的影响程度，越大则离群点的影响越高
    toler:允许支持向量离超平面的精度（小数，0 - 1）通常0.00001
    maxIter：最大迭代次数
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = dataMatrix.shape
    alphas = mat(zeros((m, 1)))
    iter = 0  # 各参数初始化
    while(iter < maxIter):
        alphaPairsChanged = 0  # 记录次轮迭代是否有alphas对改变（优化）
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T \
                * (dataMatrix * dataMatrix[i, :].T)) + b  
            # 根据公式计算当前参数下的函数值
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
            # 优先对不在边界（即0和C之间）上的不满足kkt条件的参数进行优化
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, alphas[j] - alphas[i] + C)
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if(L == H):
                    print("L == H")
                    continue
                # 随机选择j并计算出当前状态下的alphaj的上下限，保证kkt条件
                eta = dataMatrix[i] * dataMatrix[i].T + \
                    dataMatrix[j] * dataMatrix[j].T - \
                    2.0 * dataMatrix[i] * dataMatrix[j].T
                if eta <= 0:
                    print("eta <= 0")
                    continue
                # 计算当前的eta，若eta不大于0，说明无法进行优化，直接进行下一对优化
                alphas[j] = alphaJold + labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], L, H)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 根据eta更新优化alphasj，若alphasj变化不明显，直接进行下一对优化
                alphas[i] += labelMat[j] * labelMat[i] * \
                    (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) \
                    * dataMatrix[i] * dataMatrix[i].T - labelMat[j] \
                    * (alphas[j] - alphaJold) * dataMatrix[i] \
                    * dataMatrix[j].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) \
                    * dataMatrix[i] * dataMatrix[j].T - labelMat[j] \
                    * (alphas[j] - alphaJold) * dataMatrix[j] \
                    * dataMatrix[j].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % \
                    (iter, i, alphaPairsChanged))
                # 更新alphasi和b，并使记录变量加1表示此对更新优化成功
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
        # 若对整个数据集进行便利均未进行更新优化，则使迭代次数加1，表示目前随机结果无需优化
        # 反之，则可以优化并有可能继续优化下去，应该重新记录遍历次数，直到出现优化完全为止
    return b, alphas


def plotHipreplan(b, alphas):
    '''
    根据b，与alphas画出分割模拟图
    '''
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet("testSet.txt")
    dataArr = array(dataMat)
    n = dataArr.shape[0]
    alphas = array(alphas)
    weights = zeros(2)
    for i in range(n):
        weights += alphas[i] * dataArr[i] * labelMat[i]
    b = float(b)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1 . append(dataMat[i][0])
            ycord1 . append(dataMat[i][1])
        else:
            xcord2 . append(dataMat[i][0])
            ycord2 . append(dataMat[i][1])
    fig = plt . figure()
    ax = fig . add_subplot(111)
    ax . scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax . scatter(xcord2, ycord2, s=30, c='green')
    y = arange(-6, 6, 0.1)
    x = (-b - weights[1] * y) / weights[0]
    ax . plot(x, y)
    plt . xlabel('X1')
    plt . ylabel('X2')
    plt.show()


class optStruct():
    '''
    将svm所用到的数据封装在一个类中，方便进行操作
    '''
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        '''
        kTup：核函数，有线性和高斯核两种类型
        '''
        self.C = C
        self.tol = toler
        self.X = dataMatIn
        self.labelMat = classLabels
        self.m = dataMatIn.shape[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        # 存放每个向量函数当前函数差值的记录表，用于查找最优alpha对，
        # 第二列存放值，第一列标记是否优化过
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i], kTup)
        # 预先将所有核函数值计算好（打表），提高之后调用的效率


def calcEk(oS, k):
    '''
    根据核函数计算函数值并返回差值
    ---------------------------
    返回：根据核函数与当前参数计算出的函数值与实际值的差值Ek
    ----------------------------
    oS：数据结构体
    k: 第k个向量
    '''
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    '''
    利用最长步距方法寻找与alphasi配对的最佳alphasj
    ----------------------
    返回最佳alphasj的下标j与向量j的函数差值Ej
    ----------------------
    i: 需匹配的alphas下标
    oS: 数据结构体，存放于此svm相关的所有数据
    Ei: alphasi的函数差值
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    # 在已标记为更新优化后的参数中选配对参数alphaj
    if len(validEcacheList) > 1:  # 暴力查找最优配对的j
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)  # 若在已更新的参数中无最优解，则在全参数中随机选择
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    '''
    更新参数的函数差值并标记
    -------
    无返回值
    -------
    oS: svm的数据存储结构
    k: 待更新的向量
    '''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def inner(i, oS):
    '''
    完整版platt-smo的内循环函数，对下标为i的参数alphai找到最佳配对alphaj并进行优化更新
    基本过程同smoSimple中的过程
    --------------
    返回0或1，表示更新是否成功
    --------------
    i: 待配对的参数alpha的下标
    oS：svm数据
    '''
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or \
        (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)  # 利用最大步长查找配对而不是单一随机查找
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.alphas[j] - oS.alphas[i] + oS.C)
        else:
            L = max(0, oS.alphas[i] + oS.alphas[j] - oS.C)
            H = min(oS.C, oS.alphas[i] + oS.alphas[j])
        if L == H:
            print("L == H")
            return 0
        eta = oS.K[i, i] + oS.K[j, j] - 2.0 * oS.K[i, j]
        if eta <= 0:
            print("eta <= 0")
            return 0
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], L, H)
        updateEk(oS, j)  # 若更新成功则要对oS中的函数差值表进行更新
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving")
            return 0
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] \
            * (alphaJold - oS.alphas[j])
        updateEk(oS, i)  # 同上
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) \
            * (oS.K[i, i]) - oS.labelMat[j] \
            * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) \
            * oS.K[i, j] - oS.labelMat[j] \
            * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    '''
    完整版platt-smo算法
    ------------------------------
    返回计算出来的参数b，与alphas
    --------------------------
    dataMatIn: 二维的列表，存放读入数据特征向量
    classLabels: 读入（训练）数据的已知类别，只有1，-1两种情况
    C: 参数C，决定离群点的影响程度，越大则离群点的影响越高
    toler:允许支持向量离超平面的精度（小数，0 - 1）通常0.00001
    maxIter：最大迭代次数
    kTup：核函数，第一项为类型，默认为线性核，第二项当类型为高斯核时作为调整参数
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        # 外循环，在全体数据和界内数据间交替更新迭代直到收敛-
        # 即对全体数据遍历仍无参数可优化或达到迭代最大次数
        # 若界内有参数更新，则始终对界内数据进行更新优化直到无更新时换成全局更新
        # 全局更新一次后不用重复全局更新，若此时仍旧无更新，则说明优化完成
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += inner(i, oS)
            print(" fullSet, iter: %d i:%d, pairs changed %d" %\
                    (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 找出所有边界内的参数进行迭代更新优化
            for i in nonBoundIs:
                alphaPairsChanged += inner(i, oS)
                print("non - bound, iter: %d i: %d, pairs changed %d" %\
                    (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    '''
    根据拉格朗日参数alphas与所有向量点和实际函数值求出实际参数向量w
    ---------------
    返回参数回归系数向量w
    ---------------
    alphas：为拉格朗日参数alphas，array
    dataArr：全体数据向量
    classLabels：全体数据向量实际类型
    '''
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = X.shape
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i].T)
    return w


def kernelTrans(X, A, kTup):
    '''
    核函数计算函数，分为线性和高斯两种情况
    -------------
    返回输入向量集X对于向量A的核函数计算值
    -------------
    X：输入向量集
    A：带计算向量
    kTup：核函数类型以及计算参数
    '''
    m, n = X.shape
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError("Houston We Have a Problem -- the kernel is not recongnized")
    return K


def test():
    '''
    线性函数测试函数，自带画图功能
    '''
    dataArr, labelsArr = loadDataSet("testSet.txt")
    # oS1 = optStruct(mat(dataArr), mat(labelsArr).transpose(), 0.6, 0.01)
    b1, alphas1 = smoP(dataArr, labelsArr, 0.6, 0.001, 40)
    plotHipreplan(b1, alphas1)


def testRbf(k1=1.3):
    '''
    高斯核函数测试功能
    '''
    dataArr, labelArr = loadDataSet("testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % sVs.shape[0])
    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet("testSetRBF2.txt")
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = dataMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def img2vector(filename):
    '''图像转换函数，将储存为文本的数字图像转换为1*1024的向量'''
    return_vect = zeros(1024)
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[32*i+j] = int(line_str[j])
    return return_vect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigts(kTup=('rbf', 10)):
    '''
    手写数字的测试函数
    '''
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % sVs.shape[0])
    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages("testDigits")
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = dataMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
