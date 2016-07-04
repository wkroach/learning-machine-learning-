"""
    logRegres from machine learning in action coded in python3 by wkroach
<<<<<<< HEAD

"""


from numpy import*


def loadDataSet():
    '''
        载入数据集，返回数据表与标签
    '''
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # str.strip()返回去掉首尾空白字符的字符串的副本，原先字符串不变
        # 注意字符串切分出来的是字符，需要转换成浮点型
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))  # 同上
    return dataMat, labelMat


def sigmoid(inX):
=======
"""

from numpy import*
import operator

def loadDataSet () :
    '''
        载入数据集，返回数据表与标签
    '''
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines() :
        lineArr = line.strip ().split ()#str.strip()返回去掉首尾空白字符的字符串的副本，原先字符串不变
        dataMat.append ( [ 1.0, float(lineArr[0]), float(lineArr[1]) ] )#注意字符串切分出来的是字符，需要转换成浮点型
        labelMat.append(float(lineArr[2]))#同上
    return dataMat, labelMat
    
def sigmoid (inX) :
>>>>>>> c32eaef15dc1c0633d23a9592209aa988bf0086e
    '''
        sigmoid 二分函数，值域为 0 到 1
        inX为输入数据，可以是向量或单一的数
    '''
<<<<<<< HEAD
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
=======
    return 1.0 / (1 + exp ( -inX ))
    
def gradAscent (dataMatIn, classLabels) :
>>>>>>> c32eaef15dc1c0633d23a9592209aa988bf0086e
    '''
        梯度上升法求参数（回归系数）
        dataMatIn为输入数据，每个数据点均为向量
        classLabels为标签向量
        输出为参数的向量
<<<<<<< HEAD
        此算法有一个基于极大似然估计与多元函数求偏导的数学推导
        在此仅将算法的结论给出，即每次用当前值域实际值的差来作为步进
    '''
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels) . transpose()  # 由于后面要与参数列向量做差，这里标签为行向量，需要翻转
    m, n = dataMatrix . shape
    alpha = 0.001  # 梯度上升的上升步长
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))  # 初始参数向量，均为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 所有数据点均对参数向量做点乘，并返回经sigmoid计算后的向量
        error = (labelMat - h)  # 计算与实际值之间的差距
        weights = weights + alpha * dataMatrix . transpose() * error  # 梯度上升，利用向量对每个参数均做了迭代
    return weights


def plotBestFit(weights):
    '''
        weights 为数组array，注意不是list 和 matrix
        如果是matrix ，一定要输入weights.getA() 输入它的数组形式
    '''
=======
        
        此算法有一个基于极大似然估计与多元函数求偏导的数学推导
        在此仅将算法的结论给出，即每次用当前值域实际值的差来作为步进
    '''
    dataMatrix = mat (dataMatIn)
    labelMat = mat (classLabels) . transpose ()#由于后面要与参数列向量做差，这里标签为行向量，需要翻转
    m, n = dataMatrix . shape
    alpha = 0.001#梯度上升的上升步长
    maxCycles = 500#迭代次数
    weights = ones((n,1))#初始参数向量，均为1
    for k in range (maxCycles) :
        h = sigmoid (dataMatrix * weights)#所有数据点均对参数向量做点乘，并返回经sigmoid计算后的向量
        error = (labelMat - h)#计算与实际值之间的差距
        weights = weights + alpha * dataMatrix . transpose () * error#梯度上升，利用向量对每个参数均做了迭代
    return weights

def plotBestFIt (weights)  :
>>>>>>> c32eaef15dc1c0633d23a9592209aa988bf0086e
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = dataArr.shape[0]
<<<<<<< HEAD
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1 . append(dataMat[i][1])
            ycord1 . append(dataMat[i][2])
        else:
            xcord2 . append(dataMat[i][1])
            ycord2 . append(dataMat[i][2])
    fig = plt . figure()
    ax = fig . add_subplot(111)
    ax . scatter(xcord1, ycord1, s=30, c='red', marker='s')  # 画两种点的散点图
    ax . scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3, 3, 0.1)  # 取start与end之间步进为0.1（第三个参数）的所有数
    # 即将x2视为y，x1视为x，由于sigmoid函数参数inX大于0时为1，小于为0，所以以0作为分界线
    # 将方程化为w[2]*x2 + w[1]*x1 + w[0]*1 = 0的形式
    y = (-weights[0] - weights[1] * x) / weights[2]
    # 根据sigmoid函数的x值是否大于0来将两种点划分在线两边了
    # 若x大于0，sig函数为1（1类），反之为0（2类）
    ax . plot(x, y)
    plt . xlabel('X1')
    plt . ylabel('X2')
    plt.show()


def stoGradAscent0(dataMatrix, classLabels):
    '''
        dataMatrix为array，classLabels为列表
        未优化的随机化梯度上升，对单个数据点进行计算而不是对所有点进行若干次迭代计算
        dataMatrix 应该为矩阵或数组
     '''
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # 此处由于每次是对单个向量（数据点）进行计算，weights应该取一维array
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))  # array乘法是数组间对应元素的成绩，返回数组
        # 因此若要做到矩阵乘法的效果应对结果求和
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stoGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
        优化后的随机梯度上升，一方面做到alpha动态处理，
        减小之后数据对回归系数产生较大影响同时又不至于没有影响
        同时避免了alpha的严格下降
        另一方面随机取点，降低了周期性的波动
        numIter为迭代次数，默认为150
    '''
    m, n = dataMatrix.shape
    weights = ones(n)
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4 / (i + j + 1) + 0.01  # 注意这里alpha公式，
            # 越后面的数据影响力越小，但不为0
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机化选数据点
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
        del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    '''
        logistic分类器
        inX 为输入向量（array） weights为回归系数向量（array）
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    '''
        测试函数，对已经预处理过的数据文件进行回归（训练），求出回归系数后
        进行测试并统计错误率
    '''
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stoGradAscent1(
        array(trainingSet),
        trainingLabels, 100)
    errorCount = 0.0
    numTestVector = 0.0
    for line in frTest.readlines():
        numTestVector += 1
        currLine = line.strip().split('\t')
        trainVector = []
        for i in range(21):
            trainVector.append(float(currLine[i]))
        ans = classifyVector(array(trainVector), trainWeights)
        if int(ans) != int(float(currLine[21])):
            errorCount += 1
    errorRate = float(errorCount / numTestVector)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def mulTest():
    '''
        多次测试，计算错误率的平均值
    '''
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations ther average error rate is: %f" %
          (numTests, errorSum / numTests))


# mulTest()

=======
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range (n) :
        if int(labelMat [i]) == 1 :
            xcord1 . append (dataMat[i][1]); ycord1 . append (dataMat[i][2])
        else :
            xcord2 . append (dataMat[i][1]); ycord2 . append (dataMat[i][2])
    fig = plt . figure ()
    ax = fig . add_subplot (111)
    ax . scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax . scatter (xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3, 3, 0.1)#取start与end之间步进为0.1（第三个参数）的所有数
    y = (-weights[0] - weights[1] * x) / weights[2]# 即将x2视为y，x1视为x，由于sigmoid函数参数inX大于0时为1，小于为0，所以以0作为分界线
    ax . plot (x, y)
    plt . xlabel ('X1'); plt . ylabel ('X2')
    plt.show()

>>>>>>> c32eaef15dc1c0633d23a9592209aa988bf0086e
