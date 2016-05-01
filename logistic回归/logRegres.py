"""
    logRegres from machine learning in action coded in python3 by wkroach
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
    '''
        sigmoid 二分函数，值域为 0 到 1
        inX为输入数据，可以是向量或单一的数
    '''
    return 1.0 / (1 + exp ( -inX ))
    
def gradAscent (dataMatIn, classLabels) :
    '''
        梯度上升法求参数（回归系数）
        dataMatIn为输入数据，每个数据点均为向量
        classLabels为标签向量
        输出为参数的向量
        
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
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = dataArr.shape[0]
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

