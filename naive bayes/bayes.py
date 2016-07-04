""" 
    naive bayes code and simple testing data
    adapted from MLIA in python 3.5.1
    test on surface3
"""
from numpy import*

def load_data_set():
    posting_list=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0,1,0,1,0,1] 
    return posting_list,class_vec
    
def create_vocab_list(data_set):
    '''生成单词集 data_set为文本'''
    vocab_set = set()
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

list_of_posts, list_classes = load_data_set()

def set_of_words_2_vec(vocab_list, input_set):
    ''' 由词集模型生成文本向量，即每个词是否出现过
    vocab_list为目标文本，input_set为词集'''
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return return_vec
    
def train_nb0 (train_martix, train_category):
    '''训练算法， 原理见书本
    train_martix为训练词向量集，train_category为训练集的标签'''
    num_train_docs = len(train_martix)
    num_words = len(train_martix[0])
    p_abusive = sum(train_category)/float(num_train_docs)
    p0_num = ones(num_words); p1_num = ones(num_words)#初始分子为1，分母为2，防止概率为0和1时对结果产生较大影响
    p0_denom = 2.0; p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_martix[i]
            p1_denom += sum(train_martix[i])
        else:
            p0_num += train_martix[i]
            p0_denom += sum(train_martix[i])
    p1_vect = log(p1_num/p1_denom)
    p0_vect = log(p0_num/p0_denom)
    return p0_vect, p1_vect, p_abusive

def classify_nb(vec2_classify, p0_vect,p1_vect,p_class1):
    '''分类
    vec2_classify为词向量（array），后面的均为概率向量或概率
    '''
    p1 = sum(vec2_classify * p1_vect) + log(p_class1)#取对数防止下溺出
    p0 = sum(vec2_classify * p0_vect) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0
        
def testing_bayes():
    '''测试算法'''
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    train_mat = []
    for postin_doc in list_of_posts:
        train_mat.append(set_of_words_2_vec(my_vocab_list, postin_doc))
    p0V, p1V, pAb = train_nb0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words_2_vec(my_vocab_list, test_entry))
    print("['love', 'my', 'dalmation']:",classify_nb(this_doc, p0V, p1V, pAb))
    test_entry = ['stupid','garbage']
    this_doc = array(set_of_words_2_vec(my_vocab_list, test_entry))
    print("['stupid','garbage']:",classify_nb(this_doc, p0V, p1V, pAb))

def bag_of_words_2_vec_mn(vocab_list, input_set):
    '''词袋模型，记录每个词出现次数
    参数与词集模型一致
    '''
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec

def textParse (bigString):
    '''
        用正则表达式处理文本，之选长度大于等于3的字符串
        bigstring 为需要处理的字符串
    '''
    import re
    listOfTokens = re.split (r'\W*', bigString)
    return [tok.lower () for tok in listOfTokens if len (tok) > 2]
    
def spamTest ():
    ''' 
        the data in the book is wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        23th text in the email/ham is encoded in ansi!!!!!!!!! 
        fuck !!!!
    '''
    import random
    docList = []; classList = []; fullText = []
    for i in range (1,26):
        fr = open ('email/spam/%d.txt' % i)
        wordList = textParse (fr.read ())
        docList.append (wordList)
        fullText.extend (wordList)
        classList.append (1)
        fr2 = open ('email/ham/%d.txt' % i)
        wordList = textParse (fr2.read ())
        docList.append (wordList)
        fullText.extend (wordList)
        classList.append (0)
    vocabList = create_vocab_list (docList)
    trainingSet = list (range (50)); testSet = []
    for i in range (10) :#选10个作训练样本，其余作测试
        randIndex = int (random.uniform (0,len(trainingSet)))
        testSet.append (trainingSet [randIndex])
        del(trainingSet [randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet :
        trainMat.append (set_of_words_2_vec (vocabList, docList [docIndex]))
        trainClasses.append (classList [docIndex])
    p0V, p1V, pSpam = train_nb0 (array (trainMat), array (trainClasses))
    errorCount = 0
    for docIndex in testSet :
        wordVector = set_of_words_2_vec (vocabList, docList [docIndex])
        a = classify_nb (array (wordVector), p0V, p1V, pSpam)
        b = classList [docIndex]
        if a != b :
            errorCount += 1
    print ('the error rate is: ', errorCount / len (testSet))

    
<<<<<<< HEAD
=======
#spamTest()
>>>>>>> c32eaef15dc1c0633d23a9592209aa988bf0086e
    
    
    