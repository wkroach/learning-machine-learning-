from numpy import*
from operator import*
from os import listdir

def create_data_set():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group ,labels
    
    
def classify0(inX,dataset,labels,k):
    '''kNN主函数，数据分类函数，接受目标向量，已有数据集，已有类别（标签）集，k，返回目标向量的类别'''
    data_set_size = dataset.shape[0] #获取样本数据点数目(第一维大小,shape[0]表示)
    diff_mat = tile(inX,(data_set_size,1))-dataset 
    #将输入向量复制多份，并求出与每个数据点的每个属性的差值，结果保存为二维数组
    #第i行向量表示与第i个数据点的属性差值向量
    sq_diff_mat = diff_mat**2 #平方
    sq_distances = sq_diff_mat.sum(axis=1)#axis=i,就以第i维为方向的向量相加,在二维向量中，axis=0按列相加，axis=1时按行相加
    distances=sqrt(sq_distances) #开根号
    sorted_dist_indicies = distances.argsort()#argsort()返回排序后的下标序列的array
    class_count = {}
    for i in range(k):#将前k个label的分类情况进行统计，并存在字典class_count里
        vote_ilabel = labels[sorted_dist_indicies[i]]    
        class_count[vote_ilabel] = class_count.get(vote_ilabel,0) + 1
    sorted_class_count = sorted(class_count.items(),\
    key = itemgetter(1),reverse = True) #将统计字典按照从大到小排序，并将键值对存在元组里并返回一个list,调用operator里的itemgetter函数
    #写法2：
    # sorted_class_count = sorted(class_count.items(),\
    # key = lambda t: t[1], reverse = True)
    #使用lambda达到同样的效果
    return sorted_class_count[0][0]#第一个键值对的类别即频率最高的类别，即所求结果

def file2matrix(filename):
    '''数据转化函数，将文本数据转化为array（mat）形式'''
    fr = open(filename)#打开文件，open()用法见书本
    array_of_lines = fr.readlines()#readlines()读取文件每一行，返回单行字符串的列表
    number_of_lines = len(array_of_lines)
    return_mat = zeros((number_of_lines,3))#创建一个n行3列的多重数组（矩阵）
    class_label_vector = []#用来存放标签（类别）
    index = 0#迭代计数器
    for line in array_of_lines:
        line = line.strip()#str.strip()返回去掉首尾空白字符的字符串的副本，原先字符串不变
        list_from_line = line.split('\t')#以制表符为分隔符收集元素（原数据中以制表符分格元素）
        return_mat[index,:] = list_from_line[:3]#注意numpy的多重数组的分片操作方法，具体见书本
        class_label_vector.append(int(list_from_line[-1])) #记得将字符串转化为整型
        index+=1
    return return_mat,class_label_vector

def auto_norm(data_set):#data_set是numpy中的array，不是list
    '''数据归一化函数，防止属性中出现数据过大或过小的属性，影响结果'''
    min_vals = data_set.min(0)#array.min(0)表示从每列选最小值，并返回一个array
    max_vals = data_set.max(0)#与上面类似
    ranges = max_vals-min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals,(m,1))#从上面的得到的最小（大）值是一个1维数组，需要复制成多维数组再运算
    norm_data_set = norm_data_set/tile(ranges,(m,1))
    return norm_data_set, ranges,min_vals

def dating_class_test():
    '''测试函数，用来测试kNN函数的错误率，0表示完全正确，1表示完全错误
        此函数是自包含的，内部包含了数据集的转化和归一化
        默认使用数据集的前百分之10用来测试，后百分之90用来作已知数据集
        计算公式为错误次数除以测试数据数'''
    horatio = 0.10#控制测试数据占全部数据的比例
    dating_data_mat, dating_labels =\
    file2matrix('datingTestSet2.txt')
    norm_mat,ranges,min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*horatio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i,:],\
                        norm_mat[num_test_vecs:m,:],\
                        dating_labels[num_test_vecs:m],3)
        print("the classifier came back with: %d, the real answer\
         is: %d" % (classifier_result,dating_labels[i]))
        if(classifier_result != dating_labels[i]):
            error_count += 1
    print("the total error rate is: %f"%(error_count/float(num_test_vecs)))

def classify_person():
    '''预测函数，输入一个人三个参数，返回由kNN算法在已有数据集上分类的结果'''
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent splaying video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per years?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr-min_vals)/ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person: ", result_list[classifier_result-1])

def  img2vector(filename):
    '''图像转换函数，将储存为文本的数字图像转换为1*1024的向量'''
    return_vect = zeros(1024)
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[32*i+j] = int(line_str[j])
    return return_vect
    
def handwriting_class_test():
    '''自包含的手写数字测试函数，用来测试kNN算法对于手写数字识别的错误率'''
    hw_labels = []
    training_file_list = listdir('trainingDigits')#os.listdir()读取一个文件夹，以字符串数组形式返回文件夹下的所有文件名
    m = len(training_file_list)#读取文件数量
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]#每个文件的文件名是由形如"filename.filetype"的形式组成，用'.'划分出文件的名字与类型
        class_num_str = int(file_str.split('_')[0])#所有手写数字文本名形如"number_num"的形式组成，用'_'划分出数字与编号
        hw_labels.append(class_num_str)
        training_mat[i ,  :] = img2vector('trainingDigits/%s' % file_name_str)#文件名可用"目录名/文件名"的方式读取，而不需要进入目录，且可用格式化输出
    test_file_list = listdir('testDigits')#测试集处理同上
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 3)
        print("the classifier came back with: %d,  the real answer is: %d" % (classifier_result, class_num_str))
        if(classifier_result != class_num_str) :
            error_count+=1.0
    print("\nthe classifier came back with: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count/float(m_test)))#注意格式化的格式问题，要先将结果算出再带入格式化

        
      
    
    
    
