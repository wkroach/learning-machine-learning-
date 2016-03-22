from numpy import*
import operator
import math
#最近学英语太忙，先把代码搞定，注释等之后再补
def clc_shannon_ent(data_set): #计算当前数据集的香农熵
    '''data_set表示数据矩阵(list)，最后一列向量表示所属类别，其余为属性
        返回值为 数据集的香农熵'''
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set: #统计每种类别标签出现的次数
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1 
    shannon_ent = 0
    for key in label_counts: #计算香农熵 公式见书本
        prob = label_counts[key]/num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent

def create_data_set(): #简单测试数据集
    data_set = [[1, 1, 'yes'], \
                [1, 1, 'yes'], \
                [1, 0, 'no'], \
                [0, 1, 'no'], \
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

def split_data_set(data_set, axis, value):  #按给定特征划分数据集
    '''data_set表示数据集(list)，axis表示给定属性所在列数，value为属性特定值
        返回值为满足给定特征且已删去给定属性向量的数据集'''
    ret_data_set = []
    for feat_vec in data_set: 
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:]) #注意list.extend 和append的区别
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set

def choose_best_feature_2_split(data_set): #选择最好的数据集划分方式(计算出最大的信息增益/数据无序度的减小)
    ''' data_set表示数据矩阵(list)，最后一列向量表示所属类别，其余为属性
        返回 最适合划分的属性（特征）所在列数'''
    num_features = len(data_set[0]) - 1
    base_entropy = clc_shannon_ent(data_set) #基本熵
    best_info_gain = 0
    best_feature = -1
    for i in range(num_features): #对每一个属性，计算以它为基准划分得到熵，被原有数据集的熵相减，即当前划分的信息增益
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0
        for value in unique_vals: #对当前属性的每一个值进行划分，所有划分的自数据集的熵之和即当前划分的熵
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob*clc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list): #多数表决决定类别，以一个数据集中数目最多的类别标签表示类别，同KNN后半部分
    '''class_list 为没有其余属性的纯类别标签list'''
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0)+1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),  reverse = True)
    return sorted_class_count[0][0]

def create_tree(data_set, labels): #重点，ID3决策树构建主算法，递归实现
    '''data_set 为数据矩阵(list) 最后一列向量表示所属类别，其余为属性
        labels 为类别list'''
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list): #递归结束条件1：当前节点的数据集只有一种类别
        return class_list[0]
    if len(data_set[0]) ==1: #递归结束条件2：当前节点以利用所有属性，但数据集的类别仍有多种
        return majority_cnt(class_list) #利用多数表决法选择合适类别
    best_feat = choose_best_feature_2_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label:{}} #利用字典实现树
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree (split_data_set\
                                                    (data_set, best_feat,  value), sub_labels) #递归分类并建树
    return my_tree

def classify(input_tree, feat_labels, test_vec): #决策树分类算法
    '''input_tree为已构件好的决策树，feat_labels为属性列表，test_vec为被查询测试的属性向量'''
    first_str = list(input_tree.keys())[0] 
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str) #找到子树及对应的属性下标 
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if isinstance(second_dict[key], dict): #若已达到叶子节点，返回，否则继续递归
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

def store_tree(input_tree, filename): #序列化函数，将建好的决策树序列化到2进制文件上
    '''input_tree为已构建好的决策树，filename为要存放的文件'''
    import pickle
    fw = open(filename, 'wb') #由于是2进制，所以打开文件时要注意不能漏掉b
    pickle.dump(input_tree, fw) #序列化函数pickle.dump()
    fw.close()

def grab_tree(filename): #反序列化函数，将2进制文件的内容读出为数据结构
    '''filename为要反序列化的2进制文件'''
    import pickle
    fr = open(filename, 'rb') #同样注意写法
    return pickle.load(fr)

    
    
