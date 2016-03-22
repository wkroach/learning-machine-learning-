import kNN

group,labels = kNN.createDataSet()

def classify0(inX,dataset,labels,k):
    data_set_size = dataset.shape[0] #获取样本数据点数目(第一维大小,shape[0]表示)
    diff_mat = tile(inX,(data_set_size,1))-dataset 
    #将输入向量复制多份，并求出与每个数据点的每个属性的差值，结果保存为二维数组
    #第i行向量表示与第i个数据点的属性差值向量
    sq_diff_mat = diff_mat*2 #平方
    sq_distances = sq_diff_mat.sum(axis=1)
    #axis=i,就以第i维为方向的向量相加,在二维向量中，axis=0按列相加，axis=1时按行相加
    distances=sq_distances**0.5 #开根号
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_indicies[i]]    
        class_count[vote_ilabel] = class_count.get(vote_ilabel,0) + 1
    #将前k个label的分类情况进行统计，并存在字典class_count里
    sorted_class_count = sorted(class_count.items(),\
    key = itemgetter(1),reverse = True) 
    #将统计字典按照从大到小排序，并将键值对存在元组里并返回一个list
    #调用operator里的itemgetter函数
    #写法2：sorted_class_count = sorted(class_count.items(),\
    #           key = lambda t: t[1], reverse = True)
    #使用lambda达到同样的效果
    return sorted_class_count[0][0]
    #第一个键值对的类别即频率最高的类别，即所求结果