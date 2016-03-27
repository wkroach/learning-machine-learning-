""" 
    naive bayes code and simple testing data
    adapted from MLIA in python 3.5.1
    test on surface3
"""

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
    vocab_set = set()
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

list_of_posts, list_classes = load_data_set()

def set_of_words_2_vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return return_vec
    
def train_nb0 (train_martix, train_category):
    num_train_docs = len(train_martix)
    num_words = len(train_martix[0])
<<<<<<< HEAD
    p_abusive = sum(train_category) / float(num_train_docs)
=======
    p_abusive = sum(train_category)/float(num_train_docs)
>>>>>>> fd4ee23c22e5897c0433d6cc4dcdf1771c212794
    p0_num = zeros(num_words); p1_num = zeros(num_words)
    p0_denom = 0.0; p1_denom = 0.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_martix[i]
            p1_denom += sum(train_martix[i])
        else:
            p0_num += train_martix[i]
            p0_denom += sum(train_martix[i])
<<<<<<< HEAD
    p1_vect = p1_num / p1_denom
    p0_vect = p0_num / p0_denom
=======
    p1_vect = p1_num/p1_denom
    p0_vect = p0_num/p0_denom
>>>>>>> fd4ee23c22e5897c0433d6cc4dcdf1771c212794
    return p0_vect, p1_vect, p_abusive

