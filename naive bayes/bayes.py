""" 
    naive bayes code and simple testing data
    adapted from MLIA in python 3.5.1
    
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
    
