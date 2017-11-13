#CSCI 5832 Natural Language Processing
#@author Paramjot Singh

#Hotel review sentiment analysis using Naive-Bayes with add one smoothing

import math
import re

def count(train):
    word_dict = {}
    train_file = open(train, 'r')

    for line in train_file:
        #removing all the punctuations to avoid difference between wife and wife,
        #it wont help in sentiment analysis, but will reduce the the size of vocabulary
        line = re.sub('[,.:;?!]','',line)
        #changing words to lower case and splitting based on spaces
        line = line.lower().strip().split()
        line = line[1:]  #to avoid ID value for line

        for word in line:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1

    train_file.close()

    return word_dict


def prob_calc(train, test_data, train_vocab):

    prob_list = []
    word_count = 0
    for word in train:
        word_count += train[word]

    for word in test_data:
        word_freq = 0
        if word in train:
            word_freq = train[word]
        else:
            word_freq = 0

        prob = (word_freq + 1)/(word_count+len(train_vocab))
        prob_list.append(prob)
    return prob_list

def print_result(result):
    with open('singh-paramjot-assgn3-out.txt', 'w') as file:
        for id in result:
            file.write(id + '\t' + result[id] +'\n')
    file.close()


def sentiment_calc(test_data):
    result = {}
    for line in test_data:
        pos_prob_list = prob_calc(pos_train, test_data[line], train_vocabulary)
        neg_prob_list = prob_calc(neg_train, test_data[line], train_vocabulary)

        #no need to add prior in this case since its same lg(0.5)
        pos_prob = 0
        neg_prob = 0
        for prob in pos_prob_list:
            pos_prob += math.log(prob)

        for prob in neg_prob_list:
            neg_prob += math.log(prob)

        # print(line)
        if pos_prob > neg_prob:
            print(line.upper()+'\tPOS\n')
            result[line.upper()] = 'POS'

        elif neg_prob > pos_prob:
            print(line.upper() + '\tNEG\n')
            result[line.upper()] = 'NEG'

    return result


if __name__ == "__main__":
    #count positive train data words
    pos_train = count('hotelPosT-train.txt')
    #count negative train data words
    neg_train = count('hotelNegT-train.txt')


    # print("Positive dic ")
    # ps = [(k, pos_train[k]) for k in sorted(pos_train, key=pos_train.get, reverse=True)]
    # print(ps)

    # Clipping the high freq words and single occurrence words
    # pos_train = { k:v for k, v in pos_train.items() if v < 100 and v > 1}

    # print("Negative dic ")
    # ns = [(k, neg_train[k]) for k in sorted(neg_train, key=neg_train.get, reverse=True)]
    # print(ns)

    # Clipping the high freq words and single occurrence words
    # neg_train = {k: v for k, v in neg_train.items() if v < 100 and v > 1}

    ps = [(k, pos_train[k]) for k in sorted(pos_train, key=pos_train.get, reverse=True)]
    print(ps)

    ns = [(k, neg_train[k]) for k in sorted(neg_train, key=neg_train.get, reverse=True)]
    print(ns)

    #join both dictionary to form vocabulary
    train_vocabulary = pos_train.copy()
    train_vocabulary.update(neg_train)

    s = [(k, train_vocabulary[k]) for k in sorted(train_vocabulary, key=train_vocabulary.get, reverse=True)]
    print(s)
    # sorted(((v, k) for k, v in train_vocabulary.iteritems()), reverse=True)
    # print(train_vocabulary)

    #get test data
    test_data = {}
    test_file = open('HW3-testset.txt', 'r')
    for line in test_file:
        line = re.sub('[.,;:!?]', '', line)
        line = line.lower().strip().split()

        test_data[line[0]] = line[1:]

    test_file.close()

    result = sentiment_calc(test_data)

    print_result(result)

