import numpy as np
import pandas as pd
import nltk
from nltk import conlltags2tree
import _pickle as cPickle
import sklearn
import pprint
import sklearn_crfsuite


####### FEATURE EXTRACTION #######
def word_features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]

    # first word
    if i==0:
        prevword = '<START>'
        prevpos = '<START>'
    else:
        prevword = sent[i-1][0]
        prevpos = sent[i-1][1]

    # last word
    if i == len(sent)-1:
        nextword = '<END>'
        nextpos = '<END>'
    else:
        nextword = sent[i+1][0]
        nextpos = sent[i+1][1]

    # word is in gazetteer
    gazetteer = gazetteer_lookup(word)

    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]

    return {'word':word,
            'pos': pos,
            'prevword': prevword,
            'prevpos': prevpos,
            'nextword': nextword,
            'nextpos': nextpos,
            'word_is_city': gazetteer[0],
            'word_is_state': gazetteer[1],
            'word_is_county': gazetteer[2],
            'word_is_digit': word in 'DIGITDIGITDIGIT',
            'suff_1': suff_1,
            'suff_2': suff_2,
            'suff_3': suff_3,
            'suff_4': suff_4,
            'pref_1': pref_1,
            'pref_2': pref_2,
            'pref_3': pref_3,
            'pref_4': pref_4 }


#######  GEOGRAPHICAL GAZETTEER LOOKUP  #######
# reading a file containing list of US cities, states and counties
us_cities = pd.read_csv("us_cities_states_counties.csv", sep="|")

# storing cities, states and counties as sets
cities = set(us_cities['City'].str.lower())
states = set(us_cities['State full'].str.lower())
counties = set(us_cities['County'].str.lower())

# define a function to look up a given word in cities, states, county
def gazetteer_lookup(word):
    return (word in cities, word in states, word in counties)


####### PREPROCESS USER QUERY AND GET WORD FEATURES #######
def process_user_query(sent_string):
    tokens = nltk.word_tokenize(sent_string)
    pos_tags = nltk.pos_tag(tokens)

    # create features from words in query q
    query_features = [word_features(pos_tags, i) for i in range(len(pos_tags))]
    return(pos_tags, query_features)

# read tuned crf from pickle file
with open('tuned_crf_classifier.pkl', 'rb') as fid:
    crf = cPickle.load(fid)


####### RETURN IOB LABELS #######
def predict_IOB_labels(s):
    # generate query features for sentence s
    query_pos_tags, query_features = process_user_query(s)
    predicted_labels = crf.predict([query_features])[0]

    # convert the predicted labels into standard (token, pos, label) format
    query_tag_list = [(pos_tag[0], pos_tag[1], label) for pos_tag, label in list(zip(query_pos_tags, predicted_labels))]

    # convert into tree
    query_tree = conlltags2tree(query_tag_list)

    # traverse the tree and print labels of subtrees
    labels_dict = {}
    for n in query_tree:
        if isinstance(n, nltk.tree.Tree):
            label = n.label()
            leaves = ' '.join(i[0] for i in n.leaves())
            labels_dict[label] = leaves
    return labels_dict


####### SAMPLE USER QUERY #######
# s1 = 'Can you please show me flights from new york to los angeles arriving before 6 pm'
# s2 = 'I want to see flights from denver to philadelphia departing after 8 pm on monday'

# print("\n", s1)
# pprint.pprint(predict_IOB_labels(s1))

# print("\n", s2)
# pprint.pprint(predict_IOB_labels(s2))











