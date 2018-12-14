import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import pandas as pd

# data set up
df = pd.read_csv('train.csv', encoding="ISO-8859-1")

#######################################################################################################
# feature 1: questions without stopwords:

def remove_stopwords(sent):
    tokens=[token.lower() for token in nltk.word_tokenize(sent)]
    stopwords_list=list(stopwords.words("english"))
    return " ".join([token for token in tokens if token not in stopwords_list])
#######################################################################################################

# feature 2: character length ratio
def character_length_ratio(sent1,sent2):
    """
    params: two question sentences sent1 and sent2
    return: ratio
    """
    return sent1/max(1, sent2)
#######################################################################################################
# feature 3: number of same words or synonyms
"""
The method get_number_of_same_words(sent1, sent2) can return the number of same or synonyms words according to the wordnet.
Forexaple, we have sent1-"cat dog is?" and sent2-"good cat equals dog.". 
This method will return 3, since it consider "cat" is "cat", "dog" is "dog", and "equals" is synonym of "is". 
"""

def get_synonyms(word):
    """
    params: word
    return: list of synonyms of this word
    """
    synonyms=[]
    synsets=wn.synsets(word)
    if synsets:
        for synset in synsets:
            for lemma in synset.lemmas():
                synonyms.append(lemma.name())
    return list(set(synonyms))


def get_synonyms_set(sent):
    tokens = nltk.word_tokenize(sent)
    syn = [tokens]
    for token in tokens:
        if get_synonyms(token):
            syn.append(get_synonyms(token))
    return set([y for x in syn for y in x])


def get_number_of_same_words(sent1,sent2):
    count=0
    syn_set_sent1=get_synonyms_set(sent1)
    tokens=nltk.word_tokenize(sent2)
    for token in tokens:
        syn=get_synonyms(token)+[token]
        for s in syn:
            if s in syn_set_sent1:
                count+=1
                break
    return count


get_number_of_same_words("cat dogs is?", "good cat equals dog.")
#######################################################################################################
# SOW model
"""
In this model, we ignore the order of text words, grammar and syntax, 
but only record whether a word appears in the text. To be more specific, 
we have a set, which has words that is in corpus, and we mark the length of the set as  V. 
We also mark V as the length of the word vector. So for each word, 
the value of the word at the corresponding position is 1, and others remain 0. 
For example, we have a dictionary set (“this”,”cat”,”dog”,”wants”,”play”,”to”,”do”,the”,”house”) 
and we have a text “This cat wants to play”. Therefore, the word vector is [1,1,0,1,1,1,0,0,0].
"""

def get_dictionary_set():
    """
    :return: set of tokens
    """
    questions_set = set(df['question1']).union(df['question2'])
    words_set = set()
    for q in questions_set:
        tokens = nltk.word_tokenize(q)
        for token in tokens:
            words_set.add(token)
    return list(words_set)


def get_sow_vector(sent):
    """
    :param:string sent
    :return: word vecotor - list of 1s and 0s
    """
    dict_list = get_dictionary_set()
    tokens = nltk.word_tokenize(sent)
    vector = [0]*len(dict_list)
    for token in tokens:
        if token in dict_list:
            index = dict_list.index[token]
            if vector[index] == 0:
                vector[index] = 1
    return vector



#######################################################################################################
# BOW model

def get_bow_vector(sent):
    """
    BOW model
    This model is similar with SOW, but we also consider the occurrence of each word.
    For example, we have a dictionary set (“This”,”cat”,”and”,”that”,”dog”,”want”,”play”,”to”,”do”,the”,”house”)
    and we have a text “This cat and that cat want to play”, the word vector is [1,2,1,1,0,1,1,1,0,0,0].
    :param: string sent
    :return: vector-list of numbers

    """
    dict_list = list(get_dictionary_set())
    tokens = nltk.word_tokenize(sent)
    vector = [0]*len(dict_list)
    for token in tokens:
        if token in dict_list:
            index = dict_list.index[token]
            vector[index] += 1
    return vector
#######################################################################################################
# feature 6 : Syntatic features


def syntatic_features(sent):
    tokens = nltk.word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    # more code here
    return tags
#######################################################################################################


def feature_extract(sent1, sent2):
    """
    :param sent1: sent of question1 as a string
    :param sent2: sent of question2 as a string
    :return: dictionary-keys are names of features, values are features
    """
    dict={}
    dict["remove_stopwords"]=remove_stopwords(sent1)
    dict["character_length_ratio"]=character_length_ratio(sent1, sent2)
    dict["syntatic_features"]=syntatic_features(sent1)
    dict["get_number_of_same_words"]=get_number_of_same_words(sent1, sent2)
    return dict

