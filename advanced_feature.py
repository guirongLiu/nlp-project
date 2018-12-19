import numpy as np 
import pandas as pd 
from IPython import embed
from gensim.models import Word2Vec
import gensim
import nltk
from nltk.data import load
import scipy.spatial as sp
import time
import os
import collections
import pickle
from tqdm import tqdm
from featureEngineering import remove_stopwords, character_length_ratio, get_number_of_same_words, get_dictionary_set, syntatic_features
model = gensim.models.KeyedVectors.load_word2vec_format('~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz', binary=True)
import multiprocessing
f = open('syn_set', 'rb')
tag_set = pickle.load(f)
f.close()
class AdvancedFeature(object):
    def __init__(self, fname):
        df = pd.read_csv(fname)
        data = df.values
        self.X = data[:,3:5]
        self.y = data[:,5]
        f = open('wordset', 'rb')
        self.word_set = pickle.load(f)
        f.close()

    def get_tag_set(self):
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        self.tag_set = list(tagdict.keys())
        

    def get_hamming_distance(self, matrix1, matrix2):
        diff = np.logical_xor(matrix1, matrix2)
        distance = np.count_nonzero(diff, axis=1)
        return distance
    
    def get_euclidean_distance(self, matrix1, matrix2):
        distance = ((matrix1 - matrix2) ** 2).sum(axis=1)
        return np.sqrt(distance)

    def get_cosine_distance(self, matrix1, matrix2):
        cosine = sp.distance.cdist(matrix1, matrix2, 'cosine')
        distance = cosine.diagonal()
        return distance

    def pair_word_mover_distance(self, sentence1, sentence2):
        distance = model.wmdistance(sentence1, sentence2)
        return distance
    
    def get_word_mover_distance(self, sentences1, sentences2):
        # distance = []
        # distance = pool.map(self.__pair_word_mover_distance,sentences1, sentences2)
        distance = list(map(self.pair_word_mover_distance, sentences1, sentences2))
        return distance

    def get_sentence_pair(self):
        sentences1 = list(self.X[:,0])
        sentences2 = list(self.X[:,1])
        sentences1 = list(map(remove_stopwords, sentences1))
        sentences2 = list(map(remove_stopwords, sentences2))
        return sentences1, sentences2
    
    def get_length_ratio(self, sentences1, sentences2):
        length_ratio = list(map(character_length_ratio, sentences1, sentences2))
        return length_ratio
    
    def get_num_same_word(self, sentences1, sentences2):
        num_same_word = list(map(get_number_of_same_words, sentences1, sentences2))
        return num_same_word
    
    def get_sow_matrix(self, sentences):
        n = len(self.word_set)
        sow_matrix = []
        for sent in sentences:
            tokens = set(nltk.word_tokenize(sent))
            vector = [1 if x in tokens else 0 for x in self.word_set]
            sow_matrix.append(vector[:])
        return sow_matrix
    
    def get_bow_matrix(self, sentences):
        n = len(self.word_set)
        bow_matrix = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            counter = collections.Counter(tokens)
            vector = [counter[x] if x in counter else 0 for x in self.word_set]
            bow_matrix.append(vector[:])
        return bow_matrix
    
   
    # def get_syntatic_tags(self, sentences):
    #     syntatic_tags = []
    #     for sent in sentences:
    #         tokens = nltk.word_tokenize(sent)
    #         tags = nltk.pos_tag(tokens)
    #         syntatic_tags.append(tags)
    #     return syntatic_tags
    
    def get_syntatic_tags(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        return tags
    
    def get_syntatic_tag_matrix1(self, tags):
        tags_set = set(tags)
        vector = [1 if x in tags_set else 0 for x in tag_set]
        return vector
    
    def get_syntatic_tag_matrix2(self, tags):
        counter = collections.Counter(tags)
        vector = [counter[x] if x in counter else 0 for x in tag_set]
        return vector

   

    # def get_distance_sow(self, sentences1, sentences2):
    #     sow_matrix_1 = self.get_sow_matrix(sentences1)
    #     sow_matrix_2 = self.get_sow_matrix(sentences2)
    #     distance = self.get_hamming_distance(np.array(sow_matrix_1), np.array(sow_matrix_2))
    #     return distance
    

        
        

if __name__ == '__main__':
    features = AdvancedFeature('cleaned_data.csv')
    if not os.path.isfile('sentences.pkl'):
        sentences1, sentences2= features.get_sentence_pair()
        f = open('sentences.pkl', 'wb')
        pickle.dump((sentences1, sentences2), f, 0)
        f.close()

    else:
        f = open('sentences.pkl', 'rb')
        data = pickle.load(f)
        sentences1, sentences2 = data
        f.close()
    """
    length_ratio
    """
    if os.path.isfile('length_ratio'):
        with open('length_ratio', 'rb') as f:
            length_ratio = pickle.load(f)
    else:
        length_ratio = features.get_length_ratio(sentences1, sentences2)
        with open('length_ratio', 'wb') as f:
            pickle.dump(length_ratio, f, 0)
    print('get length ratio!')
    """
    num_same_word
    """
    if os.path.isfile('num_same_word'):
        with open('num_same_word', 'rb') as f:
            num_same_word = pickle.load(f)
    else:
        num_same_word = features.get_num_same_word(sentences1, sentences2)
        with open('num_same_word', 'wb') as f:
            pickle.dump(num_same_word, f, 0)
    print('get num of same word!')   
    """
    word mover distance
    """
    if os.path.isfile('wmd'):
        with open('wmd', 'rb') as f:
            wmdistance = pickle.load(f)
    else:
        wmdistance = features.get_word_mover_distance(sentences1, sentences2)
        with open('wmd', 'wb') as f:
            pickle.dump(wmdistance, f, 0)
    print('get word mover distance!')
    """
    syntactic tag hamming distance
    """
    if os.path.isfile('hamming_distance_for_tag'):
        with open('hamming_distance_for_tag', 'rb') as f:
            hamming_distance_for_tag = list(pickle.load(f))
    else:

        if os.path.isfile('syntatic_tags1'):
            f = open('syntatic_tags1', 'rb')
            syntatic_tags1 = pickle.load(f)
            syntatic_tags1 = [[x[1] for x in tag] for tag in syntatic_tags1]
            f.close()
            f = open('syntatic_tags2', 'rb')
            syntatic_tags2 = pickle.load(f)
            syntatic_tags2 = [[x[1] for x in tag] for tag in syntatic_tags2]
            f.close()
        else:
            pool = multiprocessing.Pool()
            syntatic_tags1 = pool.map(features.get_syntatic_tags, sentences1)
            syntatic_tags2 = pool.map(features.get_syntatic_tags, sentences2)
            pool.close()
            pool.join()
            f = open('syntatic_tags1', 'wb')
            pickle.dump(syntatic_tags1, f, 0)
            f.close()

            f = open('syntatic_tags2', 'wb')
            pickle.dump(syntatic_tags2, f, 0)
            f.close()
        
        if os.path.isfile('tag_matrix'):
            with open('tag_matrix', 'rb') as f:
                data = pickle.load(f)
                tag_matrix_1, tag_matrix_2 = data
        else:
            pool = multiprocessing.Pool()
            tag_matrix_1 = pool.map(features.get_syntatic_tag_matrix1, syntatic_tags1)
            tag_matrix_2 = pool.map(features.get_syntatic_tag_matrix1, syntatic_tags2)
            pool.close()
            pool.join()

        hamming_distance_for_tag = features.get_hamming_distance(tag_matrix_1, tag_matrix_2)
        f = open('hamming_distance_for_tag', 'wb')
        pickle.dump(hamming_distance_for_tag, f, 0)
        f.close()
    print('get tag hammimng distance!')
    
    """
    cosine distance and euclidean distance for syntatic tag counting
    """
    if os.path.isfile('tag_count_euclidean'):
        with open('tag_count_euclidean', 'rb') as f1, open('tag_count_cosine', 'rb') as f2:
            distance_cosine_tag_count = pickle.load(f1)
            distance_euclidean_tag_count = pickle.load(f2)
    else:
        if os.path.isfile('tag_matrix_count'):
            f = open('tag_matrix_count', 'rb')
            data = pickle.load(f)
            tag_matrix_count_1, tag_matrix_count_2 = data
            f.close()
        else:
            tag_matrix_count_1 = pool.map(features.get_syntatic_tag_matrix2, syntatic_tags1)
            tag_matrix_count_2 = pool.map(features.get_syntatic_tag_matrix2, syntatic_tags2)
            f = open('tag_matrix_count', 'wb')
            pickle.dump((tag_matrix_count_1, tag_matrix_count_2), f, 0)
            f.close()
        
        distance_cosine_tag_count = []
        distance_euclidean_tag_count = []
        n = len(tag_matrix_count_1)
        for i in tqdm(range(0,n, 1000)):
            tags1 = tag_matrix_count_1[i:i+1000]
            tags2 = tag_matrix_count_2[i:i+1000]
            cosine = features.get_cosine_distance(np.array(tags1), np.array(tags2))
            euclidean = features.get_cosine_distance(np.array(tags1), np.array(tags2))
            distance_cosine_tag_count += list(cosine)
            distance_euclidean_tag_count += list(euclidean)
        
        with open('tag_count_cosine', 'wb') as f1, open('tag_count_euclidean', 'wb') as f2:
            pickle.dump(distance_cosine_tag_count, f1, 0)
            pickle.dump(distance_euclidean_tag_count, f2, 0)

    print('get tag count cosine and euclidean distance!')


    if os.path.isfile('sow_hamming'):
        with open('sow_hamming','rb') as f1, open('bow_cosine', 'rb') as f2, open('bow_euclidean', 'rb') as f3:
            hamming_distance = pickle.load(f1)
            cosine_distance_bow = pickle.load(f2)
            euclidean_distance_bow = pickle.load(f3)
    else:
        n = len(sentences1)
        hamming_distance = []

        for i in tqdm(range(0,n, 100)):
            subset1 = sentences1[i:i+100]
            sub_matrix1 = np.array(features.get_sow_matrix(subset1))
            subset2 = sentences2[i:i+100]
            sub_matrix2 = np.array(features.get_sow_matrix(subset2))
            hamming = features.get_hamming_distance(sub_matrix1, sub_matrix2)
            hamming_distance += list(hamming)

        f = open('sow_hamming', 'wb')
        pickle.dump(hamming_distance, f, 0)
        f.close()
        print('get SOW hamming distance!')



        cosine_distance_bow = []
        euclidean_distance_bow = []
        for i in tqdm(range(0,n, 100)):
            subset1 = sentences1[i:i+100]
            sub_matrix1 = np.array(features.get_bow_matrix(subset1))
            subset2 = sentences2[i:i+100]
            sub_matrix2 = np.array(features.get_bow_matrix(subset2))
            cosine = features.get_cosine_distance(sub_matrix1, sub_matrix2)
            cosine_distance_bow+= list(cosine)

            euclidean = features.get_euclidean_distance(sub_matrix1, sub_matrix2)
            euclidean_distance_bow += list(euclidean)
        
        f = open('bow_cosine', 'wb')
        pickle.dump(cosine_distance_bow, f, 0)
        f.close()
        f = open('bow_euclidean', 'wb')
        pickle.dump(euclidean_distance_bow, f, 0)
        f.close()

    print('get BOW distance!')
    data = {
        'length_ratio': length_ratio,
        'num_same_word': num_same_word,
        'wmd': wmdistance,
        'hamming_tag': hamming_distance_for_tag,
        'distance_cosine_tag_count': distance_cosine_tag_count,
        'distance_euclidean_tag_count': distance_euclidean_tag_count,
        'sow_hamming_distance': hamming_distance,
        'bow_cosine_distance':cosine_distance_bow,
        'bow_euclidean_distance': euclidean_distance_bow, 
        'label': list(features.y)
    }
    save = pd.DataFrame(data)
    save.to_csv('new_data.csv')






