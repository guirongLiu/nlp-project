import numpy as np 
import pandas as pd 
from IPython import embed
from gensim.models import Word2Vec
import gensim
import nltk
from nltk.data import load
import scipy.spatial as sp
import time
import collections
from featureEngineering import remove_stopwords, character_length_ratio, get_number_of_same_words, get_dictionary_set, syntatic_features
model = gensim.models.KeyedVectors.load_word2vec_format('~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz', binary=True)

class AdvancedFeature(object):
    def __init__(self, fname):
        df = pd.read_csv(fname)
        data = df.values
        self.X = data[:,3:5]
        self.y = data[:6]
        self.word_set = get_dictionary_set()

    def __get_tag_set(self):
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        self.tag_set = list(tagdict.keys())
        

    def __get_hamming_distance(self, matrix1, matrix2):
        diff = np.logical_xor(matrix1, matrix2)
        distance = np.count_nonzero(diff, axis=1)
        return distance
    
    def __get_euclidean_distance(self, matrix1, matrix2):
        distance = ((matrix1 - matrix2) ** 2).sum(axis=1)
        return np.sqrt(distance)

    def __get_cosine_distance(self, matrix1, matrix2):
        cosine = sp.distance.cdist(matrix1, matrix2, 'cosine')
        distance = cosine.diagonal()
        return distance

    def __pair_word_mover_distance(self, sentence1, sentence2):
        distance = model.wmdistance(sentence1, sentence2)
        return distance
    
    def __get_word_mover_distance(self, sentences1, sentences2):
        distance = list(map(self.__pair_word_mover_distance, sentences1, sentences2))
        return distance

    def get_sentence_pair(self):
        sentences1 = list(self.X[:,0])
        sentences2 = list(self.X[:,1])
        sentences1 = list(map(remove_stopwords, sentences1))
        sentences2 = list(map(remove_stopwords, sentences2))
        return sentences1, sentences2
    
    def __get_length_ratio(self, sentences1, sentences2):
        length_ratio = list(map(character_length_ratio, sentences1, sentences2))
        return length_ratio
    
    def __get_num_same_word(self, sentences1, sentences2):
        num_same_word = list(map(get_number_of_same_words, sentences1, sentences2))
        return num_same_word
    
    def __get_sow_matrix(self, sentences):
        n = len(self.word_set)
        sow_matrix = []
        for sent in sentences:
            tokens = set(nltk.word_tokenize(sent))
            vector = [1 if x in tokens else 0 for x in range(n)]
            sow_matrix.append(vector[:])
        return sow_matrix
    
    def __get_bow_matrix(self, sentences):
        n = len(self.word_set)
        bow_matrix = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            counter = collections.Counter(tokens)
            vector = [counter[x] if x in counter else 0 for x in range(n)]
            bow_matrix.append(vector[:])
        return bow_matrix
    
    def __get_syntatic_tags(self, sentences):
        syntatic_tags = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            tags = nltk.pos_tag(tokens)
            syntatic_tags.append(tags)
        return syntatic_tags

    def __get_tag_matrix1(self, sentences):
        syntatic_tags = self.__get_syntatic_tags(sentences)
        matrix = []
        for tags in syntatic_tags:
            tags_set = set(tags)
            vector = [1 if x in tags_set else 0 for x in self.tag_set]
            matrix.append(vector[:])   
        return matrix
    
    def __get_tag_matrix2(self, sentences):
        syntatic_tags = self.__get_syntatic_tags(sentences)
        matrix = []
        for tags in syntatic_tags:
            counter = collections.Counter(tags)
            vector = [counter[x] if x in counter else 0 for x in syntatic_tags]
            matrix.append(vector[:])   
        return matrix

    def __get_distance_sow(self, sentences1, sentences2):
        sow_matrix_1 = self.__get_sow_matrix(sentences1)
        sow_matrix_2 = self.__get_sow_matrix(sentences2)
        distance = self.__get_hamming_distance(np.array(sow_matrix_1), np.array(sow_matrix_2))
        return distance
    
    def create_labeled_data(self, sentences1, sentences2):
        self.__get_tag_set()
        time1 = time.time()

        length_ratio = self.__get_length_ratio(sentences1, sentences2)
        time2 = time.time()
        print('length_ratio finished! time: {}'.format(time2 - time1))
        print(len(length_ratio), length_ratio[:5])

        num_same_word = self.__get_num_same_word(sentences1, sentences2)
        time3 = time.time()
        print('num_same_word finished! time: {}'.format(time3 - time1))
        print(len(num_same_word), num_same_word[:5])

        wdm_distance = self.__get_word_mover_distance(sentences1, sentences2)
        time4 = time.time()
        print('wdm distance finished! time: {}'.format(time4 - time1))
        print(len(wdm_distance), wdm_distance[:5])

        sow_distance = self.__get_distance_sow(sentences1, sentences2)
        time5 = time.time()
        print('sow distance finished! time: {}'.format(time5 - time1))
        print(len(sow_distance), sow_distance[:5])

        bow_matrix_1 = np.array(self.__get_bow_matrix(sentences1))
        bow_matrix_2 = np.array(self.__get_bow_matrix(sentences2))
        print('bow matrix size: ', bow_matrix_1.shape)

        bow_distance_cosine = self.__get_cosine_distance(bow_matrix_1, bow_matrix_2)
        bow_distance_euclidean = self.__get_euclidean_distance(bow_matrix_1, bow_matrix_2)
        time6 = time.time()
        print('bow distance finished! time: {}'.format(time6 - time1))
        print(len(bow_distance_cosine), bow_distance_cosine[:5])
        print(len(bow_distance_euclidean), bow_distance_euclidean[:5])


        tag_matrix_1 = np.array(self.__get_tag_matrix1(sentences1))
        tag_matrix_2 = np.array(self.__get_tag_matrix1(sentences2))
        print('tag matrix size: ', tag_matrix_1.shape)

        tag_distance_hamming = self.__get_hamming_distance(tag_matrix_1, tag_matrix_2)

        tag_counter_matrix_1 = np.array(self.__get_tag_matrix2(sentences1))
        tag_counter_matrix_2 = np.array(self.__get_tag_matrix2(sentences2))
        tag_distance_cosine = self.__get_cosine_distance(tag_counter_matrix_1, tag_counter_matrix_2)
        tag_distance_euclidean = self.__get_euclidean_distance(tag_counter_matrix_1, tag_counter_matrix_2)
        
        time7 = time.time()
        print('tag distance finished! time: {}'.format(time7 - time1))
       
        embed()
        
        



        
        



features = AdvancedFeature('cleaned_data.csv')
sentences1, sentences2= features.get_sentence_pair()
features.create_labeled_data(sentences1, sentences2)
