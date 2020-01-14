#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pickle
import pprint
from sklearn_crfsuite import CRF 
from sklearn_crfsuite import metrics

import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile

import nltk
from nltk.data import find
from nltk import word_tokenize
from nltk.tag.util import untag
from nltk.tree import ParentedTree, Tree
from nltk.corpus import brown, movie_reviews, treebank
from nltk.corpus import wordnet as wn

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def transform_to_dataset(tagged_sentences):
    X, y = [], []
    for tagged in tagged_sentences:
        X.append([features(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])
    return X, y

def load_cate(path):
    categories = []
    for line in open(path):
        tokens = line.strip().split(': ')
        categories.append(tokens[1])
    return categories

def pos_tag(sentence, model):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))

def spo_extractor(pos_tag, cate_list, filt = ['NN','IN','JJ','VB']):
    so_list = []
    p_list = []
    for item in pos_tag:
        if item[1] in ['NN'] and item[0].lower() in cate_list:
            so_list.append(item)
            continue
        if item[1] in filt:
            p_list.append(item)
    return so_list, p_list

def so_extractor(so_list, cate_list):
    return [so_list[0][0].lower(), so_list[1][0].lower()]
    
def construct_pos_list(word2vec_sample_path):
    nltk.download('word2vec_sample')
    word2vec_sample = str(find(word2vec_sample_path))
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    position_list = [
      'left',
      'right',
      'above',
      'below',
      'inside',
      'surrounding']
    x = []
    y = []
    posi_lists = []
    for i in range(len(position_list)):
        tmp_lists = []
        numberofsyn = len(wn.synsets(position_list[i]))
        for j in range(numberofsyn):
            for w in wn.synsets(position_list[i])[j].lemma_names():
                if '_' not in w and w in model:
                    tmp_lists.append(w)
                    x.append(model[w])
                    y.append(i)
        posi_lists.append(list(set(tmp_lists)))
        
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)
    x = np.array(x)
    y = np.array(y)
    clf = svm.SVC()
    clf.fit(x, y)
    return posi_lists, x, y, pca, clf, model
    
def p_extractor(p_list, pos_lists, feat_x, feat_y, pca, clf, wn_model):
    output_pos = ''
    position_list = [
      'left of',
      'right of',
      'above',
      'below',
      'inside',
      'surrounding']
    ## search for the first time
    for idx in range(len(pos_lists)):
        targ_list = pos_lists[idx]
        for item in p_list:
            if item[0] in targ_list:
                output_pos = position_list[idx]
                return [output_pos]
    for i in range(len(p_list)):
        query = p_list[i][0]
        try:
            if i == 0:
                tmp = np.array(wn_model[query])
            else:
                tmp = tmp + np.array(wn_model[query])
        except:   
            continue
    tmp = pca.transform(tmp.reshape(1,-1))
    return [position_list[clf.predict(tmp)[0]]]
            
def sg_constructor(so_list, p_list, sg_dic):
    subj = so_list[0]
    obj = so_list[1]
    pred = p_list[0]
    sub_idx = -1
    obj_idx = -1
    
    sg_dic[0]['objects'].append(subj)
    sub_idx = len(sg_dic[0]['objects'])-1
    
    if obj in sg_dic[0]['objects']:
        obj_idx = sg_dic[0]['objects'].index(obj)
        pass
    else:
        sg_dic[0]['objects'].append(obj)
        obj_idx = len(sg_dic[0]['objects'])-1
    
    
    sg_dic[0]['relationships'].append([sub_idx, pred, obj_idx])
    return sg_dic
            