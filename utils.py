from nltk.tokenize import RegexpTokenizer
import nltk
import  pandas as pd
import numpy as np
import re
import pickle
import json
import statistics
from speciteller import get_speciteller_features

# Copy the following function in speciteller.py
# def get_speciteller_features(fin):
#     '''Given a list of strings, return the speciteller features (shallow feature set and embeddings)'''
#     aligner = ModelNewText(brnspace,brnclst,embeddings)
#     aligner.loadFromFile(fin)
#     aligner.fShallow()
#     aligner.fNeuralVec()
#     aligner.fBrownCluster()
#     y,xs = aligner.transformShallow()
#     _,xw = aligner.transformWordRep()
#     xs,_ = simpleScale(xs,scales_shallow)
#     xw,_ = simpleScale(xw,scales_neuralbrn)
#     data = []
#     for i in range(len(y)):
#         if len(xs[i].values()) > 0 and len(xw[i].values()) > 0:
#             data.append(list(xs[i].values())+list(xw[i].values())[:100])
#         else:
#             continue
#     return data

word_tokenizer = RegexpTokenizer(r'\w+')

def extract_speciteller_features(a):
    return get_speciteller_features(a)

def get_pronouns(s):
    taglist = nltk.pos_tag(nltk.word_tokenize(s))
    return sum([1 for a in taglist if 'PRP' in a[1]])

def get_sentence_length(s):
    tokens = word_tokenizer.tokenize(s)
    return len(tokens)

def get_lexical_features(train_corpus, test_corpus):
    # extract TF-IDF for unigrams and bigrams
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=5)
    tmp = vectorizer.fit_transform(train_corpus)
    x = tmp.toarray().tolist()
    
    # now compute aggregate statistics
    for i in range(len(x)):
        min_tmp = min(x[i])
        max_tmp = max(x[i])
        mean_tmp = sum(x[i])/len(x[i])
        x[i] += [min_tmp, max_tmp, mean_tmp]
    
    tmp2 = vectorizer.transform(test_corpus)
    x2 = tmp2.toarray().tolist()
    # now compute aggregate statistics
    for i in range(len(x2)):
        min_tmp = min(x2[i])
        max_tmp = max(x2[i])
        mean_tmp = sum(x2[i])/len(x2[i])
        x2[i] += [min_tmp, max_tmp, mean_tmp]
    return [x, x2]

def get_syntax_features(train_corpus, test_corpus):
    # extract POS unigrams, bigrams and trigrams
    x = []    
    for s in train_corpus:    
        tags = nltk.pos_tag(nltk.word_tokenize(s))
        x.append(' '.join([a[1] for a in tags]))
    
    n = 3 # range of n-grams
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,n), lowercase=False, use_idf=False)
    vec = vectorizer.fit_transform(x)
    
    # features for training set
    f_train = vec.toarray().tolist()
    
    # features for test set
    x = []    
    for s in test_corpus:    
        tags = nltk.pos_tag(nltk.word_tokenize(s))
        x.append(' '.join([a[1] for a in tags]))
    vec = vectorizer.transform(x)
    f_test = vec.toarray().tolist()
    
    return [f_train, f_test]

def get_word_stats(s):
    # compute min length, max length, mean length, median length,
    # and frequency of word lengths between 1 and 20 (and over)
    if len(s) != 0:
        tokens = nltk.word_tokenize(s)
        len_list = [len(a) for a in tokens]
        min_len = min(len_list)
        max_len = max(len_list)
        mean = sum(len_list)/len(len_list)
        median = statistics.median(len_list)
        word_lengths = []    
        for i in range(1,20):
            word_lengths.append(sum([1 for a in len_list if a == i]))
        word_lengths.append(sum([1 for a in len_list if a >= 20]))
        return [min_len, max_len, mean, median]+word_lengths
    else:
        return [0 for i in range(24)]

def get_pronouns_separate(s):
    s2 = [a.lower() for a in nltk.word_tokenize(s)]
    first_person = ['i','me','my','mine','we','our','ours','us']
    second_person = ['you','your','yours']
    third_person = ['he','she','it','his','hers','him','her','its','they','them','their','theirs']
    return [sum([1 for a in s2 if a in first_person]), sum([1 for a in s2 if a in second_person]), sum([1 for a in s2 if a in third_person])]

def extract_dialogue_and_pronoun_features(train_corpus, test_corpus):
    b = get_lexical_features(train_corpus, test_corpus)
    c = get_syntax_features(train_corpus, test_corpus)
    a = []
    for s in train_corpus:
        i = get_pronouns(s)
        j = get_sentence_length(s)
        k = get_word_stats(s)
        a.append([i,j]+k)
    d = []
    for s in test_corpus:
        i = get_pronouns(s)
        j = get_sentence_length(s)
        k = get_word_stats(s)
        d.append([i,j]+k)
    
    x_train = []
    for i in range(len(train_corpus)):
        x_train.append(a[i] + b[0][i] + c[0][i] + get_pronouns_separate(train_corpus[i]))
    x_test = []
    for i in range(len(test_corpus)):
        x_test.append(d[i] + b[0][i] + c[0][i] + get_pronouns_separate(test_corpus[i]))
    
    return [x_train, x_test]


def collaboration_converter(a):
    if 'non' in a.lower():
        return 'Non'
    elif 'n' in a.lower():
        return 'new'
    elif 'c' in a.lower():
        return 'challenge'
    elif 'e' in a.lower():
        return 'extension'
    elif 'a' in a.lower():
        return 'agree'
    

def student_converter(a):
    if len (a) > 0:
        if a[0].lower() == 't' or a.lower() == 'teacher':
            return 'teacher'
        elif "?" in a:
            return ""
        else:
            return re.sub("[^0-9]","",a)
    else:
        return ''
        

def turn_converter(a):
    a = a.strip()
    if len(a.split(' ')) > 1:
        a = a.split(' ')[0]
    if len(a.split(',')) > 1:
        a = a.split(',')[0]
    return a.split('.')[-1].strip()

def talk_converter(a):
    if isinstance(a, str):
        return a.strip()
    else:
        return str(a).strip()

def load_transcript(file):
    '''Given a transcript filename, read the transcript and return a Pandas dataframe.'''
    converter = {'Claim': (lambda x: 'claim' if str(x).lower()=='x' else ''),
                 'Evidence': (lambda x: 'evidence' if str(x).lower()=='x' else ''),
                 'Warrant': (lambda x: 'explanation' if str(x).lower()=='x' else ''),
                 'Low': (lambda x: 'low' if str(x).lower()=='x' else ''),
                 'Med': (lambda x: 'med' if str(x).lower()=='x' else ''),
                 'High': (lambda x: 'high' if str(x).lower()=='x' else ''),
                 'Collaboration Code': (lambda x: collaboration_converter(x) if x else ''),
                 'Disc id': turn_converter,
                 'Turn of Reference': turn_converter,
                 'Argument Segmentation': talk_converter,
                 'Sp id': student_converter,
                 'Talk': talk_converter
                 }
 

    d = pd.read_excel(file, header=0, usecols=range(21), keep_default_na=False, converters=converter)
    d['Argmove'] = d.Claim.str.cat(d.Evidence).str.cat(d.Warrant)
    d['Specificity'] = d['Low'].str.cat(d['Med']).str.cat(d['High'])
    cols = ['Disc id', 'Sp id', 'Talk', 'Argument Segmentation', 'Argmove', 'Specificity', 'Collaboration Code', 'Turn of Reference']

    data = pd.DataFrame.copy(d)
    data = data[cols]
    data = data.replace('', np.nan)
    data = data.dropna(axis=0, how='all').replace(np.nan, '')
    data.rename(index=str, columns={'Disc id': 'Turn', 'Sp id':'Student', 'Talk':'Turn_text', 'Argument Segmentation':'Talk', 'Argmove':'ArgMove',
                                    'Collaboration Code':'Collaboration', 'Turn of Reference':'Reference'}, inplace=True)
    return data
