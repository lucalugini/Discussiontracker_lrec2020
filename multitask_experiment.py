from sklearn.metrics import cohen_kappa_score
import numpy as np
import string, datetime, sys, io, pickle, json, time, subprocess, os
from sklearn.metrics import precision_recall_fscore_support as prfs
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, Dropout, Input, Reshape, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.tokenize import word_tokenize, sent_tokenize
from utils import load_transcript, extract_speciteller_features, extract_dialogue_and_pronoun_features


# INITIAL PARAMETERS
dt_folder = './DiscussionTracker/' # folder containing the DiscussionTracker corpus
glove_file = './glove.6B.50d.txt' # file containing GLOVE word embeddings
experiment_mode = 'multitask' # possible values: argumentation, specificity, multitask

parameters = {}
parameters['class_weights'] = True
parameters['use_static_features'] = True
parameters['arg_weight'] = 1
parameters['spec_weight'] = 1
parameters['col_weight'] = 1
assert(experiment_mode in ['argumentation', 'specificity', 'multitask'])
if experiment_mode == 'argumentation':
    parameters['spec_weight'] = 0
    parameters['col_weight'] = 0
elif experiment_mode == 'specificity':
    parameters['arg_weight'] = 0
    parameters['col_weight'] = 0


print('### Loading data ###')

tenfold_cv = json.load(open(dt_folder + 'DT_10_fold_crossvalidation.json'))
discussions = tenfold_cv['0']['train'] + tenfold_cv['0']['test']
dataframes = {a: load_transcript(dt_folder + a) for a in discussions}


def load_glove(glove_file):
    lines = open(glove_file).readlines()
    glove = {}
    for line in lines:
        l = line.strip().split(' ')
        glove[l[0]] = np.array([float(i) for i in l[1:]])
    return glove

def encode_words(s, base):
    out = []
    for i in s:
        if i.lower() in base:
            out.append(base[i.lower()])
        else:
            out.append(base['unk'])
    return np.array(out)

glove = load_glove(glove_file)

def encode_text_glove(text_train, text_test, glove):
    data_train = []
    for i in text_train:
        data_train.append(encode_words(word_tokenize(i), glove))
    data_train = np.array(data_train)
    data_train = pad_sequences(data_train, maxlen=90, truncating='post')

    data_test = []
    for i in text_test:
        data_test.append(encode_words(word_tokenize(i), glove))
    data_test = np.array(data_test)
    data_test = pad_sequences(data_test, maxlen=90, truncating='post')
    return data_train, data_test

def encode_y(moves_nn, spec_nn, collab_nn):
    y0 = []
    y1 = []
    y2 = []
    for a in moves_nn:
        if a == 'evidence':
        # elif a == 0:
            i = [1, 0, 0]  # evidence
        elif a == 'explanation':
        # elif a == 1:
            i = [0, 1, 0]  # warrant
        # elif a == 2:
        elif a == 'claim':
            i = [0, 0, 1]  # claim
        y0.append(i)

    for b in spec_nn:
        # if b == 0:
        if b == 'low':
            j = [1, 0, 0]  # low
        # elif b == 1:
        elif b == 'med':
            j = [0, 1, 0]  # med
        # elif b == 2:
        elif b == 'high':
            j = [0, 0, 1]  # high
        y1.append(j)

    for c in collab_nn:
        # COLLABORATION
        if c == 'B-new':
            k = [1, 0, 0, 0, 0, 0, 0, 0]
        elif c == 'I-new':
            k = [0, 1, 0, 0, 0, 0, 0, 0]
        elif c == 'B-extension':
            k = [0, 0, 1, 0, 0, 0, 0, 0]
        elif c == 'I-extension':
            k = [0, 0, 0, 1, 0, 0, 0, 0]
        elif c == 'B-challenge':
            k = [0, 0, 0, 0, 1, 0, 0, 0]
        elif c == 'I-challenge':
            k = [0, 0, 0, 0, 0, 1, 0, 0]
        elif c == 'B-agree':
            k = [0, 0, 0, 0, 0, 0, 1, 0]
        elif c == 'I-agree':
            k = [0, 0, 0, 0, 0, 0, 0, 1]
        y2.append(k)

        y = [y0, y1, y2]

    return y


### CROSS VALIDATION
print('### Crossvalidation ###')
ct_cv = 0
argumentation_results = []
specificity_results = []
collaboration_results = []

for i in tenfold_cv:
    print('Fold {}'.format(ct_cv))
    text_train = []
    text_test = []
    speciteller_features_train = []
    speciteller_features_test = []
    spec_train = []
    spec_test = []
    argument_train = []
    argument_test = []
    collaboration_train = []
    collaboration_test = []
    student_train = []
    student_test = []
    move_number = 0
    for a in tenfold_cv[i]['train']:
        disc_tmp = dataframes[a]
        disc = disc_tmp[(disc_tmp.Talk != '') & (disc_tmp.Specificity != '') & (disc_tmp.ArgMove != '')]
        disc_text = list(disc['Talk'])
        text_train += disc_text
        student_train_tmp = list(disc['Student'])
        speciteller_features_train += extract_speciteller_features(disc_text)
        spec_train += list(disc['Specificity'])
        argument_train += list(disc['ArgMove'])
        collaboration_train_tmp = list(disc['Collaboration'])
        for j in range(len(collaboration_train_tmp)):
            # convert collaboration labels to BIO tags
            if student_train_tmp[j] == '':
                student_train_tmp[j] = student_train_tmp[j-1]
            if collaboration_train_tmp[j] != '':
                collaboration_train_tmp[j] = 'B-' + collaboration_train_tmp[j]
            else:
                collaboration_train_tmp[j] = 'I-' + collaboration_train_tmp[j-1][2:]
        student_train_tmp = np.array(student_train_tmp)
        student_train = np.concatenate([student_train, student_train_tmp])
        collaboration_train += collaboration_train_tmp
        move_number += 1

    move_number = 0
    for a in tenfold_cv[i]['test']:
        disc_tmp = dataframes[a]
        disc = disc_tmp[(disc_tmp.Talk != '') & (disc_tmp.Specificity != '') & (disc_tmp.ArgMove != '')]
        disc_text = list(disc['Talk'])
        text_test += disc_text
        student_test_tmp = list(disc['Student'])
        speciteller_features_test += extract_speciteller_features(disc_text)
        spec_test += list(disc['Specificity'])
        argument_test += list(disc['ArgMove'])
        collaboration_test_tmp = list(disc['Collaboration'])
        for j in range(len(collaboration_test_tmp)):
            if student_test_tmp[j] == '':
                student_test_tmp[j] = student_test_tmp[j-1]
            if collaboration_test_tmp[j] != '':
                collaboration_test_tmp[j] = 'B-' + collaboration_test_tmp[j]
            else:
                collaboration_test_tmp[j] = 'I-' + collaboration_test_tmp[j-1][2:]
        student_test_tmp = np.array(student_test_tmp)
        student_test = np.concatenate([student_test, student_test_tmp])
        collaboration_test += collaboration_test_tmp
        move_number += 1
    speciteller_features_train = np.array(speciteller_features_train)
    speciteller_features_test = np.array(speciteller_features_test)
    dialog_features_train, dialog_features_test = extract_dialogue_and_pronoun_features(text_train, text_test)


    data_train, data_test = encode_text_glove(text_train, text_test, glove)
    y_train = encode_y(argument_train, spec_train, collaboration_train)
    y_test = encode_y(argument_test, spec_test, collaboration_test)

    features_train = np.hstack((speciteller_features_train, dialog_features_train))
    features_test = np.hstack((speciteller_features_test, dialog_features_test))
    feats_size = features_train.shape[1]


    train_fold_cv = list(range(len(text_train)))
    test_fold_cv = list(range(len(text_test)))
    ct_cv += 1

    # es = EarlyStopping(monitor='val_arg_model_loss', patience=5, verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    mc = ModelCheckpoint(filepath='model_checkpoint', monitor='val_loss', verbose=1,
                         save_best_only=True, save_weights_only=True)
    epochs = 1#50
    b_size = 128  # batch size

    ##### Convolutional Neural Network
    NB_FILTER = 16
    KERNEL_SIZE = 7

    cnn_inputs = Input(shape=(data_train.shape[1], data_train.shape[2], 1))
    feat_inputs = Input(shape=(feats_size,))

    # 1st layer
    first_layer = Conv2D(NB_FILTER, KERNEL_SIZE, activation='relu', padding='same')(cnn_inputs)
    first_layer = MaxPooling2D(pool_size=(3, 1))(first_layer)
    first_layer = Dropout(rate=0.4)(first_layer)
    # 2nd layer
    second_layer = Conv2D(NB_FILTER, KERNEL_SIZE, activation='relu', padding='same')(first_layer)
    second_layer = MaxPooling2D(pool_size=(3, 1))(second_layer)
    second_layer = Dropout(rate=0.4)(second_layer)
    # 3rd layer
    third_layer = Conv2D(NB_FILTER, KERNEL_SIZE, activation='relu', padding='same')(second_layer)
    third_layer = MaxPooling2D(pool_size=(3, 1))(third_layer)
    third_layer = Dropout(rate=0.4)(third_layer)

    # flatten
    flatten_layer = Flatten(name='move_flatten')(third_layer)

    # final embedding
    argmove_embedding = concatenate([flatten_layer, feat_inputs])

    model_argumentation = Dense(3, activation='softmax', name='arg_model')(flatten_layer)
    model_specificity = Dense(3, activation='softmax', name='spec_model')(flatten_layer)
    model_collaboration = Dense(8, activation='softmax', name='col_model')(flatten_layer)

    model = Model(inputs=[cnn_inputs, feat_inputs], outputs=[model_argumentation, model_specificity, model_collaboration])

    model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[parameters['arg_weight'], parameters['spec_weight'], parameters['col_weight']], metrics=['categorical_accuracy'])
    model.summary()
        

    x_train = data_train
    x_test = data_test
    x_train_feat = features_train
    x_test_feat = features_test


    if parameters['use_static_features']:
        x_train_feat = features_train
        x_test_feat = features_test

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    
    cw = {'arg_model':{0:1, 1:1, 2:1}, 'spec_model':{0:1, 1:1, 2:1}, 'col_model':{0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1}}
    if parameters['class_weights']:
        cw['arg_model'] = {0:2, 1:4, 2:1}
    
    
    hist = model.fit([x_train, x_train_feat], [y_train[0], y_train[1], y_train[2]], batch_size=b_size, epochs=epochs,
                    class_weight=cw, verbose=2, validation_split=0.1, callbacks=[es, mc])
    
    history = hist.history

    model.load_weights('model_checkpoint')

    tmp_pred = model.predict([x_test, x_test_feat])
    test_predictions = np.array([[np.argmax(j) for j in k] for k in tmp_pred]).T
    test_labels = np.array([[np.argmax(j) for j in k] for k in y_test]).T
    
    argumentation_results.append([cohen_kappa_score(test_labels[:,0], test_predictions[:,0])] +
                                list(prfs(test_labels[:,0], test_predictions[:,0], average='macro')[:-1]))
    specificity_results.append([cohen_kappa_score(test_labels[:,1], test_predictions[:,1], weights='quadratic')] +
                                list(prfs(test_labels[:,1], test_predictions[:,1], average='macro')[:-1]))
    collaboration_results.append([cohen_kappa_score(test_labels[:,2], test_predictions[:,2])] +
                                list(prfs(test_labels[:,2], test_predictions[:,2], average='macro')[:-1]))
    

# print results
print('Results:')
print(argumentation_results)
print(specificity_results)
print(collaboration_results)
print('Argumenentation - kappa, precision, recall, F-score: {}'.format(np.average(np.array(argumentation_results),axis=0)))
print('Specificity - qwkappa, precision, recall, F-score: {}'.format(np.average(np.array(specificity_results),axis=0)))
print('Collaboration - kappa, precision, recall, F-score: {}'.format(np.average(np.array(collaboration_results),axis=0)))

print('### Experiment complete ###')
