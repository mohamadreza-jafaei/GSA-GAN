import scipy.io as io
import pickle as cp
import numpy as np
import pandas as pd
from keras.engine.saving import load_model
from keras.utils import to_categorical
from sklearn import preprocessing, decomposition
from numpy.lib.stride_tricks import as_strided as ast
import itertools
import operator
import os

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

data_files_subject1 = [
    # subject 1
    './DataSet/Dataset Annotation Labels/Subject1/repetition 2/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 2/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 2/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 2/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 2/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject1/repetition 3/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 3/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 3/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 3/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 3/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject1/repetition 4/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 4/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 4/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 4/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 4/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject1/repetition 5/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 5/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 5/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject1/repetition 5/MT_00200493-000-000_00342488.csv',
    './DataSet/Subject1/repetition 5/MT_00200493-000-000_00342486.csv']

data_files_subject2 = [
    # subject 2
    './DataSet/Subject2/repetition 1/MT_00200493-000-000_00342485.csv',
    './DataSet/Subject2/repetition 1/MT_00200493-000-000_00342489.csv',
    './DataSet/Subject2/repetition 1/MT_00200493-000-000_00342490.csv',
    './DataSet/Subject2/repetition 1/MT_00200493-000-000_00342488.csv',
    './DataSet/Subject2/repetition 1/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject2/repetition 2/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject2/repetition 2/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject2/repetition 2/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject2/repetition 2/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject2/repetition 2/MT_00200493-000-000_00342486.csv']

data_files_subject3 = [
    # subject 3
    './DataSet/Dataset Annotation Labels/Subject3/repetition 2/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 2/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 2/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 2/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 2/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject3/repetition 3/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 3/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 3/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 3/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 3/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject3/repetition 4/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 4/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 4/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 4/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 4/MT_00200493-000-000_00342486.csv',

    './DataSet/Dataset Annotation Labels/Subject3/repetition 5/MT_00200493-000-000_00342485.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 5/MT_00200493-000-000_00342489.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 5/MT_00200493-000-000_00342490.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 5/MT_00200493-000-000_00342488.csv',
    './DataSet/Dataset Annotation Labels/Subject3/repetition 5/MT_00200493-000-000_00342486.csv']

data_files_subject5 = [
    # subject 5

    '../Dataset Annotation Labels/Subject5/repetition 1/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject5/repetition 1/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject5/repetition 1/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject5/repetition 1/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject5/repetition 1/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject5/repetition 2/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject5/repetition 2/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject5/repetition 2/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject5/repetition 2/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject5/repetition 2/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject5/repetition 3/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject5/repetition 3/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject5/repetition 3/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject5/repetition 3/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject5/repetition 3/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject5/repetition 4/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject5/repetition 4/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject5/repetition 4/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject5/repetition 4/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject5/repetition 4/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject5/repetition 5/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject5/repetition 5/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject5/repetition 5/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject5/repetition 5/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject5/repetition 5/MT_00200493-000-000_00342486.csv']

data_files_subject6 = [
    # subject 6
    '../Dataset Annotation Labels/Subject6/repetition 2/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject6/repetition 2/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject6/repetition 2/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject6/repetition 2/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject6/repetition 2/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject6/repetition 3/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject6/repetition 3/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject6/repetition 3/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject6/repetition 3/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject6/repetition 3/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject6/repetition 4/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject6/repetition 4/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject6/repetition 4/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject6/repetition 4/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject6/repetition 4/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject6/repetition 5/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject6/repetition 5/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject6/repetition 5/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject6/repetition 5/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject6/repetition 5/MT_00200493-000-000_00342486.csv']

# subject 7
data_files_subject7 = [
    '../Dataset Annotation Labels/Subject7/repetition 1/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject7/repetition 1/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject7/repetition 1/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject7/repetition 1/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject7/repetition 1/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject7/repetition 2/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject7/repetition 2/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject7/repetition 2/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject7/repetition 2/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject7/repetition 2/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject7/repetition 3/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject7/repetition 3/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject7/repetition 3/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject7/repetition 3/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject7/repetition 3/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject7/repetition 4/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject7/repetition 4/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject7/repetition 4/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject7/repetition 4/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject7/repetition 4/MT_00200493-000-000_00342486.csv',

    '../Dataset Annotation Labels/Subject7/repetition 5/MT_00200493-000-000_00342485.csv',
    '../Dataset Annotation Labels/Subject7/repetition 5/MT_00200493-000-000_00342489.csv',
    '../Dataset Annotation Labels/Subject7/repetition 5/MT_00200493-000-000_00342490.csv',
    '../Dataset Annotation Labels/Subject7/repetition 5/MT_00200493-000-000_00342488.csv',
    '../Dataset Annotation Labels/Subject7/repetition 5/MT_00200493-000-000_00342486.csv']


# __________________________________________________
def load_data(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()
    return data


# __________________________________________________
def read_sensors_data(subject_number, label_level=5, remove_time=2):
    """ @:param remove_time : set it 0 to include time columns (0,1) put it 2 to exclude time"""
    # 00342485 right arm
    # 00342489 left arm
    # 00342490 right leg
    # 00342488 left leg
    # 00342486 chest
    data_files = [
        # subject 1
        'MT_00200493-000-000_00342485.csv',
        'MT_00200493-000-000_00342489.csv',
        'MT_00200493-000-000_00342490.csv',
        'MT_00200493-000-000_00342488.csv',
        'MT_00200493-000-000_00342486.csv'
    ]

    data = []
    print('Loading dataset files ...')
    for repetition in range(1, 6):  # because we have 5 repetition
        rep_tmp = []
        label_f_name = '../Subject' + str(subject_number) + '/repetition ' + str(repetition) + '/Subject ' + str(
            subject_number) + '_Rep' + str(repetition) + '.' + '.csv'
        label_tmp = np.genfromtxt(label_f_name, delimiter=',', dtype=str, comments='//')[1:, :]
        for filename in data_files:
            try:
                fn = '../Subject' + str(subject_number) + '/repetition ' + str(repetition) + '/' + filename
                print('Loading ' + fn)
                tmp = np.loadtxt(fn, delimiter=',', dtype=float, comments='//', skiprows=5, encoding='utf-8')
                if rep_tmp == []:
                    rep_tmp = tmp
                else:
                    rep_tmp = np.concatenate((rep_tmp, tmp), axis=1)
            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

        if data==[]:
            label = label_tmp
            data = rep_tmp[0:len(label_tmp), :]

        else:
            data = np.concatenate((data, rep_tmp[0:len(label_tmp), :]), axis=0)
            label = np.concatenate((label, label_tmp), axis=0)
    # Select columns

    data = remove_nan(data)
    # data = select_columns(data)
    return data[:, remove_time:], label[:, label_level]


# __________________________________________
def select_columns(data):
    # # Select accelerometers data only
    # selected_features = np.asarray( [2,3,4,5,6,7,8,9,10,18,19,20,23,24,25,26,27,28,29,30,31,39,40,41,44,
    #                                  45,46,47,48,49,50,51,52,60,61,62,65,66,67,
    #                                  68,69,70,71,72,73,81,82,83,86,87,88,89,90,
    #                                  91,92,93,94,102,103,104] )

    selected_features = np.asarray([2, 3, 4, 13, 14, 15, 18, 19, 20])
    # 23,24,25,33,34,35,39,40,41,
    # 45,45,46,54,55,56,60,61,62,
    # 65,66,67,75,76,77,81,82,83,
    # 86,87,88,96,97,98,102,103,104] )

    return data[:, selected_features]


# ____________________________________________
def remove_nan(data):
    data[np.isnan(data)] = 0
    # later we can replace NaN with most common one
    return data


# ___________________________________________
def make_label(label):
    le = preprocessing.LabelEncoder()
    encoded_lbl = le.fit_transform(label)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    return encoded_lbl, mapping


# __________________________________________
def normalize_data(data):
    """Normalizes all sensor channels
    :param data: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    max_list = np.max(data, axis=0)
    min_list = np.min(data, axis=0)
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        if diffs[i] == 0:
            diffs[i] = 1
        data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
    # Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


# __________________________________________
def normalize_data_pampa2(data):
    """Normalizes all sensor channels
    :param data: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    max_list = np.zeros(data.shape[1])
    min_list = np.zeros(data.shape[1])
    for i in np.arange(data.shape[1]):
        max_list[i] = max(data[:, i])
        min_list[i] = min(data[:, i])

    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        if diffs[i] == 0:
            diffs[i] = 1
        data[:, i] = (data[:, i] - min_list[i]) / diffs[i]
    # Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


# __________________________________________
def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
    Returns
        an array containing each n-dimensional window from a
    """

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = list(filter(lambda i: i != 1, dim))
    slides = strided.reshape(dim)
    reShapedSlides = []
    for i in range(len(slides)):
        reShapedSlides.append(np.ravel(slides[i]))
    return np.array(reShapedSlides)


# __________________________________________________________


def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
    Parameters
        shape - an int, or a tuple of ints
    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


# ________________________________________________________________
def opp_sliding_window(data_x, data_y, ws, ss, flatten=True):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1), flatten)
    # This Command will assign the last sample class to be the window class
    # data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    # This Commands will assign the most dominant class to be the window class
    windows = sliding_window(data_y, ws, ss, flatten)
    mostDom = []
    for i in windows:
        mostDom.append(most_common(i))
    data_y = np.asarray(mostDom)
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


# ____________________________________________________________
def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


# __________________________________________________________________
def save_all_data(data, label, target_filename, subject=7):
    nb_samples = [[0, 0, 0, 0, 0],  # subject1
                  [0, 0, 0, 0, 0],  # subject2
                  [0, 0, 0, 0, 0],  # subject3
                  [0, 0, 0, 0, 0],  # subject4
                  [0, 0, 0, 0, 0],  # subject5
                  [0, 0, 0, 0, 0],  # subject6
                  [20820, 19940, 19933, 19664, 19707]  # subject7
                  ]
    end_training = sum(nb_samples[subject - 1][0:3])
    end_validation = end_training + nb_samples[subject - 1][3]
    #  subject 7, REP 1,2,3 are training  set,REP 4 for validation, REP 5 for test
    X_train, y_train = data[:end_training, :], label[:end_training]
    X_validation, y_validation = data[end_training:end_validation, :], label[end_training:end_validation]
    X_test, y_test = data[end_validation:, :], label[end_validation:]

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))

    obj = [(X_train, y_train), (X_validation, y_validation), (X_test, y_test)]
    f = open(os.path.join('./data/', target_filename), 'wb+')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


# ______________________________________________________________
def load_dataset(filename):
    # Function to Load Processed Dataset
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()
    X_train, y_train = data[0]
    X_validation, y_validation = data[1]
    X_test, y_test = data[2]

    print(" ..from file {}".format(filename))

    X_train = X_train.astype(np.float32)
    X_validation = X_validation.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_validation = y_validation.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))

    return X_train, y_train, X_validation, y_validation, X_test, y_test


# ___________________________________________________
def remove_null_class(X, Y, remove_short_activities=1):
    if remove_short_activities:
        breaks = [0, 5, 6, 13, 15, 18, 24, 31, 34, 37, 46, 14, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                  48]  # 10 is relaxation
    else:
        breaks = [0, 5, 6, 13, 15, 18, 24, 31, 34, 37, 46, 14]

    for brk in breaks:
        null_indexes = np.where(Y == brk)[0]
        X = np.delete(X, null_indexes, axis=0)
        Y = np.delete(Y, null_indexes, axis=0)

    Y, qq = make_label(Y)  # labels in number 0-49
    return X, Y


# ____________________________________________________
def read_subject_data(subject_number, repetitions, label_level=3, remove_time=2):
    data_files = [
        'MT_00200493-000-000_00342485.csv',
        'MT_00200493-000-000_00342489.csv',
        'MT_00200493-000-000_00342490.csv',
        'MT_00200493-000-000_00342488.csv',
        'MT_00200493-000-000_00342486.csv'
    ]
    data = []
    print('Loading dataset files ...')
    for repetition in repetitions:  # because we have 5 repetition
        rep_tmp = []
        label_f_name = './DataSet/Subject ' + str(subject_number) + '/repetition ' + str(
            repetition) + '/Subject ' + str(
            subject_number) + '_Rep ' + str(repetition) + '.' + '.csv'
        label_tmp = np.genfromtxt(label_f_name, delimiter=',', dtype=str, comments='//')[1:, :]
        true_header = 'PacketCounter,Time,Acc_X,Acc_Y,Acc_Z,VelInc_X,VelInc_Y,VelInc_Z,OriInc_q0,OriInc_q1,OriInc_q2,OriInc_q3,Roll,Pitch,Yaw,SCRAcc_X,SCRAcc_Y,SCRAcc_Z,SCRGyr_X,SCRGyr_Y,SCRGyr_Z'
        for filename in data_files:
            try:
                fn = './DataSet/Subject ' + str(subject_number) + '/repetition ' + str(
                    repetition) + '/' + filename
                print('Loading ' + fn)
                tmp = np.loadtxt(fn, delimiter=',', comments='//', encoding='utf-8', skiprows=10)

                # tmp = tmp[:,remove_time:]
                # Select columns
                tmp = select_columns(tmp)
                # header = tmp[10, :]
                # if header == true_header:
                #     print("yes")
                # else:
                #     print("no")

                if rep_tmp == []:
                    rep_tmp = tmp
                else:
                    rep_tmp = np.concatenate( (rep_tmp, tmp), axis=1 )
            except KeyError:
                print('ERROR: Did not find {0} in zip file'.format(filename))

        if data == []:
            label = label_tmp
            data = rep_tmp[0:len(label_tmp), :]

        else:
            data = np.concatenate((data, rep_tmp[0:len(label_tmp), :]), axis=0)
            label = np.concatenate((label, label_tmp), axis=0)

    data = remove_nan(data)

    return data, label[:, label_level]


# __________________________________________________________
def read_all_subject_data(time_removal=2, lbl_level=2):
    s1 = [2, 3, 4, 5]
    s2 = [1, 2]
    s3 = [2, 3, 4, 5]
    s5 = [1, 2, 3, 4, 5]
    s6 = [2, 3, 4, 5]
    s7 = [1, 2, 3, 4, 5]
    data1, label1 = read_subject_data(subject_number=1, repetitions=s1, label_level=lbl_level, remove_time=time_removal)
    data2, label2 = read_subject_data(subject_number=2, repetitions=s2, label_level=lbl_level, remove_time=time_removal)
    data3, label3 = read_subject_data(subject_number=3, repetitions=s3, label_level=lbl_level, remove_time=time_removal)
    data5, label5 = read_subject_data(subject_number=5, repetitions=s5, label_level=lbl_level, remove_time=time_removal)
    data6, label6 = read_subject_data(subject_number=6, repetitions=s6, label_level=lbl_level, remove_time=time_removal)
    data7, label7 = read_subject_data(subject_number=7, repetitions=s7, label_level=lbl_level, remove_time=time_removal)

    data = np.concatenate((data1, data2, data3, data5, data6, data7), axis=0)
    label = np.concatenate((label1, label2, label3, label5, label6, label7), axis=0)

    return data, label


# ____________________________________
def null_portion(data):
    n_sensors = data.shape[1]
    n_samples = data.shape[0]
    null_portions = np.zeros((n_sensors, 1))
    for i in range(n_sensors):
        null_portions[i] = sum(np.isnan(data[:, i]))

    print("done")

    return 0


# _____________________________________________
def save_all_data_allsubjects(data, label, target_filename, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP,
                              remove_null_classes=0):
    # randomize data distribution
    if remove_null_classes:
        data, label = remove_null_class(data, label)
    data = normalize_data(data)
    data, label = opp_sliding_window(data, label, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    idx = np.random.permutation(len(data))
    data, label = data[idx], label[idx]

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.30, random_state=42)

    #  take 80 percent of all data for train and the rest for test
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    print(
        "Final datasets with size: | train {0} | validation {1} | test {2} | ".format(X_train.shape, X_validation.shape,
                                                                                      X_test.shape))

    obj = [(X_train, y_train), (X_validation, y_validation), (X_test, y_test)]
    f = open(os.path.join('./Dataset/', target_filename), 'wb+')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


# _______________________________________________________________
def read_and_segment(subject, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP, generating_data=0, lbl_level=2, target=0):
    # lbl_level=2 -> high_level
    output_file = "Lissi_hl_activity_subject" + str(subject) + ".data"

    s1 = [2, 3, 4, 5]
    s2 = [1, 2]
    s3 = [2, 3, 4, 5]
    s4 = [0]
    s5 = [1, 2, 3, 4, 5]
    s6 = [2, 3, 4, 5]
    s7 = [1, 2, 3, 4, 5]

    repetitions = [s1, s2, s3, s4, s5, s6, s7]

    if generating_data:
        data, label = read_subject_data(subject_number=subject, repetitions=repetitions[subject - 1],
                                        label_level=lbl_level, remove_time=1)
        label, dic = make_label(label)  # labels in number 0-49
        save_all_data_allsubjects(data, label, output_file, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset('./DataSet/' + output_file)

    # mean_time, mean_sample = mf.statistics_investigation(data, encoded_label, np.unique(encoded_label), qq)

    pkl_filename = "pca_model_lissi.pkl"
    n_features = 100
    my_pca = decomposition.PCA(n_components=n_features)
    if target:
        with open(pkl_filename, 'rb') as file:
            my_pca = cp.load(file)
    else:
        with open(pkl_filename, 'wb') as file:
            my_pca.fit(x_train)
            cp.dump(my_pca, file)

    x_train = my_pca.transform(x_train)
    x_val = my_pca.transform(x_val)
    x_test = my_pca.transform(x_test)

    number_of_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=number_of_classes)  # binary labels
    y_val = to_categorical(y_val, num_classes=number_of_classes)  # binary labels
    y_test = to_categorical(y_test, num_classes=number_of_classes)  # binary labels

    return x_train, y_train, x_val, y_val, x_test, y_test


# _________________________________________________________________________
def save_to_mat(x_train, x_val, x_test, y_train, y_val, y_test, filename):
    data = {}
    data['x_train'] = x_train
    data['x_val'] = x_val
    data['x_test'] = x_test
    data['y_train'] = y_train
    data['y_val'] = y_val
    data['y_test'] = y_test

    io.savemat(filename + '.mat', data)


# _______________________________
def gan_model_evaluation(gan, source_subject, target_subject, x_test, y_test):
    clsf = gan.build_classifier()

    clsf.load_weights('./Weights/lissi_adopted_classifier_best_S' + str(source_subject) + '_to_S' + str(
        target_subject) + '.hdf5')

    pred_val = np.argmax(clsf.predict(x_test), 1)
    print(classification_report(np.argmax(y_test, 1), pred_val))


# ____________________________________________________
def plot_loss(x_test, y_test, source_subject, target_subject):
    # confusion_matrix()
    clsf = load_model(
        './Weights/S' + str(source_subject) + ' to S' + str(target_subject) + '/lissi_adopted_classifier_best_S' + str(
            source_subject) + '_to_S' + str(target_subject) + '.hdf5')

    pred_val = np.argmax(clsf.predict(x_test), 1)
    print(accuracy_score(np.argmax(y_test, 1), pred_val))
    print(classification_report(np.argmax(y_test, 1), pred_val))
    print(confusion_matrix(np.argmax(y_test, 1), pred_val))
    print('S%d -> S%d' % (source_subject, target_subject))
    # file = open('train_lissi.log')
    # file_content = file.read()
    # lines = re.split('INFO:root:#Epoch', file_content)
    # lines = lines[1:]
    # epochs = len(lines)
    # f1 = np.zeros((epochs, 1))
    # cls_tr_loss = np.zeros((epochs, 1))
    # d_tr_loss = np.zeros((epochs, 1))
    # combined_tr_loss = np.zeros((epochs, 1))

    #
    # for i in range(0, epochs):
    #     lines[i] = lines[i].replace('[', ' ')
    #     lines[i] = lines[i].replace(']', ' ')
    #     brackets = re.split(':|,| |\n', lines[i])
    #     f1[i] = brackets[45]
    #     cls_tr_loss[i] = brackets[11]
    #     d_tr_loss[i] = brackets[18]
    #     combined_tr_loss[i] = (gan.lambda_clf * float(brackets[24]) + gan.lambda_adv * float(brackets[28])) / 101
    #
    # plt.plot(range(epochs), f1, marker='o', linestyle='dashed', color='b')
    # plt.plot(range(epochs), d_tr_loss, marker='o', linestyle='dashed', color='r')
    # plt.plot(range(epochs), cls_tr_loss, marker='o', linestyle='dashed', color='g')
    # plt.plot(range(epochs), combined_tr_loss, marker='o', linestyle='dashed', color='y')
    # plt.legend(['F1', 'D_tr_loss', 'CLS_tr_loss', 'Combined(G)_tr_loss'])
    # plt.grid(True)


# __________________________________________
def check_headers():
    s1 = [2, 3, 4, 5]
    s2 = [1, 2]
    s3 = [2, 3, 4, 5]
    s4 = []
    s5 = [1, 2, 3, 4, 5]
    s6 = [2, 3, 4, 5]
    s7 = [1, 2, 3, 4, 5]
    s8 = [1, 2, 3]
    s9 = [1, 2, 3, 4, 5]
    s10 = [1, 2, 5]
    s11 = [1, 2, 3, 5]
    s13 = [1, 2, 3, 4, 5]
    s14 = [1, 2, 3, 4, 5]
    s15 = [2, 3, 4, 5]
    s16 = [1, 2, 4, 5]
    s17 = [2, 3, 4, 5]
    s20 = [1, 2, 4]

    available_rep = {'S1': s1, 'S2': s2, 'S3': s3, 'S5': s5, 'S6': s6, 'S7': s7, 'S8': s8, 'S9': s9, 'S10': s10,
                     'S11': s11,
                     'S13': s13, 'S14': s14, 'S15': s15, 'S16': s16, 'S17': s17, 'S20': s20}

    data_files = [
        'MT_00200493-000-000_00342485.csv',
        'MT_00200493-000-000_00342489.csv',
        'MT_00200493-000-000_00342490.csv',
        'MT_00200493-000-000_00342488.csv',
        'MT_00200493-000-000_00342486.csv'
    ]
    true_header = 'PacketCounter,Time,Acc_X,Acc_Y,Acc_Z,VelInc_X,VelInc_Y,VelInc_Z,OriInc_q0,OriInc_q1,OriInc_q2,OriInc_q3,Roll,Pitch,Yaw,SCRAcc_X,SCRAcc_Y,SCRAcc_Z,SCRGyr_X,SCRGyr_Y,SCRGyr_Z\n'
    second_header = 'PacketCounter,Time,Acc_X,Acc_Y,Acc_Z,VelInc_X,VelInc_Y,VelInc_Z,OriInc_q0,OriInc_q1,OriInc_q2,OriInc_q3,Roll,Pitch,Yaw,FreeAcc_X,FreeAcc_Y,FreeAcc_Z,Gyr_X,Gyr_Y,Gyr_Z\n'

    third_header = 'PacketCounter,time,SampleTimeFine,Acc_X,Acc_Y,Acc_Z,FreeAcc_X,FreeAcc_Y,FreeAcc_Z,Gyr_X,Gyr_Y,Gyr_Z,Mag_X,Mag_Y,Mag_Z,VelInc_X,VelInc_Y,VelInc_Z,OriInc_q0,OriInc_q1,OriInc_q2,OriInc_q3,Roll,Pitch,Yaw\r\n'
    first_h = 0
    second_h = 0
    third_h = 0

    print('First Header: ' + true_header)
    print('Second Header: ' + second_header)
    print('Third Header: ' + third_header)
    print('Subject, Repetition, Filename, Header_Type')

    for key in available_rep:
        repetitions = available_rep[key]
        subject_number = int(key[1:])
        print('Loading dataset files ...')
        for repetition in repetitions:  # because we have 5 repetition
            # rep_tmp = []
            # label_f_name = './DataSet/Dataset Annotation Labels/Subject ' + str(subject_number) + '/repetition ' + str(repetition) + '/Subject ' + str(
            #     subject_number) + '_Rep ' + str(repetition) + '.' + '.csv'
            # label_tmp = np.genfromtxt( label_f_name, delimiter=',', dtype=str, comments='//' )[1:, :]
            for filename in data_files:
                try:

                    fn = './DatasetGathering/Dataset Annotation Labels/Subject ' + str(
                        subject_number) + '/repetition ' + str(repetition) + '/' + filename

                    print('Loading ' + fn)
                    # tmp = np.loadtxt(fn, delimiter=',', comments='//', encoding='utf-8')
                    my_filtered_csv = pd.read_csv(fn, usecols=['PacketCounter', 'Time', 'Acc_X'], skiprows=9)
                    tmp = open(fn, 'rb').readlines()
                    my_filtered_csv = pd.read_csv(fn, usecols=['PacketCounter', 'Time', 'Acc_X'], skiprows=9)
                    # tmp = tmp[:,remove_time:]
                    # Select columns
                    header = tmp[9].decode("utf-8")
                    if header == true_header:
                        first_h = first_h + 1
                        print(str(subject_number) + ', ' + str(repetition) + ',' + filename + '1')
                    elif header == second_header:
                        second_h = second_h + 1
                        print(str(subject_number) + ', ' + str(repetition) + ',' + filename + '2')

                    elif header.casefold() == third_header.casefold():
                        third_h = third_h + 1
                        print(str(subject_number) + ', ' + str(repetition) + ',' + filename + '3')

                    else:
                        print("no")

                    # if rep_tmp == []:
                    #     rep_tmp = tmp
                    # else:
                    #     rep_tmp = np.concatenate( (rep_tmp, tmp), axis=1 )
                except KeyError:
                    print('ERROR: Did not find {0} in zip file'.format(filename))

    return 0
