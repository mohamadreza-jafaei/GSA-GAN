import logging
from collections import Counter

import scipy
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.utils import plot_model
import lissi_dataset_preprocessing as pp
import numpy as np
import model_functions as mf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.stats import wasserstein_distance


data_mode = "hl_activity"
SLIDING_WINDOW_LENGTH = 76
SLIDING_WINDOW_STEP = 40
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']
# 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11',
# 'class 12', 'class 13', 'class 14', 'class 15', 'class 16', 'class 17']

generating_data = 0
source_subject = 6
target_subject = 2

logging.basicConfig( filename='compare.log', level=logging.INFO )
logging.info( 'Started' )

def compare_subject():
    for i in [1, 2, 3, 5,6,7]:
        source_subject = i
        x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = pp.read_and_segment(
            source_subject,
            SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP,
            generating_data=generating_data )

        for j in [1, 2, 3, 5,6,7]:

            target_subject = j
            x_train_trg, y_train_trg, x_val_trg, y_val_trg, x_test_trg, y_test_trg = pp.read_and_segment(
                target_subject,
                SLIDING_WINDOW_LENGTH,
                SLIDING_WINDOW_STEP,
                generating_data=generating_data,
                target=1 )

            distance = np.zeros( [x_train_src.shape[1], 1] )
            ps = np.zeros( [x_train_src.shape[1], 1] )
            pt = np.zeros( [x_train_trg.shape[1], 1] )
            class_size_src = Counter( np.argmax( y_train_src, 1 ) )
            class_size_trg = Counter( np.argmax( y_train_trg, 1 ) )
            for i in range(6):
                ps[i] = class_size_src[i]
                pt[i] = class_size_trg[i]

            kl_distance = scipy.stats.entropy( ps, pt )

            for i in range( x_train_src.shape[1] ):
                distance[i] = wasserstein_distance( x_train_src[:, i], x_train_trg[:, i] )

            logging.info( 'Source:' + str( source_subject ) + ',target:' + str( target_subject ) + ', W:' + str(
                sum( sum( distance ) ) ) + ' KL: = ' + str( kl_distance ) )


pp.check_headers()
x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = pp.read_and_segment(source_subject,
                                                                                             SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP,
                                                                                             generating_data=generating_data)

x_train_trg, y_train_trg, x_val_trg, y_val_trg, x_test_trg, y_test_trg = pp.read_and_segment(target_subject,
                                                                                              SLIDING_WINDOW_LENGTH,
                                                                                              SLIDING_WINDOW_STEP,
                                                                                              generating_data=generating_data,
                                                                                              target=1)

distance = np.zeros([x_train_src.shape[1],1])
for i in range(x_train_src.shape[1]):
    distance[i] = wasserstein_distance(x_train_src[:,i],x_train_trg[:,i])


print('D( subject '+ str(source_subject)+', subject '+str(target_subject)+' ) = ' + str(sum(sum(distance))))
# print('KL( subject '+ str(source_subject)+', subject '+str(target_subject)+' ) = ' + str(kl_distance))

file_path = './Weights/lissi_source_classifier_' + data_mode + '_best_subject_' + str( source_subject ) + '.hdf5'
batch_size = 100
epoch_size = 40
# Classification Configuration
checkpoint = ModelCheckpoint( file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max' )
callbacks_list = [checkpoint]

print("Classification Model Fitting")
# Dense model:
# model_base = mf.base_model_exp2(n_features=88, n_classes=6)
# plot_model(model_base, to_file='exp3_base_model_'+DATA_MODE+'.png')


# conv model
tmp = x_train_src.shape
x_train_src = np.reshape(x_train_src, (tmp[0], tmp[1], 1))

tmp = x_val_src.shape
x_val_src = np.reshape(x_val_src, (tmp[0], tmp[1], 1))

tmp = x_test_src.shape
x_test_src = np.reshape(x_test_src, (tmp[0], tmp[1], 1))

tmp = x_test_trg.shape
x_test_trg = np.reshape(x_test_trg, (tmp[0], tmp[1], 1))


model_base = mf.conv_model(input_shape=(100,1,), num_classes =8)
model_base.compile( loss='categorical_crossentropy', optimizer=Adam(0.00002), metrics=['accuracy'] )
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlyStopping = EarlyStopping( monitor='val_loss', patience=30, verbose=0, mode='auto')
callbacks_list = [checkpoint, earlyStopping]

model_base.fit(x=x_train_src, y=y_train_src, epochs=epoch_size, batch_size=batch_size, shuffle=False, verbose=0,
               callbacks=callbacks_list, validation_data = (x_val_src, y_val_src))
# model_base = load_model( file_path)
# plot_model(model_base, to_file='my_model_hl.png')

print('model evaluation on source\n')
mf.model_evaluation_base_conv(model_base, x_test_src, y_test_src, target_names)

print('model evaluation on target (WO transfer)')
# mf.model_evaluation_base(model_base, x_test_trg, y_test_trg, target_names)
y_pred_base = np.argmax( model_base.predict( x_test_trg ), 1 )
y_truth_base = np.argmax( y_test_trg, 1 )
test_loss, test_acc = model_base.evaluate( x_test_trg, y_test_trg, verbose=1 )

print( 'test accuracy ' + str( test_acc ) )
print( classification_report( y_truth_base, y_pred_base, target_names=target_names ) )

print('done')

