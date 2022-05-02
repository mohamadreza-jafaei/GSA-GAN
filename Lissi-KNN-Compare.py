import sklearn
from sklearn.metrics import classification_report
import lissi_dataset_preprocessing as pp
import numpy as np
from sklearn.neighbors import  KNeighborsClassifier

data_mode = "hl_activity"
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7']

SLIDING_WINDOW_LENGTH = 76# 96 -64
SLIDING_WINDOW_STEP = 40

source_subject = 1
target_subject = 6
generating_data = 0

x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = pp.read_and_segment(source_subject,
                                                                                             SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP,
                                                                                             generating_data=generating_data)


# Supervised method
print("Classification Model Fitting")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train_src, y_train_src)

for i in [1,2,3,5,6,7]:
    target_subject = i
    x_train_trg, y_train_trg, x_val_trg, y_val_trg, x_test_trg, y_test_trg = pp.read_and_segment(target_subject,
                                                                                                 SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP,
                                                                                                 generating_data=generating_data, target= 1)

    y_pred = neigh.predict(x_test_trg)
    print('#____________________________________________________________________\n\n from S%d -> S%d\n'%(source_subject,target_subject))
    print(classification_report( y_test_trg, y_pred, target_names=target_names))
    print('Accuracy: %.5f\n'% sklearn.metrics.accuracy_score( np.argmax( y_test_trg, 1 ), np.argmax(y_pred,1)))
    print( sklearn.metrics.confusion_matrix( np.argmax( y_test_trg, 1 ), np.argmax(y_pred,1)))
print('done')



