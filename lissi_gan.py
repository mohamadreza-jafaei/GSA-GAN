from __future__ import print_function, division

from imblearn.over_sampling import SMOTE
import sklearn
from sklearn.utils import class_weight
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Dropout, Concatenate, Conv1D
from keras.layers import BatchNormalization, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lissi_dataset_preprocessing as pp
from datetime import datetime
import logging

logger_fn = "train_lissi_" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"

logger = logging.getLogger()
fhandler = logging.FileHandler(filename=logger_fn, mode='a')
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


class GAN():
    def __init__(self):

        # Input
        self.n_features = 100
        self.channels = 1
        self.input_shape = (self.n_features, self.channels)
        self.num_classes = 8

        self.best_accuracy = -10.

        # Loss weights
        self.lambda_adv = 1.
        self.lambda_clf = 1.

        # Output of D
        self.discriminator_output = (7, 1)

        # Number of residual blocks for G
        self.residual_blocks = 4

        # Number of filters in first layer of D, G, and C
        self.df = 32
        self.cf = 64
        self.gf = 64

        # Build and compile D
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=SGD(0.01), metrics=['accuracy'])

        # Build G
        self.generator = self.build_generator()

        # Build C
        self.classifer = self.build_classifier()
        self.classifer.compile(loss='categorical_crossentropy',
                               optimizer=RMSprop(0.1),
                               metrics=['accuracy'])

        # Input samples from both domains
        sample_A = Input(shape=self.input_shape)
        sample_B = Input(shape=self.input_shape)

        # Translate sample from domain A to domain B and classify it
        fake_B = self.generator(sample_A)
        predicted_class = self.classifer(fake_B)

        # Only G and C will be trained in this step
        self.discriminator.trainable = False
        self.classifer.trainable = False
        self.generator.trainable = True

        # D evaluate validity(fake or real) of generated samples
        validity = self.discriminator(fake_B)

        self.combined = Model(sample_A, [validity, predicted_class])
        self.combined.compile(loss=['mse', 'categorical_crossentropy'],
                              loss_weights=[self.lambda_adv, self.lambda_clf],
                              optimizer=Adam(0.1, 0.8),
                              metrics=['accuracy'])

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            # Convolutional blocks for D
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            d = Dropout(0.2)(d)
            return d

        input_data = Input(shape=self.input_shape)
        d1 = d_layer(input_data, self.df, normalization=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        output = Conv1D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(d4)

        return Model(input_data, output)

    def build_generator(self):

        def residual_block(layer_input):
            d = Conv1D(self.gf, kernel_size=3, strides=1, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = Activation('relu')(d)
            d = Conv1D(self.gf, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Dropout(0.5)(d)
            d = Add()([d, layer_input])
            return d

        input_data = Input(shape=self.input_shape)
        l1 = Conv1D(self.gf, kernel_size=3, padding='same', activation='relu')(input_data)
        # make series of residual blocks to build generator
        r = residual_block(l1)
        for _ in range(self.residual_blocks - 1):
            r = residual_block(r)
        output = Conv1D(self.channels, kernel_size=3, padding='same', activation='sigmoid')(r)

        return Model(input_data, output)

    def build_classifier(self):

        def clf_layer(layer_input, filters, f_size=4, normalization=True):
            # Convolutional blocks for C
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = BatchNormalization()(d)
            d = Dropout(0.5)(d)
            return d

        input_data = Input(shape=self.input_shape)

        c1 = clf_layer(input_data, self.cf, normalization=False)
        c2 = clf_layer(c1, self.cf * 2)
        c3 = clf_layer(c2, self.cf * 4)
        c4 = clf_layer(c3, self.cf * 8)
        c5 = clf_layer(c4, self.cf * 8)
        c6 = clf_layer(c5, self.cf * 8)
        c7 = clf_layer(c6, self.cf * 8)
        class_prediction = Dense(self.num_classes, activation='softmax')(Flatten()(c7))

        return Model(input_data, class_prediction)

    def train(self, epochs, batch_size=128):

        test_accs = []
        validation_acc = []
        f1_epoch = []
        # Adversarial GT
        valid = 0.9 * np.ones((batch_size, *self.discriminator_output))

        fake = -0.1 * np.random.randint(9, 11, size=(batch_size, *self.discriminator_output))

        iter_source = dataset_source.make_one_shot_iterator()
        iter_target = dataset_target.make_one_shot_iterator()
        sess = tf.Session()
        element_source = iter_source.get_next()
        element_target = iter_target.get_next()

        for epoch in range(epochs):

            test_acc_batch = []
            g_loss_temp = []
            d_loss_temp = []
            cls_loss_temp = []
            for step in range(number_of_batches):

                temp_src = sess.run(element_source)
                temp_trg = sess.run(element_target)

                batch_data_A, labels_A = temp_src[0], np.argmax(temp_src[1], 1)
                batch_data_B, labels_B = temp_trg[0], np.argmax(temp_trg[1], 1)


                # batch_data_A, labels_A = tr_src_generator.__getitem__(step)[0], np.argmax(
                #     tr_src_generator.__getitem__(step)[1], 1)
                # batch_data_B, labels_B = tr_target_generator.__getitem__(step)[0], np.argmax(
                #     tr_target_generator.__getitem__(step)[1], 1)
                # batch_data_A = np.expand_dims(batch_data_A, -1)
                # batch_data_B = np.expand_dims(batch_data_B, -1)

                # Translate samples from domain A to domain B
                fake_B = self.generator.predict(batch_data_A)

                # ### Train D ### #
                # Original data = real | generated data = Fake
                noise = np.random.uniform(0, 1, batch_data_B.shape)
                d_loss_real = self.discriminator.train_on_batch(np.add(noise, batch_data_B), valid)
                noise = np.random.uniform(0, 1, batch_data_B.shape)
                d_loss_fake = self.discriminator.train_on_batch(np.add(noise, fake_B), fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_loss_temp.append(d_loss)

                # ###  Train C ### #
                # encoding of labels
                labels_A = to_categorical(labels_A, num_classes=self.num_classes)

                t_loss_fake = self.classifer.train_on_batch(batch_data_A, labels_A, class_weight=class_weight_src)
                t_loss_real = self.classifer.train_on_batch(fake_B, labels_A, class_weight=class_weight_src)
                cls_tr_loss = 0.5 * np.add(t_loss_fake, t_loss_real)
                cls_loss_temp.append(cls_tr_loss)

                # ###  Train G ### #
                # Optional: Adding noise to prevent collapse
                noise = np.zeros(batch_data_A.shape)

                g_loss = self.combined.train_on_batch(np.add(batch_data_A, noise), [fake, labels_A],
                                                      class_weight=[[], class_weight_trg])
                g_loss_temp.append(g_loss)

                # Evaluation on target domain B
                pred_B = self.classifer.predict(batch_data_B)
                test_acc_batch.append(accuracy_score(labels_B, np.argmax(pred_B, 1)))

                # Plot the progress
                print("%d: [D -tr_loss: %.5f, tr_acc: %3d%%],[G -tr_loss: %.5f], [classifier - tr_loss: %.5f, tr_acc: %3d%%]"
                      % (step, d_loss[0], 100 * float(d_loss[1]),
                         g_loss[1], cls_tr_loss[0], 100*cls_tr_loss[1]))

            # test on target domain
            test_accs.append(np.mean(test_acc_batch))
            # validate the model on validation set of target
            pred_val = np.argmax(self.classifer.predict(x_val_trg), 1)
            validation_acc.append(accuracy_score(np.argmax(y_val_trg, 1), pred_val))
            f1_epoch.append(f1_score(np.argmax(y_val_trg, 1), pred_val, average='weighted'))
            if f1_epoch[epoch] > self.best_accuracy:
                self.classifer.save('./Weights/lissi_adopted_classifier_best_S' + str(source_subject) + '_to_S' + str(
                    target_subject) + '.hdf5')
                self.best_accuracy = f1_epoch[epoch]

            logging.info(
                "#Epoch%d: [cls: tr_Acc: %.5f, tr_loss: %.4f] [D: tr_loss: %.4f], G[D_tr_loss: %.4f,"
                " clf_tr_loss: %.4f]   ,Val Acc(C): %.5f   , F1:   %.5f" %
                (epoch, test_accs[epoch], np.mean(cls_loss_temp, 0)[0], np.mean(d_loss_temp, 0)[0],
                 np.mean(g_loss_temp, 0)[1], np.mean(g_loss_temp, 0)[2], validation_acc[epoch], f1_epoch[epoch]))
            logging.info(sklearn.metrics.confusion_matrix(np.argmax(y_val_trg, 1), pred_val))

            print(
                "#Epoch%d: [cls: tr_Acc: %.5f, tr_loss: %.4f] [D: tr_loss: %.4f], G[D_tr_loss: %.4f,"
                " clf_tr_loss: %.4f]   ,Val Acc(C): %.5f   , F1:   %.5f" %
                (epoch, test_accs[epoch], np.mean(cls_loss_temp, 0)[0], np.mean(d_loss_temp, 0)[0],
                 np.mean(g_loss_temp, 0)[1], np.mean(g_loss_temp, 0)[2], validation_acc[epoch], f1_epoch[epoch]))
            print(sklearn.metrics.confusion_matrix(np.argmax(y_val_trg, 1), pred_val))

if __name__ == '__main__':
    SLIDING_WINDOW_LENGTH = 76  # 96 -64
    SLIDING_WINDOW_STEP = 40

    source_subject = 1
    target_subject = 3
    # data generation flag: 1= to generate sets (for first time), 0 = to load already generated sets
    DG = 0

    x_train_src, y_train_src, x_val_src, y_val_src, x_test_src, y_test_src = pp.read_and_segment(source_subject,
                                                                                                 SLIDING_WINDOW_LENGTH,
                                                                                                 SLIDING_WINDOW_STEP,
                                                                                                 generating_data=DG)

    x_train_trg, y_train_trg, x_val_trg, y_val_trg, x_test_trg, y_test_trg = pp.read_and_segment(target_subject,
                                                                                                 SLIDING_WINDOW_LENGTH,
                                                                                                 SLIDING_WINDOW_STEP,
                                                                                                 generating_data=DG,
                                                                                                 target=1)

    over_sampler = SMOTE()
    x_train_trg, y_train_trg = over_sampler.fit_sample(x_train_trg, y_train_trg)
    x_train_src, y_train_src = over_sampler.fit_sample(x_train_src, y_train_src)

    class_weight_src = class_weight.compute_sample_weight('balanced', np.unique(np.argmax(y_train_src, 1)),
                                                          np.argmax(y_train_src, 1))
    class_weight_trg = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train_trg, 1)),
                                                         np.argmax(y_train_trg, 1))

    # class_weight_src = np.log(class_weight_src)
    # class_weight_trg = np.log(class_weight_trg)

    tmp = x_train_src.shape
    x_train_src = np.reshape(x_train_src, (tmp[0], tmp[1], 1))

    tmp = x_train_trg.shape
    x_train_trg = np.reshape(x_train_trg, (tmp[0], tmp[1], 1))

    tmp = x_test_trg.shape
    x_test_trg = np.reshape(x_test_trg, (tmp[0], tmp[1], 1))

    tmp = x_val_trg.shape
    x_val_trg = np.reshape(x_val_trg, (tmp[0], tmp[1], 1))


    batch_size = 64
    dataset_source = tf.data.Dataset.from_tensor_slices((x_train_src, y_train_src))
    dataset_source = dataset_source.shuffle(buffer_size=y_train_src.shape[0])
    dataset_source = dataset_source.batch(batch_size, drop_remainder=True).repeat()

    dataset_target = tf.data.Dataset.from_tensor_slices((x_train_trg, y_train_trg))
    dataset_target = dataset_target.shuffle(buffer_size=y_train_trg.shape[0])
    dataset_target = dataset_target.batch(batch_size, drop_remainder=True).repeat()

    number_of_batches = np.int(np.max([x_train_src.shape[0] / batch_size, x_train_trg.shape[0] / batch_size]))

    # tr_src_generator = BalancedBatchGenerator(x_train_src, y_train_src, sampler=NearMiss(), batch_size=bs)
    # tr_target_generator = BalancedBatchGenerator(x_train_trg, y_train_trg, sampler=NearMiss(), batch_size=bs)
    # number_of_batches = np.min([tr_src_generator.__len__(), tr_target_generator.__len__()])

    gan = GAN()
    gan.train(epochs=400, batch_size=batch_size)

    # Test the adapted classifier
    clsf = gan.build_classifier()
    clsf.load_weights(
        './Weights/lissi_adopted_classifier_best_S' + str(source_subject) + '_to_S' + str(
            target_subject) + '.hdf5')
    pred_test = np.argmax(clsf.predict(x_test_trg), 1)
    print(accuracy_score(np.argmax(y_test_trg, 1), pred_test))
    print(classification_report(np.argmax(y_test_trg, 1), pred_test))
    pp.plot_loss(x_test_trg, y_test_trg, source_subject, target_subject)

    logging.info('Training is finished')
