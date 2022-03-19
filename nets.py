#!/usr/bin/python

"""
Copyright 2018 Luciano Melodia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import scipy.io as sio
import config as cfg
import visualize as vs
from sklearn.preprocessing import MinMaxScaler

from keras.models import *
from keras.layers import Conv2D, Deconv2D, UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Reshape, Input
from keras.layers.merge import multiply
from keras.optimizers import Nadam
from keras import losses
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.backend import minimum, maximum

def IoU_metric(y_true, y_pred, smooth=K.epsilon()):
    min = K.sum(minimum(K.abs(y_true), K.abs(y_pred)))
    max = K.sum(maximum(K.abs(y_true), K.abs(y_pred)))
    sum = (min + smooth) / (max + smooth)
    return K.mean(sum)

def IoU(y_true, y_pred):
    return 1 - IoU_metric(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    return losses.mse(y_true, y_pred)

def clinic_loss(y_true, y_pred):
    loss = K.sum(K.tf.multiply(((y_true - y_pred)**2),(y_true/K.sum(y_true)))) * 100
    return loss

class uNet:
    trainQuantil = 0
    batch_size = 0
    epochs = 0

    def __init__(self):
        self.trainQuantil = cfg.settings['trainQuantil']
        self.batch_size = cfg.settings['batch_size']
        self.epochs = cfg.settings['epochs']

    def load(self, path, name, appendix=None, save=False, boost=False, fac=None, norm=False):
        """
        :param path: .mat files required for reading with saved double array (no struct arrays here)
        :param name: name of the resulting file
        :param save: save file (true/false)
        :param mode: decide wether to use npz or direct access mode
        :param appendix: appendix will be added at the end of the filename as an extension
        :param boost: upsamling the data (true/false)
        :param fac: upsampling factor
        :param norm: normalization (true/false
        :return:
        Function returns an array of the required data.
        Data has to be 4-dim. Path specifies the path to the data folder.
        """
        files = sorted(os.listdir(path))
        data = np.empty([9,9,9,0], dtype='longfloat')
        amount = len(files)

        for f in files:
            amount = amount - 1
            f=f.replace("._","")
            if f.endswith(".mat"):
                dvk = sio.loadmat(path + f)
                data = np.concatenate((data, dvk[name]), axis=3)
                if amount == 0:
                    break

        data = data.T

        if norm == True:
            x_min = data.min()
            x_max = data.max()
            data = (data - x_min) / (x_max - x_min)

        return(data)

    def store(self, path, name, appendix="", save=False, boost=False, fac=10, amount=10, norm=False):
        self.load(path, name, appendix, save, boost, fac, amount, norm)
        return "Data has been loaded."

    def restore(self, start_path, target_path):
        """
        :param start_path: npz file of compressed densities (sparse coded)
        :param target_path: npz file of compressed kernels(sparse coded)
        :return:
        """
        start = np.load(start_path)
        target = np.load(target_path)
        return start['a'], target['a']

    def uNet(self, trainingData):
        """
        :param trainingData: numpy array of the training data.
        :return:
        """
        input_shape = (9, 9, 9)
        inputs = Input(shape=input_shape)

        re1 = Reshape((27, 27, 1))(inputs)
        up1 = UpSampling2D(size=(2, 2), data_format='channels_last')(re1)

        conv1 = Conv2D(8, 3, kernel_initializer="uniform")(up1)
        conv1 = LeakyReLU(alpha=5.5)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv2 = Conv2D(8, 3, kernel_initializer="uniform")(conv1)
        conv2 = LeakyReLU(alpha=5.5)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv3 = Conv2D(16, 3, kernel_initializer="uniform")(conv2)
        conv3 = LeakyReLU(alpha=5.5)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(16, 3, kernel_initializer="uniform")(conv3)
        conv4 = LeakyReLU(alpha=5.5)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv5 = Conv2D(32, 3, kernel_initializer="uniform")(conv4)
        conv5 = LeakyReLU(alpha=5.5)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(32, 3, kernel_initializer="uniform")(conv5)
        conv6 = LeakyReLU(alpha=5.5)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv7 = Conv2D(64, 3, kernel_initializer="uniform")(conv6)
        conv7 = LeakyReLU(alpha=5.5)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv8 = Conv2D(64, 3, kernel_initializer="uniform")(conv7)
        conv8 = LeakyReLU(alpha=5.5)(conv8)
        conv8 = BatchNormalization()(conv8)
        drop1 = Dropout(0.2)(conv8)

        conv9 = Conv2D(128, 3, kernel_initializer="uniform")(drop1)
        conv9 = LeakyReLU(alpha=5.5)(conv9)
        conv9 = BatchNormalization()(conv9)


        deconv8a = Deconv2D(64, 3, kernel_initializer="uniform")(conv9)
        deconv8a = LeakyReLU(alpha=5.5)(deconv8a)
        deconv8a = BatchNormalization()(deconv8a)
        merge7 = multiply([conv8, deconv8a])
        deconv7a = Deconv2D(64, 3, kernel_initializer="uniform")(merge7)
        deconv7a = LeakyReLU(alpha=5.5)(deconv7a)
        deconv7a = BatchNormalization()(deconv7a)
        merge6 = multiply([conv7, deconv7a])
        deconv6a = Deconv2D(32, 3, kernel_initializer="uniform")(merge6)
        deconv6a = LeakyReLU(alpha=5.5)(deconv6a)
        deconv6a = BatchNormalization()(deconv6a)
        merge5 = multiply([conv6, deconv6a])
        deconv5a = Deconv2D(32, 3, kernel_initializer="uniform")(merge5)
        deconv5a = LeakyReLU(alpha=5.5)(deconv5a)
        deconv5a = BatchNormalization()(deconv5a)
        merge4 = multiply([conv5, deconv5a])
        deconv4a = Deconv2D(16, 3, kernel_initializer="uniform")(merge4)
        deconv4a = LeakyReLU(alpha=5.5)(deconv4a)
        deconv4a = BatchNormalization()(deconv4a)
        merge3 = multiply([conv4, deconv4a])
        deconv3a = Deconv2D(16, 3, kernel_initializer="uniform")(merge3)
        deconv3a = LeakyReLU(alpha=5.5)(deconv3a)
        deconv3a = BatchNormalization()(deconv3a)
        merge2 = multiply([conv3, deconv3a])
        deconv2a = Deconv2D(8, 3, kernel_initializer="uniform")(merge2)
        deconv2a = LeakyReLU(alpha=5.5)(deconv2a)
        deconv2a = BatchNormalization()(deconv2a)
        merge1 = multiply([conv2, deconv2a])
        deconv1a = Deconv2D(8, 3, kernel_initializer="uniform")(merge1)
        deconv1a = LeakyReLU(alpha=5.5)(deconv1a)
        deconv1a = BatchNormalization()(deconv1a)

        flat = Flatten()(deconv1a)
        dense = Dense(729, activation="relu")(flat)
        outputs = Reshape((9, 9, 9))(dense)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        model.load_weights("model/checkpoint.hdf5")
        model.compile(optimizer = Nadam(lr=1e-5), loss = IoU, metrics = ["MAE", mean_squared_error, IoU_metric, clinic_loss])

        return model

    def train(self, trainingPath, targetPath, mode = "dir", name="model"):
        """
        :param trainingPath: path to the npz file of the training data or path to the directory of the training data
        :param targetPath: path to the npz file of the target data or path to the directory of the target data
        :param mode: decide wether to use npz or direct access mode
        :param appendix: appendix will be added at the end of the filename as an extension
        :param boost: upsamling the data (true/false)
        :param fac: upsampling factor
        :param amount: amount of files to read to build the dataset
        :param norm: normalization (true/false)
        :param map64: files are 9x9x9 -> 90x90x90, upsamling to 63x63x63 leads to an even representation of 64x64x64 by adding 0 as parameter
        Example
        object = uNet()
        object.train("data/density/", "data/kernel/")
        :return:
        """
        trainingPath=trainingPath.replace("._","")
        targetPath=targetPath.replace("._","")
        trainingData = self.load(trainingPath, "density_f", appendix="", norm=True)
        targetData = self.load(targetPath, "kernel_f", appendix="", norm=True)


        # separating the data
        quantil_upper = len(trainingData) - (round(len(trainingData) * self.trainQuantil))
        x = trainingData[0:quantil_upper]
        y = targetData[0:quantil_upper]
        x_valid = trainingData[quantil_upper:len(trainingData)]
        y_valid = targetData[quantil_upper:len(targetData)]

        # training the model
        early_stopping = EarlyStopping(monitor='loss', min_delta=1e-6, patience=50, verbose=1, mode='auto')
        model_checkpoint = ModelCheckpoint('./model/checkpoint.hdf5', monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=cfg.settings['period'])
        csv_logger = CSVLogger('./model/model_1.log')

        model = self.uNet(trainingData)
        model_callbacks = [model_checkpoint, csv_logger, early_stopping]
        model.fit(x, y, validation_data=(x_valid, y_valid), batch_size=self.batch_size, epochs=self.epochs, verbose=cfg.settings['verbose'], shuffle=True, callbacks=model_callbacks)
        model.save('model/' + name + '.h5')

        print("Finished with the training.")
        return print("Done!")

    def predict(self, model, trainingPath, targetPath, organ="pizza", save = False, x = 55, y = 5, plot = True):
        """
        :param model: path to the saved keras model
        :param train: path to the .mat training file
        :return: goal: path to the .mat target file
        Example command:
        object = uNet()
        object.predict("model/simpleModel.h5", "data/density/dense_0.mat", "data/kernel/kernel_0.mat")
        """
        model = load_model(model, custom_objects={"IoU": IoU, "IoU_metric": IoU_metric, "clinic_loss": clinic_loss})
        targetData = self.load(targetPath, "kernel_f", appendix="", norm=True)
        trainingData = self.load(trainingPath, "density_f", appendix="", norm=True)

        prediction = model.predict(trainingData[5000:5100])

        if plot == True:
            data = [trainingData[x][y], targetData[x][y], prediction[x][y], np.absolute(targetData[x][y] - prediction[x][y])]
            vs.show_field(data, organ=organ)
        if save == True:
            sio.savemat('kernel_theresa.mat', {'kernel_prediction': prediction})
        return prediction



obj = uNet()
obj.train("data/density/", "data/kernel/")
#prediction = obj.predict("model/checkpoint.hdf5", "data/density/", "data/kernel/")
