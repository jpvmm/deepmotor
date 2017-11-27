from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from time import time

from keras.callbacks import TensorBoard
import numpy as np



def build_model(x_train,x_test,y_train,y_test):
    '''
    Build LSTM model
    :return: LSTM model
    '''

    #Reshape data (only for test)

    x_train = x_train.reshape(x_train.shape[0], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, 1)

    model = Sequential()
    model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))

    model.compile(loss='mae', optimizer='adam')


    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model.fit(x_train, y_train, epochs=50, batch_size=30, validation_data=(x_test, y_test), verbose=1, callbacks=[tensorboard])


    return model


