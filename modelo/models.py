from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout

from time import time

from keras.callbacks import TensorBoard
import numpy as np
from data import *
import matplotlib.pyplot as plt


def build_model(x_train,x_test,y_train,y_test):
    '''
    Build LSTM model
    :return: Trained LSTM model
    '''


    model = Sequential()
    model.add(LSTM(120,
                   return_sequences= True,
                   activation='relu',
                   input_shape=(x_train.shape[1], x_train.shape[2])))

    model.add(Dropout(0.25))
    model.add(LSTM(80,
                   return_sequences=True,
                   activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(LSTM(60,
                   return_sequences=True,
                   activation='relu'))
    model.add(Dropout(0.25))
    model.add(LSTM(20,
                   return_sequences=True,
                   activation='relu'))
    model.add(Dropout(0.25))
    model.add(LSTM(10,
                   activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='rmsprop')


    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model.fit(x_train, y_train, epochs=200,
              batch_size=10, validation_data=(x_test, y_test),
              verbose=1, callbacks=[tensorboard])


    return model


if __name__ == '__main__':
    vc, vl, rpm, va, ia = loadData()
    data = createFrames(vc, vl, rpm, va, ia)
    x_train, x_test, y_train, y_test = dataPreparation(data, 0.8)

    model = build_model(x_train,x_test,y_train,y_test)
    print model.summary()
    yhat = model.predict(x_test)


    y_test = y_test.reshape(len(y_test), 1)
    plt.plot(yhat,'r')
    plt.plot(y_test,'b')
    plt.show()