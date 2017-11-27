import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def loadData():
    ''' Load the data from .mat file'''

    #Load the data from .mat file
    content = sio.loadmat('SimDataMatrix_Completo.mat')
    data = content['SimDataMatrix']

    data = data.T

    #System inputs
    vc = data[2] #input voltage
    vl = data[8] #?

    #Outputs
    rpm = data[5] #rpm of the motor
    va = data[6] #voltage been applied in the motor
    ia = data[7] #ouput current

    return vc, vl, rpm, va, ia

def dataPreparation(vc,vl,rpm,va,ia):
    '''
    This function prepare the data for using in LSTMs
    inputs = the data that need to be prepared
    returns = the data prepared for using in the Keras library
    '''


    #Resphape the inputs to (n_obs, 1)
    vc = vc.reshape(vc.shape[0], 1)
    vl = vl.reshape(vl.shape[0], 1)

    rpm = rpm.reshape(rpm.shape[0], 1)
    va = va.reshape(va.shape[0], 1)
    ia = ia.reshape(ia.shape[0], 1)

    #Scale the features to a 0,1 range
    scaler = MinMaxScaler(feature_range=(0, 1))

    vc = scaler.fit_transform(vc)
    rpm = scaler.fit_transform(rpm)
    vl = scaler.fit_transform(vl)
    va = scaler.fit_transform(va)
    ia = scaler.fit_transform(ia)

    #Split into train and test sets
    x_train, x_test = vc[:45000], vc[45000:]

    y_train, y_test = rpm[:45000], rpm[45000:]

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    vc, vl, rpm, va, ia = loadData()
    x_train, x_test, y_train, y_test = dataPreparation(vc,vl,rpm,va,ia)

