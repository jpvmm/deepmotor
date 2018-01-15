import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler


def loadData():
    ''' Load the data from .mat file'''

    #Load the data from .mat file
    content = sio.loadmat('SimDataMatrix_Completo.mat')
    data = content['SimDataMatrix']

    data = data.T

    #Scale the feature between (0,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled =  scaler.fit_transform(data[0:8]) #Normalize all features less the vl

    #System inputs
    vc = data_scaled[2] #input voltage
    vl = data[8] #Load in the motor

    #Outputs
    rpm = data_scaled[5] #rpm of the motor
    va = data_scaled[6] #voltage been applied in the motor
    ia = data_scaled[7] #ouput current

    return vc, vl, rpm, va, ia

def createLaggedFrames(vc,vl,rpm,va,ia):
    '''
    This function create lagged frames based on the values of input.
    inputs = the data that need to be prepared
    returns = the data prepared for using in the Keras library
    '''

    #Making Frames to everyone
    vc_frame = DataFrame()
    vl_frame = DataFrame()
    rpm_frame = DataFrame()
    va_frame = DataFrame()
    ia_frame = DataFrame()

    vc_frame['vc(t)'] = vc
    vl_frame['vl(t)'] = vl
    rpm_frame['rpm(t)'] = rpm
    va_frame['va(t)'] = va
    ia_frame['ia(t)'] = ia

    #input sequence
    vc_frame['vc(t-1)'] = vc_frame['vc(t)'].shift(1)
    vl_frame['vl(t-1)'] = vl_frame['vl(t)'].shift(1)

    #forecast sequence

    rpm_frame['rpm(t+1)'] = rpm_frame['rpm(t)'].shift(-1)
    va_frame['va(t+1)'] = va_frame['va(t)'].shift(-1)
    ia_frame['ia(t+1)'] = ia_frame['ia(t)'].shift(-1)


    conc = concat([vc_frame['vc(t-1)'], vl_frame['vl(t-1)'],
                   rpm_frame['rpm(t+1)'], va_frame['va(t+1)'],
                   ia_frame['ia(t+1)']], axis=1)

    conc.dropna(inplace= True)

    return conc


def createFrames(vc,vl,rpm,va,ia):
    '''This function creates a data frame without lag in thge data'''

    # Making Frames to everyone
    vc_frame = DataFrame()
    vl_frame = DataFrame()
    rpm_frame = DataFrame()
    va_frame = DataFrame()
    ia_frame = DataFrame()

    vc_frame['vc(t)'] = vc
    vl_frame['vl(t)'] = vl
    rpm_frame['rpm(t)'] = rpm
    va_frame['va(t)'] = va
    ia_frame['ia(t)'] = ia

    conc = concat([vc_frame['vc(t)'], vl_frame['vl(t)'],
                   rpm_frame['rpm(t)'], va_frame['va(t)'], ia_frame['ia(t)']], axis=1)

    conc.dropna(inplace=True)

    return conc


def dataPreparation(conc,train_size, lagged = None):
    ''' Splits Data into train and test sets
        inputs = Pandas DataFrame containing the data, percentage to training
        output = train and test sets
    '''

    if lagged == True:
        #get only the Xs
        X = conc[['vc(t-1)', 'vl(t-1)']]
        X = X.values
    else:
        X = conc[['vc(t)', 'vl(t)']]
        X = X.values

    look_back = int(len(X) * train_size)

    x_train, x_test = X[:look_back:], X[look_back:]

    x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])



    #Get the ys

    if lagged == True:

        y = conc['rpm(t+1)']
        y = y.values
    else:
        y = conc['rpm(t)']
        y = y.values

    y_train, y_test = y[:look_back:], y[look_back:]



    return x_train,x_test,y_train,y_test




if __name__ == '__main__':
    vc, vl, rpm, va, ia = loadData()
    data = createFrames(vc,vl,rpm,va,ia)
    x_train,x_test,y_train,y_test = dataPreparation(data, 0.8)

