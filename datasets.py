import numpy as np
from sklearn.model_selection import train_test_split
import os
np.random.seed(2019)

class IHDP(object):
    def __init__(self, path_data="../data/IHDP/", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]
        self.data = np.load(self.path_data + 'ihdp_npci_1-1000.train.npz')
        self.data_test = np.load(self.path_data + 'ihdp_npci_1-1000.test.npz')

    def __iter__(self):
        for i in range(self.replications):
            data = self.data
            x = data['x'][:,:,i]
            t = data['t'][:,i]
            y = data['yf'][:,i]
            y_cf = data['ycf'][:,i]
            mu_0 = data['mu0'][:,i]
            mu_1 = data['mu1'][:,i]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = self.data
            x = data['x'][:,:,i]
            t = np.reshape(data['t'][:,i], (-1,1))
            y = np.reshape(data['yf'][:,i], (-1,1))
            y_cf = np.reshape(data['ycf'][:,i], (-1,1))
            mu_0 = np.reshape(data['mu0'][:,i], (-1,1))
            mu_1 = np.reshape(data['mu1'][:,i], (-1,1))

            data_test = self.data_test
            x_test = data_test['x'][:,:,i]
            t_test = np.reshape(data_test['t'][:,i], (-1,1))
            y_test = np.reshape(data_test['yf'][:,i], (-1,1))
            y_cf_test = np.reshape(data_test['ycf'][:,i], (-1,1))
            mu_0_test = np.reshape(data_test['mu0'][:,i], (-1,1))
            mu_1_test = np.reshape(data_test['mu1'][:,i], (-1,1))

            x[:, 13] -= 1
            x_test[:13] -= 1
            idxtrain, iva = train_test_split(np.arange(x.shape[0]), test_size=0.3)

            train = (x[idxtrain], t[idxtrain], y[idxtrain]), (y_cf[idxtrain], mu_0[idxtrain], mu_1[idxtrain])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x_test, t_test, y_test), (y_cf_test, mu_0_test, mu_1_test)


            yield train, valid, test, self.contfeats, self.binfeats

class TWINS(object):
    def __init__(self, path_data="../data/TWINS/", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are continuous
        self.contfeats = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 32, 33, 34, 35, 36, 37, 38, 39]
        # which features are binary
        self.binfeats = [ i for i in range(40) if i not in self.contfeats ]
        self.data = np.load(self.path_data + 'twins_1-10.train.npz')
        self.data_test = np.load(self.path_data + 'twins_1-10.test.npz')

    def __iter__(self):
        for i in range(self.replications):
            data = self.data
            x = data['x'][:,:,i]
            t = data['t'][:,i]
            y = data['yf'][:,i]
            y_cf = data['ycf'][:,i]

            mu0 = y * (1 - t) + y_cf * t
            mu1 = y_cf * (1 - t) + y * t

            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = self.data
            x = data['x'][:,:,i]
            t = np.reshape(data['t'][:,i], (-1,1))
            y = np.reshape(data['yf'][:,i], (-1,1))
            y_cf = np.reshape(data['ycf'][:,i], (-1,1))
            mu_0 = np.reshape(y * (1 - t) + y_cf * t, (-1,1))
            mu_1 = np.reshape(y_cf * (1 - t) + y * t, (-1,1))

            data_test = self.data_test
            x_test = data_test['x'][:,:,i]
            t_test = np.reshape(data_test['t'][:,i], (-1,1))
            y_test = np.reshape(data_test['yf'][:,i], (-1,1))
            y_cf_test = np.reshape(data_test['ycf'][:,i], (-1,1))
            mu_0_test = np.reshape(y_test * (1 - t_test) + y_cf_test * t_test, (-1,1))
            mu_1_test = np.reshape(y_cf_test * (1 - t_test) + y_test * t_test, (-1,1))

            idxtrain, iva = train_test_split(np.arange(x.shape[0]), test_size=0.3)

            train = (x[idxtrain], t[idxtrain], y[idxtrain]), (y_cf[idxtrain], mu_0[idxtrain], mu_1[idxtrain])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x_test, t_test, y_test), (y_cf_test, mu_0_test, mu_1_test)


            yield train, valid, test, self.contfeats, self.binfeats

class JOBS(object):
    def __init__(self, path_data="../data/JOBS/", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [ 2,3,4,5,13,14,16 ]
        # which features are continuous
        self.contfeats = [ i for i in range(17) if i not in self.binfeats ]
        self.data = np.load(self.path_data + 'jobs.train.npz')
        self.data_test = np.load(self.path_data + 'jobs.test.npz')

    def __iter__(self):
        for i in range(self.replications):
            data = self.data
            x = data['x'][:,:,i]
            t = data['t'][:,i]
            y = data['yf'][:,i]
            e = data['e'][:,i]

            yield (x, t, y), e

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = self.data
            x = data['x'][:,:,i]
            t = np.reshape(data['t'][:,i], (-1,1))
            y = np.reshape(data['yf'][:,i], (-1,1))
            e = np.reshape(data['e'][:,i], (-1,1))

            data_test = self.data_test
            x_test = data_test['x'][:,:,i]
            t_test = np.reshape(data_test['t'][:,i], (-1,1))
            y_test = np.reshape(data_test['yf'][:,i], (-1,1))
            e_test = np.reshape(data_test['e'][:,i], (-1,1))

            # validation set split
            idxtrain, iva = train_test_split(np.arange(x.shape[0]), test_size=0.3)

            train = (x[idxtrain], t[idxtrain], y[idxtrain]), e[idxtrain]
            valid = (x[iva], t[iva], y[iva]), e[iva]
            test = (x_test, t_test, y_test), e_test


            yield train, valid, test, self.contfeats, self.binfeats
