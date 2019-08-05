import numpy as np

class TWINS_loader:

    def __init__(self, file_name, split_type, in_sample=False, batch_size=32):
        """dataloader for TWINS dataset
        
        Arguments:
            file_name {str} -- datapath for TWINS dataset
            split_type {str} -- split type of dataloader. train/valid/test
        
        Keyword Arguments:
            in_sample {bool} -- for test dataloader we distinguish in/out-sample error (default: {False})
            batch_size {int} -- mini-batch size (default: {32})
        """
        assert (split_type == 'train') or (split_type == 'valid') or (split_type == 'test')
        if in_sample:
            assert split_type == 'test'
        
        self.split_type = split_type
        self.batch_size = batch_size

        # set data_path
        if self.split_type=='train' or self.split_type=='valid':
            file_name = file_name + '.train.npz'
        elif self.split_type=='test':
            if in_sample:
                file_name = file_name + '.train.npz'
            else:
                file_name = file_name + '.test.npz'

        # get data
        self.data = np.load(file_name)
        self.x, self.t, self.yf, self.ycf = None, None, None, None

    def set_id(self, i):
        """modify repetition number for dataloader
        
        Arguments:
            i {int} -- repetition id
        """
        self.x = self.data['x'][:,:,i]
        self.t = self.data['t'][:,i]
        self.yf = self.data['yf'][:,i]
        self.ycf = self.data['ycf'][:,i]

        self.train_size = int(0.7 * len(self.x))
        if type=='train':
            self.x = self.x[:self.train_size]
            self.t = self.t[:self.train_size]
            self.yf = self.yf[:self.train_size]
            self.ycf = self.ycf[:self.train_size]
        elif type=='valid':
            self.x = self.x[self.train_size:]
            self.t = self.t[self.train_size:]
            self.yf = self.yf[self.train_size:]
            self.ycf = self.ycf[self.train_size:]

        self.input_dim = self.x.shape[1]

        self.x_mean = np.mean(self.x, axis=0)
        self.yf_0_mean = np.mean(self.yf[self.t==0], axis=0)
        self.yf_1_mean = np.mean(self.yf[self.t==1], axis=0)
        return None

    def __next__(self):
        # work as generator for time efficiency
        if self.split_type=='train':
            mb_arr = np.random.choice(list(range(self.train_size)), self.batch_size)
            mb_x = self.x[mb_arr]
            mb_t = self.t[mb_arr]
            mb_yf = self.yf[mb_arr]
            mb_ycf = self.ycf[mb_arr]
            return mb_x, mb_t, mb_yf, mb_ycf

        else:
            return self.x, self.t, self.yf, self.ycf