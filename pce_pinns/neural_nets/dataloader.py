
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def init_dataloader(X, Y, batch_size, test_size=0.1, val_size=0.1, shuffle=False, seed=0):
    """
    Initialize the dataloader
    
    Note that we're taking care that data is not shuffled within one batch to, e.g., retain flattened spatiotemporal dimensions

    Args:
        X np.array((n_data, dim_in))
        Y np.array((n_data, dim_out))
        batch_size int: Batch size
        test_size float: Size of test dataset in percent of total dataset
        val_size float: Size of validation dataset in percent of total dataset
        shuffle bool: If True, shuffles dataset. The option is not true, s.t., the rand instance in train and test can be correlated
    
    Returns:
        train_loader DataLoader
        val_loader DataLoader
        test_loader DataLoader
    """
    if test_size*X.shape[0]%batch_size != 0. or val_size*X.shape[0]%batch_size != 0.:
        print('test size %0.3f, val_size %0.3f, batch_size %d, n_samples %d'%(test_size, val_size, batch_size, X.shape[0]))
        raise ValueError('Enter test or val size as valid percentage')

    if test_size > 0:
        X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=False, random_state=seed)
        test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
        val_size = val_size/(1.-test_size)
    else:
        test_loader = None

    if val_size > 0:
        X, X_val, Y, Y_val = train_test_split(X, Y, test_size=val_size, shuffle=False, random_state=seed+1)
        val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        val_loader = None

    train_dataset = RegressionDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader
