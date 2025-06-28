from torch.utils.data import Dataset
import joblib
import random
embedding_len = 1024

class MyDataSet(Dataset):
    def __init__(self, tensor_affinity_path, is_train=False, multi_chain=False):
        self.tensor_protein = joblib.load(tensor_affinity_path)
        self.is_train = is_train
        self.multi_chain = multi_chain

    def __len__(self):
        return len(self.tensor_protein)

    def __getitem__(self, index):
        data = self.tensor_protein[index][0]
        mean = self.tensor_protein[index][1]
        affinity = self.tensor_protein[index][2]
        if random.randint(0, 128) == 0 and self.is_train:
            self.exchange_origin_mut(data)
            self.exchange_origin_mut(mean)
            affinity *= -1
        return data, mean, affinity

    def exchange_origin_mut(self, data):
        if len(data) == 0:
            return
        if len(data) < 10:  # data is embedding
            if self.multi_chain:
                data[3:6], data[6:9] = data[6:9], data[3:6].clone()
            else:
                data[1], data[2] = data[2], data[1].clone()
        else:  # data is mean
            if self.multi_chain:
                data[3*embedding_len:6*embedding_len], data[6*embedding_len:9*embedding_len]\
                    = data[6*embedding_len:9*embedding_len], data[3*embedding_len:6*embedding_len].clone()
            else:
                data[embedding_len:2*embedding_len], data[2*embedding_len:]\
                    = data[2*embedding_len:], data[embedding_len:2*embedding_len].clone()