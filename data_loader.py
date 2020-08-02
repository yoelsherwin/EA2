import torch
import torch.utils.data as data
import create_params


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class MyDataLoader(torch.utils.data.Dataset):
    def __init__(self, file_path, num_lines=-1):
        self.items = create_params.special_normalize(file_path, num_lines)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]



train_batch = 2000
dataset = MyDataLoader("train.csv", num_lines=700000)
lengths = []
lengths.append(100000)
lengths.append(600000)

fit, mut = torch.utils.data.random_split(dataset, lengths)

fit = data.DataLoader(fit, batch_size=train_batch, shuffle=False, pin_memory=False)

test_data = data.DataLoader(MyDataLoader("test.csv", num_lines=50000),
                                      batch_size=train_batch,
                                      shuffle=False,
                                      pin_memory=False)

