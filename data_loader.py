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
        """
        Create a DataLoader for a given file
        :param file_path: the path to file
        :param num_lines: max number of lines to read, if negative will read all lines
        """
        self.items = create_params.special_normalize(file_path, num_lines)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # tensor, label
        return self.items[index]


def parse_file(file_path, num_lines):
    with open(file_path, "r") as file:
        parsed = []

        for line in file:
            split = line.split(",")

            # grab class(label) and features, cast class to int and each feature to a float
            label = try_parse_int(split[0], default=0)
            features = list(map(float, split[1:]))

            # append to parsed the features and their matching label
            parsed.append((torch.tensor(features), label))

            # check if finished
            num_lines -= 1
            if num_lines == 0:
                break

        return parsed


def try_parse_int(string, default=0):
    try:
        return int(string)
    except:
        return default


train_batch = 10000
dataset = MyDataLoader("train.csv", num_lines=700000)
lengths = []
lengths.append(100000)
lengths.append(600000)
fit, mut = torch.utils.data.random_split(dataset, lengths)
fit = data.DataLoader(fit, batch_size=train_batch, shuffle=False, pin_memory=False)
#mut = data.DataLoader(mut, batch_size=train_batch, shuffle=True, pin_memory=True)

# train_loader = data.DataLoader(MyDataLoader("train.csv", num_lines=200000),
#                                       batch_size=train_batch,
#                                       shuffle=True,
#                                       pin_memory=True)

test_data = data.DataLoader(MyDataLoader("test.csv", num_lines=50000),
                                      batch_size=train_batch,
                                      shuffle=False,
                                      pin_memory=False)

