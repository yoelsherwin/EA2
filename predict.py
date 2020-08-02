import torch
import train
import data_loader as dl
import sys


def main(input, output, numlines):
    model = train.Model()
    model.load_state_dict(torch.load("best.pt"))
    file = open(output, 'w')
    data = torch.utils.data.DataLoader(dl.MyDataLoader(input, num_lines=numlines),
                                      batch_size=dl.train_batch,
                                      shuffle=False,
                                      pin_memory=False)
    for x,y in data:
        temp = model(x)
        temp = temp.argmax(dim=1)
        for i in range(len(temp)):
            if (temp[i] == 1):
                file.write("1\n")
            else:
                file.write("0\n")
    file.close()



if __name__ == "__main__":
    if (len(sys.argv) > 3):
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # change here input, output and number of lines
        main("input.csv", "output.txt", 500000)