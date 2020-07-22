import torch
import train
import data_loader as dl


FILENAME = "train.csv"
SIZE = 10000

def main():
    model = train.Model()
    model.load_state_dict(torch.load("best.pt"))
    #model.eval()
    file = open("313326019_205385560_17.txt", 'w')
    data = dl.test_data
    for x,y in data:
        temp = model(x)
        temp = temp.argmax(dim=1)
        for i in range(len(temp)):
            if (temp[i] == 1):
                file.write("1\n")
            else:
                file.write("0\n")
            #file.write(str(temp[i]) + "\n")
            #print(str(temp[i]) + "\n")
    file.close()



if __name__ == "__main__":
    main()