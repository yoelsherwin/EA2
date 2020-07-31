import train
import torch
import copy
import data_loader as dl
import random

POP_SIZE = 5
MAX_GEN = 35
THRESHOLD = -0.35

fit = dl.fit
mut = dl.mut
#tl = dl.train_loader


def compute(model, data):
    TP = TN = FP = FN = 0
    for x,y in data:
        res = model(x)
        #res = res.argmax(dim=1)
        ans = []
        for i in range(len(res)):
            if res[i][0] + THRESHOLD < res[i][1]:
                ans.append(1)
            else:
                ans.append(0)
        res = ans
        for i in range(len(res)):
            if res[i] == y[i] and y[i] == 1:
                TP += 1
            elif res[i] == y[i] and y[i] == 0:
                TN += 1
            elif res[i] == 0 and y[i] == 1:
                FN += 1
            else:
                FP += 1
    if ((TP + FP) == 0):
        precision = 0
    else:
        precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if ((TP + FN) == 0):
        recall = 0
    else:
        recall = TP / (TP + FN)
    #if ((0.0156 * precision + recall) != 0):
    if ((0.0625 * precision + recall) != 0):
        Fbeta = (1.0625 * precision * recall) / (0.0625 * precision + recall)
     #   Fbeta = (1.0156 * precision * recall) / (0.0156 * precision + recall)
    else:
        Fbeta = 0
    print("TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN) +
          " recall: " + str(recall) + " precision: " + str(precision) +
          " accuracy: " + str(accuracy) + " Fbeta: " + str(Fbeta))
    return Fbeta


def run(pool):
    gen = 0
    all_time_max = -1
    hof = None
    stats = open("stats.csv", 'w')
    while True:
        best = -1
        max = -1
        all = 0
        for i in range(POP_SIZE):
            fitness = compute(pool[i], fit)
            all += fitness
            if fitness > max:
                best = i
                max = fitness
                if all_time_max < max:
                    all_time_max = max
                    hof = copy.deepcopy(pool[best])
                    torch.save(hof.state_dict(), "best.pt")
        all /= POP_SIZE
        stats.write(str(max) + "," + str(all) + "\n")
        print("\n********\nall time best: " + str(all_time_max) + ", curr best: " + str(max) + ", gen: " + str(
            gen) + "\n********\n")
        gen += 1
        if gen > MAX_GEN:
            break
        new_pool = []
        new_pool.append(pool[best])
        lengths = []
        lengths.append(15000)
        lengths.append(15000)
        lengths.append(15000)
        lengths.append(15000)
        a,b,c,d = torch.utils.data.random_split(mut, lengths)
        helper = []
        a = torch.utils.data.DataLoader(dl.MyDataset(a), batch_size=dl.train_batch, shuffle=True, pin_memory=False)
        b = torch.utils.data.DataLoader(dl.MyDataset(b), batch_size=dl.train_batch, shuffle=True, pin_memory=False)
        c = torch.utils.data.DataLoader(dl.MyDataset(c), batch_size=dl.train_batch, shuffle=True, pin_memory=False)
        d = torch.utils.data.DataLoader(dl.MyDataset(d), batch_size=dl.train_batch, shuffle=True, pin_memory=False)
        helper.append(a)
        helper.append(b)
        helper.append(c)
        helper.append(d)
        for i in range(POP_SIZE - 1):
            temp = copy.deepcopy(pool[best])
            train.train(temp, helper[i])
            new_pool.append(temp)
        pool = new_pool
    file = open("313326019_205385560_46.txt", 'w')
    for x, y in dl.test_data:
        temp = hof(x)
        temp = temp.argmax(dim=1)
        for i in range(len(temp)):
            if(temp[i] == 1):
                file.write("1\n")
            else:
                file.write("0\n")
    file.close()
    stats.close()


def main():
    random.seed(10)
    pool = []
    for i in range(POP_SIZE):
        pool.append(train.Model())
    run(pool)


if __name__ == "__main__":
    main()