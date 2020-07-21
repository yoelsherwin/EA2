import train
import create_params
import random
import torch
import copy
import data_loader as dl

POP_SIZE = 5
FILENAME = "train.csv"
SIZE = 100000
MAX_GEN = 30

data = dl.train_loader
fit_data = []
mut_data = []

def mix(full, fit, mut):
    fcount = mcount = 0
    for i in range(SIZE):
        if fcount >= SIZE // 2:
            mut.append(full[i])
            mcount += 1
        elif mcount >= SIZE // 2:
            fit.append(full[i])
            fcount += 1
        else:
            if random.randint(0, 1) == 1:
                mut.append(full[i])
                mcount += 1
            else:
                fit.append(full[i])
                fcount += 1

#mix(data, fit_data, mut_data)

def compute(model):
    global fit_data
    TP = TN = FP = FN = 0
    for x,y in data:
        res = model(x)
        res = res.argmax(dim=1)
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
    if ((0.0625 * precision + recall) != 0):
        Fbeta = (1.0625 * precision * recall) / (0.0625 * precision + recall)
    else:
        Fbeta = 0
    print("TP: " + str(TP) + " TN: " + str(TN) + " FP: " + str(FP) + " FN: " + str(FN) +
          " recall: " + str(recall) + " precision: " + str(precision) +
          " accuracy: " + str(accuracy) + " Fbeta: " + str(Fbeta))
    return Fbeta


def run(pool):
    global fit_data
    global mut_data
    gen = 0
    all_time_max = -1
    hof = None
    while True:
        fit_data = []
        mut_data = []
        #mix(data, fit_data, mut_data)
        best = -1
        max = -1
        for i in range(POP_SIZE):
            fitness = compute(pool[i])
            if fitness > max:
                best = i
                max = fitness
                if all_time_max < max:
                    all_time_max = max
                    hof = copy.deepcopy(pool[best])
                    torch.save(hof.state_dict(), "best.pt")
        print("\n********\nall time best: " + str(all_time_max) + ", curr best: " + str(max) + ", gen: " + str(
            gen) + "\n********\n")
        gen += 1
        if gen > MAX_GEN:
            break
        new_pool = []
        new_pool.append(pool[best])
        for i in range(POP_SIZE - 1):
            temp = copy.deepcopy(pool[best])
            train.train(temp)
            new_pool.append(temp)
        pool = new_pool
    file = open("313326019_205385560_15.txt", 'w')
    train_batch = 1500
    for x, y in dl.test_data:
        temp = hof(x)
        temp = temp.argmax(dim=1)
        for i in range(len(temp)):
            if(temp[i] == 1):
                file.write("1\n")
            else:
                file.write("0\n")
            #file.write(str(temp[i]) + "\n")
            #print(str(temp[i]) + "\n")
    file.close()


def main():
    pool = []
    for i in range(POP_SIZE):
        pool.append(train.Model())
    run(pool)


def split(data):
    a = []
    b = []
    c = []
    d = []
    ac = bc = cc = dc = 0
    for line in data:
        if (ac == bc == cc == SIZE // 8):
            d.append(line)
            dc += 1
        elif (ac == bc == dc == SIZE // 8):
            c.append(line)
            cc += 1
        elif (ac == cc == dc == SIZE // 8):
            b.append(line)
            bc += 1
        elif (bc == cc == dc == SIZE // 8):
            a.append(line)
            ac += 1
        elif (ac == bc == SIZE // 8):
            test = random.randint(0, 1)
            if test == 1:
                c.append(line)
                cc += 1
            else:
                d.append(line)
                dc += 1
        elif (ac == cc == SIZE // 8):
            test = random.randint(0, 1)
            if test == 1:
                b.append(line)
                bc += 1
            else:
                d.append(line)
                dc += 1
        elif (ac == dc == SIZE // 8):
            test = random.randint(0, 1)
            if test == 1:
                c.append(line)
                cc += 1
            else:
                b.append(line)
                bc += 1
        elif (bc == cc == SIZE // 8):
            test = random.randint(0, 1)
            if test == 1:
                a.append(line)
                ac += 1
            else:
                d.append(line)
                dc += 1
        elif (bc == dc == SIZE // 8):
            test = random.randint(0, 1)
            if test == 1:
                a.append(line)
                ac += 1
            else:
                c.append(line)
                cc += 1
        elif (cc == dc == SIZE // 8):
            test = random.randint(0, 1)
            if test == 1:
                a.append(line)
                ac += 1
            else:
                b.append(line)
                bc += 1
        elif (ac == SIZE // 8):
            test = random.randint(0, 2)
            if test == 0:
                b.append(line)
                bc += 1
            elif test == 1:
                c.append(line)
                cc += 1
            else:
                d.append(line)
                dc += 1
        elif (bc == SIZE // 8):
            test = random.randint(0, 2)
            if test == 0:
                a.append(line)
                ac += 1
            elif test == 1:
                c.append(line)
                cc += 1
            else:
                d.append(line)
                dc += 1
        elif (cc == SIZE // 8):
            test = random.randint(0, 2)
            if test == 0:
                b.append(line)
                bc += 1
            elif test == 1:
                a.append(line)
                ac += 1
            else:
                d.append(line)
                dc += 1
        elif (dc == SIZE // 8):
            test = random.randint(0, 2)
            if test == 0:
                b.append(line)
                bc += 1
            elif test == 1:
                c.append(line)
                cc += 1
            else:
                a.append(line)
                ac += 1
        else:
            test = random.randint(0, 4)
            if test == 0:
                a.append(line)
                ac += 1
            elif test == 1:
                b.append(line)
                bc += 1
            elif test == 2:
                c.append(line)
                cc += 1
            else:
                d.append(line)
                dc += 1
    return a, b, c, d





if __name__ == "__main__":
    main()