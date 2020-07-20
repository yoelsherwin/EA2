import train
import create_params
import random
import torch

POP_SIZE = 5
FILENAME = "train.csv"
SIZE = 3000
MAX_GEN = 30

data = create_params.special_normalize(FILENAME, SIZE)
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

mix(data, fit_data, mut_data)

def compute(model):
    global fit_data
    TP = TN = FP = FN = 0
    for x,y in data:
        res = model(x)
        res = res.argmax(dim=1)
        ans = y.argmx(dim=1)
        if res == ans == 1:
            TP += 1
        elif res == ans == 0:
            TN += 1
        elif res == 0 and ans == 1:
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
    while gen < MAX_GEN:
        fit_data = []
        mut_data = []
        mix(data, fit_data, mut_data)
        best = -1
        max = -1
        for i in range(POP_SIZE):
            fitness = compute(pool[i])
            if fitness > max:
                best = i
                max = fitness
        if gen % 5 == 0:
            torch.save(pool[best].)
        gen += 1


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