import os
import matplotlib.pyplot as plt


if True:
    dir_ = 'checkpoints3'
    txt = os.path.join(os.getcwd(), dir_, 'records.txt')
    fh = open(txt, 'r')
    lines = fh.readlines()
    # loss = []
    loss_cnt = []
    loss_G = []
    loss_D = []
    i = 0

    for line in lines:
        # loss.append(float(line.split(" ")[-1][:-2]))
        tmp = line.split(",")
        loss_cnt.append(float(tmp[1].split(" ")[-1]))
        loss_G.append(float(tmp[2].split(" ")[-1]))
        loss_D.append(float(tmp[3].split(" ")[-1][:-1]))
        # loss.append(float(line.split(" ")[-1][:-1]))
        i += 1
    print(i)
    # plt.plot(range(i), loss)
    plt.plot(range(i), loss_cnt, label="Content loss")
    plt.plot(range(i), loss_G, label="Adversarial loss: G")
    plt.plot(range(i), loss_D, label="Adversarial loss: D")
    plt.legend()
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.margins(0,0)
    plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'loss.png'))