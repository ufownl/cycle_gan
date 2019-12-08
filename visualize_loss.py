import re
import numpy as np
import matplotlib.pyplot as plt


def visualize(lines):
    regex = re.compile("^\[Epoch ([0-9]+)  Batch ([0-9]+)\]  dis_a_loss (\S+)  dis_b_loss (\S+)  gen_loss (\S+)")
    batch_x = []
    batch_dis_a_loss = []
    batch_dis_b_loss = []
    batch_gen_loss = []
    for line in lines:
        m = regex.match(line)
        if m:
            batch_x.append((int(m.group(1)), int(m.group(2))))
            batch_dis_a_loss.append(float(m.group(3)))
            batch_dis_b_loss.append(float(m.group(4)))
            batch_gen_loss.append(float(m.group(5)))
    batches = max(batch_x, key=lambda x: x[1])[1]
    batch_x = [epoch + batch / batches for epoch, batch in batch_x]
    regex = re.compile("^\[Epoch ([0-9]+)\]  training_dis_a_loss (\S+)  training_dis_b_loss (\S+)  training_gen_loss (\S+)")
    epoch_x = []
    training_dis_a_loss = []
    training_dis_b_loss = []
    training_gen_loss = []
    for line in lines:
        m = regex.match(line)
        if m:
            epoch_x.append(int(m.group(1)))
            training_dis_a_loss.append(float(m.group(2)))
            training_dis_b_loss.append(float(m.group(3)))
            training_gen_loss.append(float(m.group(4)))
    plt.subplot(2, 1, 1)
    plt.plot(np.array(batch_x), np.array(batch_dis_a_loss), label="batch dis_a loss")
    plt.plot(np.array(batch_x), np.array(batch_dis_b_loss), label="batch dis_b loss")
    plt.plot(np.array(batch_x), np.array(batch_gen_loss), label="batch gan loss")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(np.array(epoch_x), np.array(training_dis_a_loss), label="training dis_a loss")
    plt.plot(np.array(epoch_x), np.array(training_dis_b_loss), label="training dis_b loss")
    plt.plot(np.array(epoch_x), np.array(training_gen_loss), label="training gan loss")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lines = []
    while True:
        try:
            lines.append(input())
        except EOFError:
            break
    visualize(lines)
