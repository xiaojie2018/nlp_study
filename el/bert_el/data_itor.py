# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/20 14:07
# software: PyCharm


from tqdm import tqdm, trange
import time


class DataGenerator:
    def __init__(self, data, batch_size=4):
        self.data = data
        self.batch_size = batch_size

        length = len(self.data)

        self.steps = length // self.batch_size
        if length % self.batch_size != 0:
            self.steps += 1
        print("self.steps: ", self.steps)

    def __len__(self):
        return self.steps

    def __iter__(self):
        # while True:
        idxs = range(len(self.data))
        # np.random.shuffle(idxs)
        num = 0
        X1 = []
        for s, i in enumerate(idxs):
            num += 1
            X1.append(self.data[i])

            if len(X1) == self.batch_size or i == idxs[-1]:

                yield X1
                # num += 1
                X1 = []

            #     if num > 50:
            #         break
            #
            # if num > 50:
            #     break


if __name__ == '__main__':
    train_data = ["a"+str(i) for i in range(100)]
    batch_size = 16
    train_d = DataGenerator(train_data, batch_size)
    # print(len(train_d))
    train_iterator = trange(int(5), desc="Epoch")
    for _ in train_iterator:
        for step, d in enumerate(train_d):
            print(step)
            print(d)
        print("*"*30)
        time.sleep(1)
