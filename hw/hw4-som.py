from collections import defaultdict, Counter

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from matplotlib import pyplot as plt
# import matplotlib
# matplotlib.use("Qt4agg")

# %%

xs = np.load('./hw/Alphabets.npy')
ys = np.load('./hw/Alphabet_labels.npy')

xs = xs / xs.max()

dim = xs.shape[1]

# %%
d = 25
# weights = np.random.uniform(-0.01, 0.01, (d, d, dim))
weights = xs[np.random.choice(xs.shape[0], d ** 2, replace=True), :].reshape(d, d, -1)


# def show():
#     d2 = [[Counter() for _ in range(d)] for __ in range(d)]
#
#     for x, y in zip(xs, ys):
#         D = ((weights.reshape(d ** 2, -1) - x) ** 2).sum(axis=1)
#         index = np.argmin(D)
#         i = index // d
#         j = index % d
#         d2[i][j][y] += 1
#
#     d3 = np.array([[max(a.items(), key=lambda x : x[1])[0]+1 if a.items() else 0 for a in b] for b in d2])
#     plt.imshow(d3)
#     plt.show()
#     return d2

def show2():
    im = np.zeros((d * 28, d * 28))
    a = weights.reshape(d, d, 28, 28)
    for i in range(d):
        for j in range(d):
            im[28 * i:(28 * (i + 1)),28 * j:(28 * (j+1))] = a[i, j]
    plt.imshow(im)
    plt.show()


alpha_max = 1e-2
R_max = 4
num_epochs = 100

for i_epoch in tqdm(range(num_epochs)):
    decay = 1 - np.power(i_epoch / num_epochs, 2)
    alpha = alpha_max * decay
    R = int(R_max * decay)
    for x in shuffle(xs):
        D = ((weights.reshape(d ** 2, -1) - x) ** 2).sum(axis=1)
        index = np.argmin(D)
        i = index // d
        j = index % d
        weights[max([i - R, 0]):min([i + R + 1, d]), max([j - R, 0]):min([j + R + 1, d])] += \
            alpha * (x - weights[max([i - R, 0]):min([i + R + 1, d]), max([j - R, 0]):min([j + R + 1, d])])
    if i_epoch % 1 == 0:
        show2()
# %%

#%%
