import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

tag_name = np.load("tag_name.npy").reshape(19, 1)
print(tag_name)
train_X = np.load("train_X.npy")
train_binary_Y = np.load("train_binary_Y.npy")
valid_test_X = np.load("valid_test_X.npy")
print(train_X[0][:, 0, :].shape)
# fig1 = plt.figure()
# plt.imshow(train_X[0][0, :, :], cmap="cool", interpolation="none")
# fig2 = plt.figure()
# plt.imshow(train_X[0][:, 0, :], cmap="jet", interpolation="none")
# fig3 = plt.figure()
# plt.imshow(train_X[0][:, :, 0], cmap="hot", interpolation="none")
# plt.show()
# print(train_binary_Y.shape)
min = []
max = []
for i in range(26):
    min.append(np.amin(train_X[2000][i, :, :]))
    max.append(np.amax(train_X[2000][i, :, :]))
print(np.array(min) * np.power(1, 15))
print(np.array(max) * np.power(1, 15))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = np.arange(train_X[0].shape[0])
# y = np.arange(train_X[0].shape[1])
# z = np.arange(train_X[0].shape[2])
# ax.scatter(x, y, z, c=train_X[0])
# plt.show()
# print(x)