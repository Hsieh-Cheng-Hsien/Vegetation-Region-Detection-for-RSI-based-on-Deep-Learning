from densenet.classifiers.one_d import DenseNet121
import tensorflow as tf
import numpy as np
import pandas as pd
import os, time, datetime
start_time = time.time()
import cv2

num_ones = 90714
num_zeros = 201438
num_total = num_ones + num_zeros

df = np.load('../data/Human_data_4Bands.npz')['arr_0']
gt = np.load('../data/Human_gt_4Bands.npz')['arr_0']
print(np.shape(df), np.shape(gt))
df = df.reshape((np.shape(df)[0], 2, 2))

random_select_1 = pd.read_csv('../data/Random_Permutation_Position_1.csv').to_numpy()
random_select_0 = pd.read_csv('../data/Random_Permutation_Position_0.csv').to_numpy()

random_select_1 = np.delete(random_select_1, 0, axis=1).reshape((np.shape(random_select_1)[0]))
random_select_0 = np.delete(random_select_0, 0, axis=1).reshape((np.shape(random_select_0)[0]))

print(type(df), np.shape(df), type(gt), np.shape(gt), np.shape(random_select_1), np.shape(random_select_0))

x_train = np.concatenate((df[random_select_1[:num_ones//2]], df[random_select_0[:num_zeros//2]]), axis=0)
y_train = np.concatenate((np.ones((num_ones//2), dtype=int), np.zeros((num_zeros//2), dtype=int)), axis=0)
x_test = np.concatenate((df[random_select_1[num_ones//2:]], df[random_select_0[num_zeros//2:]]), axis=0)
y_test = np.concatenate((np.ones((num_ones//2), dtype=int), np.zeros((num_zeros//2), dtype=int)), axis=0)
ps = np.concatenate((random_select_1, random_select_0), axis=0)
print(np.shape(ps))
print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))
print(np.max(x_train), np.max(y_train), np.max(x_test), np.max(y_test))



model = DenseNet121(input_shape=(2, 2), num_outputs=2)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=None,
              weighted_metrics=None)


log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 定義TensorBoard對象.histogram_freq 如果設置為0，則不會計算直方圖。
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(x_train, y_train, validation_split=0.2, epochs=150, batch_size=num_total//2, callbacks=[tensorboard_callback]) #validation_split=0.2
print("\n\n------------------------------------------ Test Accuracy ------------------------------------------")
model.evaluate(x_test, y_test)


end_time = time.localtime()
model.save('%s_%s_%s_%s_%s_%s_binary_Human_Selection_Trained.h5' %(end_time[0], end_time[1], end_time[2], end_time[3], end_time[4], end_time[5]))
# ps = pd.DataFrame(ps)
# ps.to_csv('%s_%s_%s_%s_%s_%s_binary_Human_Selection_Trained_1_0.csv' %(end_time[0], end_time[1], end_time[2], end_time[3], end_time[4], end_time[5]))


end_time = time.time()
print(end_time-start_time)
