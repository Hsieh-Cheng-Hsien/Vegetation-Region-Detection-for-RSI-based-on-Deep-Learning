import tensorflow as tf
import numpy as np
from modules.MnistData import MnistData
from modules.Build_model import Build_net
import pandas as pd
import time, datetime
num_shape = 13
num_ones  = 90714
num_zeros = 201438
df = np.load('../data/Human_2D_data_size%s.npz' %num_shape)['arr_0']
gt = np.load('../data/Human_2D_gt_size%s.npz' %num_shape)['arr_0']

random_select_1 = pd.read_csv('../data/Random_Permutation_Position_1.csv').to_numpy()
random_select_0 = pd.read_csv('../data/Random_Permutation_Position_0.csv').to_numpy()
random_select_1 = np.delete(random_select_1, 0, axis=1).reshape((np.shape(random_select_1)[0]))
random_select_0 = np.delete(random_select_0, 0, axis=1).reshape((np.shape(random_select_0)[0]))


print(type(df), np.shape(df), type(gt), np.shape(gt), np.shape(random_select_1), np.shape(random_select_0))
print(np.shape(random_select_1)[0]//2)
print(np.shape(df[random_select_1[:num_ones//2]]))

x_train = np.concatenate((df[random_select_1[:num_ones//2]], df[random_select_0[:num_zeros//2]]), axis=0)

y_train = np.concatenate((np.ones((num_ones//2), dtype=int), np.zeros((num_zeros//2), dtype=int)), axis=0)

x_test = np.concatenate((df[random_select_1[num_ones//2:]], df[random_select_0[num_zeros//2:]]), axis=0)

y_test = np.concatenate((np.ones((num_ones//2), dtype=int), np.zeros((num_zeros//2), dtype=int)), axis=0)

print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))
print(np.max(x_train), np.max(y_train), np.max(x_test), np.max(y_test))




model = Build_net([num_shape, num_shape], 2, 64)
model.summary()

log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=0.01), # Adam Adadelta
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=150,  verbose=1, validation_split=0.2, shuffle=True, callbacks=[tensorboard_callback])

end_time = time.localtime()
model.save('%s_%s_%s_%s_%s_%s_DenseNet2D_binary_classification.h5' %(end_time[0], end_time[1], end_time[2], end_time[3], end_time[4], end_time[5]))

model.evaluate(x_test,  y_test, verbose=2)




