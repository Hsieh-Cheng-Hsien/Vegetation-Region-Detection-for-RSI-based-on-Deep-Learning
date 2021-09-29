import tensorflow as tf
import numpy as np
import pandas as pd
import time
import cv2
tf.random.set_seed(1000)
start_time = time.time()

# Training data
df = pd.read_csv('data/Human_data_4Bands.csv').to_numpy()
gt = pd.read_csv('data/Human_gt_4Bands.csv').to_numpy()
ps = pd.read_csv('data/Human_ps_4Bands.csv').to_numpy()
print(np.shape(df), np.shape(gt))

# Delete index from row and column
df = np.delete(df, 0, axis=1)#.reshape((np.shape(df)[0], 2, 2))
gt = np.delete(gt, 0, axis=1) #.reshape(len(gt))
ps = np.delete(ps, 0, axis=1)
print(np.shape(df), np.shape(gt), np.shape(ps))

num_one = 90714
num_zero = 201438
num_total = num_one + num_zero

'''
# random_1 = np.random.permutation(range(num_one))
# random_0 = np.random.permutation(range(num_one, num_total))

# test_position = pd.read_csv('2021_6_4_20_23_28_binary_Human_Selection_Trained_1_0.csv').to_numpy()
# test_position = np.delete(test_position, 0, axis=1)
#
# print(np.shape(test_position))
# test_position_0  = test_position[num_one:].reshape(num_zero)
# test_position_1 = test_position[:num_one].reshape(num_one)
# print(np.shape(test_position_0 ), np.shape(test_position_1))

# random_1 = np.empty([1], dtype=np.uint16)
# random_0 = np.empty([1], dtype=np.uint16)
#
# for i in test_position_1:
#     random_1 = np.append(random_1, np.where(ps == i)[0][0])
# random_1 = np.delete(random_1, 0, 0)
# print(np.shape(random_1))
# random_1 = pd.DataFrame(random_1)
# random_1.to_csv('Random_Select_Position_1.csv')
#
# for i in test_position_0:
#     random_0 = np.append(random_0, np.where(ps == i)[0][0])
# random_0 = np.delete(random_0, 0, 0)
# print(np.shape(random_0))
# random_0 = pd.DataFrame(random_0)
# random_0.to_csv('Random_Select_Position_0.csv')
'''

random_1 = pd.read_csv('data/Random_Permutation_Position_1.csv').to_numpy()
random_0 = pd.read_csv('data/Random_Permutation_Position_0.csv').to_numpy()
random_1 = np.delete(random_1, 0, axis=1)
random_0 = np.delete(random_0, 0, axis=1)
print(np.shape(random_1), np.shape(random_0))

train_df_1 = df[random_1[:num_one//2]]
train_gt_1 = gt[random_1[:num_one//2]]
test_df_1  = df[random_1[num_one//2:]]
test_gt_1  = gt[random_1[num_one//2:]]
test_ps_1  = ps[random_1]

train_df_0 = df[random_0[:num_zero//2]]
train_gt_0 = gt[random_0[:num_zero//2]]
test_df_0  = df[random_0[num_zero//2:]]
test_gt_0  = gt[random_0[num_zero//2:]]
test_ps_0  = ps[random_0]

train_df = np.concatenate((train_df_1, train_df_0), axis=0)
train_gt = np.concatenate((train_gt_1, train_gt_0), axis=0)
test_df  = np.concatenate((test_df_1, test_df_0), axis=0)
test_gt  = np.concatenate((test_gt_1, test_gt_0), axis=0)
ps = np.append(test_ps_1, test_ps_0, axis=0)
print(type(train_df), type(test_gt), np.shape(train_df), np.shape(train_gt), np.shape(test_df), np.shape(test_gt))
print(np.shape(test_ps_0), np.shape(test_ps_1))
tmp = np.full((507015), 255)
tmp[ps[:num_zero]] = 0
tmp[ps[num_zero:]] = 1

# tmp = np.reshape(tmp, (593, 855))
# cv2.imwrite("data/Random_Select_position_MLP.png", tmp)

# Create network
model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(516, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(2, activation='softmax')
])

# 選擇損失函數、optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              loss_weights=None,
              weighted_metrics=None)

# 訓練模型
model.fit(train_df, train_gt, epochs=300)


print("\n\n------------------------ Test Accuracy ----------------------------")
model.evaluate(train_df, train_gt)

end_time = time.time()
print("Execution time: ", end_time - start_time)
end_time = time.localtime()
model.save('%s_%s_%s_%s_%s_%s_binary_discriminatorTrained_%s.h5' %(end_time[0], end_time[1], end_time[2], end_time[3], end_time[4], end_time[5], 'DenseLayer'))
