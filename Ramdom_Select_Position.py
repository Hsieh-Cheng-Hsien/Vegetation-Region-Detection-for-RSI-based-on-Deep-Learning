import numpy as np
import pandas as pd
import time
import cv2


def Random_Permutation(ps):

    # Delete index of row and column from .csv
    ps = np.delete(ps, 0, axis=1)
    print(np.shape(ps))
    num_ones = 90714
    num_zeros = 201438
    num_total = num_ones + num_zeros

    random_index_1 = np.random.permutation(range(num_ones))
    random_index_0 = np.random.permutation(range(num_ones, num_total))
    test_position_1  = ps[random_index_1].reshape(num_ones)
    test_position_0  = ps[random_index_0].reshape(num_zeros)
    # test_position = np.append(test_ps_1, test_ps_0, axis=0)

    random_1 = np.empty([1], dtype=np.uint16)
    random_0 = np.empty([1], dtype=np.uint16)
    #找尋Class_0 和 Class_1 的pixel位置index與
    for i in test_position_1:
        random_1 = np.append(random_1, np.where(ps == i)[0][0])
    random_1 = np.delete(random_1, 0, 0)
    print(np.shape(random_1))
    random_1_pd = pd.DataFrame(random_1)
    random_1_pd.to_csv('data/Random_Permutation_Position_1.csv')

    for i in test_position_0:
        random_0 = np.append(random_0, np.where(ps == i)[0][0])
    random_0 = np.delete(random_0, 0, 0)
    print(np.shape(random_0))
    random_0_pd = pd.DataFrame(random_0)
    random_0_pd.to_csv('data/Random_Permutation_Position_0.csv')

    # 畫出隨機選取的位置圖
    tmp = np.full((507015), 255)
    tmp[test_position_1[:num_ones]] = 1
    tmp[test_position_0[num_ones:]] = 0
    print(type(tmp))
    tmp = np.reshape(tmp, (593, 855))
    cv2.imwrite("data/Random_Select_position.png", tmp)

if __name__ == '__main__':
    # Training data
    ps = pd.read_csv('data/Human_ps_4Bands.csv').to_numpy()
    print(np.shape(ps))

    print("========= Random Permutation processing=========")
    start_time = time.time()

    Random_Permutation(ps)

    end_time = time.time()
    print("Execute Time:", end_time - start_time)
