import cv2
import numpy as np
from osgeo import gdal
import  pandas as pd
import cv2
import time
import os, shutil

def dataProcessing(img_file, gt_file):
    print('=== Start dataProcessing... ===')
    num_shape = 13
    num_pad = num_shape//2
    G022_city = gdal.Open("%s" %img_file)
    NDVI_G022_city_0133877 = gdal.Open("%s" %gt_file)

    G022_city_b1 = G022_city.GetRasterBand(1).ReadAsArray()[:, :, np.newaxis]
    G022_city_b2 = G022_city.GetRasterBand(2).ReadAsArray()[:, :, np.newaxis]
    G022_city_b3 = G022_city.GetRasterBand(3).ReadAsArray()[:, :, np.newaxis]
    G022_city_b4 = G022_city.GetRasterBand(4).ReadAsArray()[:, :, np.newaxis]
    NDVI_G022_city_0133877_b1 = NDVI_G022_city_0133877.GetRasterBand(1).ReadAsArray()

    hight  = np.shape(G022_city_b1)[0]
    width = np.shape(G022_city_b1)[1]
    print(hight, width, np.shape(NDVI_G022_city_0133877_b1)[0], np.shape(NDVI_G022_city_0133877_b1)[1])

    G022_city_b1 = np.pad(G022_city_b1, ((num_pad, num_pad), (num_pad, num_pad), (0, 0)), 'constant')
    G022_city_b2 = np.pad(G022_city_b2, ((num_pad, num_pad), (num_pad, num_pad), (0, 0)), 'constant')
    G022_city_b3 = np.pad(G022_city_b3, ((num_pad, num_pad), (num_pad, num_pad), (0, 0)), 'constant')
    G022_city_b4 = np.pad(G022_city_b4, ((num_pad, num_pad), (num_pad, num_pad), (0, 0)), 'constant')

    G022_city = np.concatenate((G022_city_b1, G022_city_b2, G022_city_b3, G022_city_b4), axis=2)
    print(np.shape(G022_city))

    index0 = []
    index1 = []
    num_0 = 201438
    num_1 = 90714
    df0 = np.zeros((num_0, num_shape, num_shape, 4))
    df1 = np.zeros((num_1, num_shape, num_shape, 4))
    gt0 = np.zeros((num_0), dtype=int)
    gt1 = np.ones((num_1),  dtype=int)
    num_index_0 = 0
    num_index_1 = 0
    for i in range(0, hight):
        for j in range(0, width):
            index = i*width+j
            if NDVI_G022_city_0133877_b1[i, j] == 0:
                i = i+num_pad
                j = j+num_pad
                temp = G022_city[i-num_pad:i+num_pad+1, j-num_pad:j+num_pad+1, :]
                df0[num_index_0, :, :, :] = temp.reshape((1, num_shape, num_shape, 4))
                index0.append(index)
                i = i-num_pad
                j = j-num_pad
                num_index_0 += 1

            elif NDVI_G022_city_0133877_b1[i, j] == 1:
                i = i+num_pad
                j = j+num_pad
                temp = G022_city[i-num_pad:i+num_pad+1, j-num_pad:j+num_pad+1, :]
                df1[num_index_1, :, :, :] = temp.reshape((1, num_shape, num_shape, 4))
                index1.append(index)
                i = i-num_pad
                j = j-num_pad
                num_index_1 += 1


    # gt0 = np.zeros((201438))
    # gt1 = np.ones((90714))
    df = np.concatenate((df1, df0), axis=0)
    gt = np.concatenate((gt1, gt0), axis=0)
    np.savez_compressed('data/Human_2D_data_size%s.npz' %num_shape, df)
    np.savez_compressed('data/Human_2D_gt_size%s.npz' %num_shape, gt)

    if img_file.split('_')[3].split('.')[0] == 'Orig':
        np.savez_compressed('data/Human_2D_data_size%s_Orig.npz' %num_shape, df)
        np.savez_compressed('data/Human_2D_gt_size%s_Orig.npz' %num_shape, gt)

    else:
        np.savez_compressed('data/Human_2D_data_size%s.npz' %num_shape, df)
        np.savez_compressed('data/Human_2D_gt_size%s.npz' %num_shape, gt)

    print('=== End of dataProcessing... ===')

# Main
if __name__ == '__main__':
    start_time = time.time()

    img_file = 'Image/G022_city_NoData_0.tif' # G022_city_NoData_0.tif
    gt_file  = 'Image/mosaic_1_0.tif'
    dataProcessing(img_file=img_file, gt_file=gt_file)

    end_time = time.time()
    print("Execute Time:", end_time - start_time)
