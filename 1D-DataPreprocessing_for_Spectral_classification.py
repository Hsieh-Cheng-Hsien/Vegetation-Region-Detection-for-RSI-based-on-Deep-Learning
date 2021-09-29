import cv2
import numpy as np
from osgeo import gdal
import  pandas as pd
from PIL import Image
import time

def dataProcessing(img_file1, img_file2):
    Green_img_file   = gdal.Open("%s" %img_file1)
    NoGreen_img_file = gdal.Open("%s" %img_file2)

    Green_band1 = Green_img_file.GetRasterBand(1).ReadAsArray()[:, :, np.newaxis]
    Green_band2 = Green_img_file.GetRasterBand(2).ReadAsArray()[:, :, np.newaxis]
    Green_band3 = Green_img_file.GetRasterBand(3).ReadAsArray()[:, :, np.newaxis]
    Green_band4 = Green_img_file.GetRasterBand(4).ReadAsArray()[:, :, np.newaxis]
    Green_bands = np.concatenate((Green_band1, Green_band2, Green_band3, Green_band4), axis=2)
    print(type(Green_bands), np.shape(Green_bands))
    #因為植生影像左下角缺一排和一列，所以這邊補NA值；補NA值參數：((上, 下), (左, 右))，這邊補最下一列(row)和最左一行(column)
    Green_bands = np.pad(Green_bands, ((0, 1), (1, 0), (0, 0)), 'constant', constant_values=((0, -3.40282306074e+38), (-3.40282306074e+38, 0), (0, 0)))
    print(type(Green_bands), np.shape(Green_bands))

    NoGreen_band1 = NoGreen_img_file.GetRasterBand(1).ReadAsArray()[:, :, np.newaxis]
    NoGreen_band2 = NoGreen_img_file.GetRasterBand(2).ReadAsArray()[:, :, np.newaxis]
    NoGreen_band3 = NoGreen_img_file.GetRasterBand(3).ReadAsArray()[:, :, np.newaxis]
    NoGreen_band4 = NoGreen_img_file.GetRasterBand(4).ReadAsArray()[:, :, np.newaxis]
    NoGreen_bands = np.concatenate((NoGreen_band1, NoGreen_band2, NoGreen_band3, NoGreen_band4), axis=2)
    print(type(NoGreen_bands), np.shape(NoGreen_bands))
    # #因為植生影像左下角缺一排和一列，所以這邊補NA值；補NA值參數：((上, 下), (左, 右))，這邊補最下一列(row)和最左一行(column)
    # NoGreen_bands = np.pad(NoGreen_bands, ((0, 1), (1, 0), (0, 0)), 'constant', constant_values=((0, -3.40282306074e+38), (-3.40282306074e+38, 0), (0, 0)))
    print(type(NoGreen_bands), np.shape(NoGreen_bands))

    # n_Green   = np.sum(Green_bands[:][:][:] >= 0)
    # n_NoGreen = np.sum(NoGreen_bands >= 0)
    rows = np.shape(Green_bands)[0]
    cols = np.shape(Green_bands)[1]
    print(rows, cols)
    n_Green   = np.sum(np.logical_and(Green_band1 >= 0, Green_band1!=65535))
    n_NoGreen = np.sum(np.logical_and(NoGreen_band1 >= 0, NoGreen_band1!=65535))

    print(n_Green, n_NoGreen)
    print(type(n_Green), type(n_NoGreen))
    temp1 = np.empty([n_Green, 1], dtype= np.uint16)
    temp2 = np.empty([n_NoGreen, 1], dtype=np.uint16)


    for band_index in range(0, 4, 1):
        Green_img   = Green_bands[:, :, band_index]
        NoGreen_img = NoGreen_bands[:, :, band_index]

        height = np.shape(Green_img)[0]
        width  = np.shape(Green_img)[1]
        height2 = np.shape(NoGreen_img)[0]
        width2  = np.shape(NoGreen_img)[1]

        index = 0
        Green = np.empty([1], dtype=np.uint16)
        position_index_1 = np.empty([1], dtype=np.uint16)
        position_index_0 = np.empty([1], dtype=np.uint16)
        print(np.shape(Green_img), np.shape(Green))
        for i in range(height):
            for j in range(width):
                if Green_img[i][j] >= 0:
                    if int(len(Green)-1)%256 == 0:
                        # print(len(Green))
                        index += 1
                    position_index_1 = np.append(position_index_1, np.array([i*width+j]).reshape(1))
                    Green = np.append(Green, Green_img[i][j].reshape(1))
                    Green_img[i][j] = 255
        im = Image.fromarray(Green_img).convert("L") #畫出 Class_1 示意圖
        im.save('data/Class_1示意圖.png')

        Green = np.delete(Green, 0, 0) # delete empty position
        Green = Green.reshape(len(Green), 1)
        position_index_1 = np.delete(position_index_1, 0, 0)
        print(np.shape(position_index_1))
        position_index_1 = position_index_1.reshape(len(position_index_1), 1)
        print(np.shape(position_index_1))

        if band_index == 0:
            gt1 = np.ones(len(Green))

        print(np.shape(Green))
        temp1 = np.append(temp1, Green, axis=1)

        NoGreen = np.empty([1])
        print(np.shape(NoGreen_img), np.shape(NoGreen))

        for i in range(height2):
            for j in range(width2):
                if NoGreen_img[i][j] >= 0 and NoGreen_img[i][j] != 65535:
                    position_index_0 = np.append(position_index_0, np.array([i*width2+j]).reshape(1))
                    NoGreen = np.append(NoGreen, NoGreen_img[i][j].reshape(1))
                    NoGreen_img[i][j] = 255
        im = Image.fromarray(NoGreen_img).convert("L") #畫出 Class_0 示意圖
        im.save('data/Class_0示意圖.png')

        NoGreen = np.delete(NoGreen, 0, 0) # delte empty position
        NoGreen = NoGreen.reshape(len(NoGreen), 1)

        position_index_0 = np.delete(position_index_0, 0, 0)
        print(np.shape(position_index_0))
        position_index_0 = position_index_0.reshape(len(position_index_0), 1)
        print(np.shape(position_index_0))

        if band_index == 0:
            gt0 = np.zeros(len(NoGreen))

        print(np.shape(NoGreen))
        temp2 = np.append(temp2, NoGreen, axis=1)

    temp1 = np.delete(temp1, 0, 1)
    temp2 = np.delete(temp2, 0, 1)
    data = np.append(temp1, temp2, axis=0)
    gt = np.append(gt1, gt0, axis=0)
    ps = np.append(position_index_1, position_index_0, axis=0)
    print(np.shape(ps))
    data = pd.DataFrame(data)
    gt = pd.DataFrame(gt)
    ps = pd.DataFrame(ps)
    if img_file1.split('_')[3].split('.')[0] == 'Orig':
        data.to_csv('data/%s_Orig_data_4Bands.csv'%(img_file1.split('_')[0]))
        gt.to_csv('data/%s_Orig_gt_4Bands.csv'%(img_file1.split('_')[0]))
        ps.to_csv('data/%s_ps_4Bands.csv'%(img_file1.split('_')[0]))
        np.savez_compressed('data/%s_Orig_data_4Bands.npz' %(img_file1.split('_')[0]), data)
        np.savez_compressed('data/%s_Orig_gt_4Bands.npz' %(img_file1.split('_')[0]), gt)
        np.savez_compressed('data/%s_Orig_ps_4Bands.npz' %(img_file1.split('_')[0]), ps)
    else:
        data.to_csv('data/%s_data_4Bands.csv' %(img_file1.split('_')[0]))
        gt.to_csv('data/%s_gt_4Bands.csv' %(img_file1.split('_')[0]))
        ps.to_csv('data/%s_ps_4Bands.csv' %(img_file1.split('_')[0]))
        np.savez_compressed('data/%s_data_4Bands.npz' %(img_file1.split('_')[0]), data)
        np.savez_compressed('data/%s_gt_4Bands.npz' %(img_file1.split('_')[0]), gt)
        np.savez_compressed('data/%s_ps_4Bands.npz' %(img_file1.split('_')[0]), ps)

    # # Create ReadMe file
    # ReadMe = open('%s_ReadMe.txt'%(img_file1.split('_')[0]), 'w')
    # ReadMe.write('Data Size: %s\nGround Truth: %s\nData Info: Green%s NoGreen%s' %(np.shape(data), np.shape(gt), np.shape(Green), np.shape(NoGreen)))
    # ReadMe.close()

# Main
if __name__ == '__main__':

    img_file1 = 'Image/Human_Select_G022_Green.tif'
    img_file2 = 'Image/Human_Select_G022_NoGreen.tif'
    print("=========Training data processing=========")
    start_time = time.time()

    dataProcessing(img_file1=img_file1, img_file2=img_file2)

    end_time = time.time()
    print("Execute Time:", end_time - start_time)


