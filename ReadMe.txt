
子計畫五：以深度學習進行福衛五號綠地覆蓋偵測


============ 影像與資料集 ============

1.  data: 訓練/測試用的資料

2. Image: 遙測影像



============ 影像資料前處理 ============

首先執行對於Image資料夾內遙測影像的前處理: 

1. 1D-DataPreprocessing_for_Spectral_classification.py          >>> Human_data_4Bands.csv(.npz)、Human_gt_4Bands.csv(.npz)、Human_ps_4Bands.csv、Class_0示意圖.png、Class_1示意圖.png

2. 2D-DataPreprocessing_for_Spatial_Spectral_classification.py  >>> Human_2D_data_size13.npz、Human_2D_gt_size13.npz

3. Ramdom_Select_Position.py                                    >>> Random_Permutation_Position_0.csv、Random_Permutation_Position_1.csv、Random_Select_position_MLP.png



============ 模 型 訓 練 ============

使用data資料夾內處理過後的資料集進行訓練

1. Spectral_MLP.py 

2. Spatial-Spectral_DenseNet/Spatial-Spectral_DenseNet.py.py

3. Spectral DenseNet/Spectral DenseNet.py


