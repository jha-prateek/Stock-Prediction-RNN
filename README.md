# Stock Prediction using Time Series Analysis
	Closing Price prediction of Yahoo stocks from 2010 - 2016 using Gated Recurrant Units	
	Model is already trained and saved in 'stock_price_GRU.h5' file	
	To obtain the trained model just comment out the lines 47-55 and 60-62, then uncomment the lines 57-58 to load 'stock_price_GRU.h5' file	

	Highly Recommend using GPU version of Tensorflow for running the model	

#### DATA
	INPUT_DATA
	date             open        low       high      close
	2010-01-04  16.940001  16.879999  17.200001  17.100000
	2010-01-05  17.219999  17.000000  17.230000  17.230000
	2010-01-06  17.170000  17.070000  17.299999  17.170000
	2010-01-07  16.809999  16.570000  16.900000  16.700001
	2010-01-08  16.680000  16.620001  16.760000  16.700001

	LABEL_DATA
	date		  close
	2010-01-04    17.230000
	2010-01-05    17.170000
	2010-01-06    16.700001
	2010-01-07    16.700001
	2010-01-08    16.740000

#### MODEL
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
gru_1 (GRU)                  (None, 1, 512)            794112    
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 512)            0         
_________________________________________________________________
gru_2 (GRU)                  (None, 256)               590592    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 257       
_________________________________________________________________
Total params: 1,384,961
Trainable params: 1,384,961
Non-trainable params: 0
_________________________________________________________________

#### TRAINING
	Epoch 500/500
	250/1061 [======>.......................] - ETA: 0s - loss: 7.2934e-04
	750/1061 [====================>.........] - ETA: 0s - loss: 6.7267e-04
	1061/1061 [==============================] - 0s 111us/step - loss: 6.4617e-04 - val_loss: 6.4601e-04

	32/582 [>.............................] - ETA: 0s
	352/582 [=================>............] - ETA: 0s
	582/582 [==============================] - 0s 154us/step
	Score: 0.000513115886573222	

#### RESULTS
    33% of Data used for Testing 
    Plot only shows the last points of test set and predicted values	
![alt text](https://github.com/jha-prateek/Stock-Prediction-RNN/blob/master/predicted_test.JPG)
