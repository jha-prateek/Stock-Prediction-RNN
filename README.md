## Stock Prediction using Time Series Analysis

# MODEL
Layer (type)                 Output Shape              Param #   
=================================================================
gru_1 (GRU)                  (None, 1, 512)            794112    
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 512)            0         
_________________________________________________________________
gru_2 (GRU)                  (None, 256)               590592    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 257       
=================================================================
Total params: 1,384,961
Trainable params: 1,384,961
Non-trainable params: 0
_________________________________________________________________

# Validation
![alt text](https://github.com/jha-prateek/Stock-Prediction-RNN/blob/master/predicted_test.JPG)
