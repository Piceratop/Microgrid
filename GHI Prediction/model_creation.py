import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def create_cnn_lstm_model(input_shape, num_features):
    model = Sequential([
        Input(shape=input_shape),
        
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # LSTM layers for sequence learning
        LSTM(units=128, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(units=64),
        Dropout(0.3),
        
        # Output layer
        Dense(num_features)
    ])
    
    return model

def create_cnn_gru_model(input_shape, num_features):
    model = Sequential([
        Input(shape=input_shape),
        
        # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # GRU layers for sequence learning
        GRU(units=128, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        GRU(units=64),
        Dropout(0.3),
        
        # Output layer
        Dense(num_features)
    ])
    
    return model