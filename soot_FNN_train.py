# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 08:57:52 2023

@author: elusi
"""

import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import seaborn as sns
import tensorflow as tf

# Defining paths
database = pd.ExcelFile('C:/Users/elusi/Downloads/proj/dataset/dataset1.xlsx')
metrics_path = r'C:/Users/elusi/Downloads/proj/metrics.csv'
metrics_df = pd.read_csv(metrics_path)

# Defining callbacks
earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

reducelearning = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.2,
                                   patience=3,
                                   min_lr=0.000001,
                                   verbose=1)

# Defining cross-validations hyperparameters for model fine tuning
#callbacks = [earlystop, reducelearning]
#batch_sizes = [1, 5, 10, 32]
#activations = ['linear', LeakyReLU(alpha=0.1), LeakyReLU(alpha=0.01)]
#optimizers = ['adagrad', 'adam', 'rmsprop', 'nadam']

# Hyperparameters that were found optimal
callbacks = [reducelearning]
batch_sizes = [5]
activations = ['linear']
optimizers = ['rmsprop']

#__________________________________________________________________________________#
#______________________Data preprocessing and exploration__________________________#
#__________________________________________________________________________________#

# Creating 3d dataframe (10000, 66, 5) from the .xlsx database containing 10,000 worksheets containing (66,5) data each
sheets = database.sheet_names
dataframes = {}
for s in sheets:
    dataframes[s] = database.parse(s)
    
# Randomly pick 5 dataframes (out of 10,000) and plot their f/T & RGB
df_list = list(dataframes.values())
rand_df = random.sample(df_list, 5)

correlation_matrices = []

for i, df in enumerate(rand_df):
    # Get correlation matrix for each df
    corr_matrix = df.corr()
    correlation_matrices.append(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix for DataFrame {i + 1}')
    plt.show()
        
    # Plot fraction vol. and Temp.
    plt.figure(figsize=(12, 6))
    # First plot
    ax = plt.gca()
    line1, = ax.plot(df[df.columns[0]], label='f', color='darkviolet')
    ax.set_ylabel('Soot volume fraction', color='darkviolet')
    ax.tick_params(axis='y', labelcolor='darkviolet')
    # Second plot
    ax2 = ax.twinx()
    line2, = ax2.plot(df[df.columns[1]], label='T', color='firebrick')
    ax2.set_ylabel('Temperature (K)', color='firebrick', rotation=270)
    ax2.tick_params(axis='y', labelcolor='firebrick')
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.set_xlabel('Time')
    plt.title('Soot Volume Fraction and Temperature')
    plt.show()
    
    # Plot R G B pixels intensity
    plt.figure(figsize=(12, 6))
    plt.plot(df[df.columns[2]], label='Red', color='red', linewidth=2)
    plt.plot(df[df.columns[3]], label='Green', color='limegreen', linewidth=2)
    plt.plot(df[df.columns[4]], label='Blue', color='royalblue', linewidth=2)
    plt.xlabel('Pixel Index', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('RGB Pixel Intensity', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Split flames to 80% for training and 20% for testing
dataframes_keys = list(dataframes.keys())
random.shuffle(dataframes_keys)

split_ratio = int(0.8*len(dataframes_keys))
train_dataframes = {key: dataframes[key] for key in dataframes_keys[:split_ratio]}
test_dataframes = {key: dataframes[key] for key in dataframes_keys[split_ratio:]}

train_dataframes = {key: train_dataframes[key].sort_index() for key in sorted(train_dataframes)}
test_dataframes = {key: test_dataframes[key].sort_index() for key in sorted(test_dataframes)}
    
# for simulation only - randomly picking 100 dataframes (out of 8,000 train soots)
train_df_list = list(train_dataframes.values())

# split train data to X and Y1/Y2
x_df = []
for x in train_df_list:
    drop_cols = [0, 1]
    x = x.drop(x.columns[drop_cols], axis=1, inplace=False)
    x_df.append(x)
    
y1_df = []
for y1 in train_df_list:
    drop_cols = [1, 2, 3, 4]
    y1 = y1.drop(y1.columns[drop_cols], axis=1, inplace=False)
    y1_df.append(y1)

y2_df = []
for y2 in train_df_list:
    drop_cols = [0, 2, 3, 4]
    y2 = y2.drop(y2.columns[drop_cols], axis=1, inplace=False)
    y2_df.append(y2)

# stack X/Y1/Y2 to move from list to array of dataframes
concat_xdf = np.stack([df.values for df in x_df])
concat_y1df = np.stack([df.values for df in y1_df])
concat_y2df = np.stack([df.values for df in y2_df])

# Reshape the 3D arrays to 2D arrays for scaling
n_samples, n_timesteps, n_features = concat_y1df.shape
concat_y1df_2d = concat_y1df.reshape(n_samples * n_timesteps, n_features)
concat_y2df_2d = concat_y2df.reshape(n_samples * n_timesteps, n_features)

# Scale the 2D arrays
scaler_y1 = MinMaxScaler()
scaler_y2 = MinMaxScaler()
scaled_y1_2d = scaler_y1.fit_transform(concat_y1df_2d)
scaled_y2_2d = scaler_y2.fit_transform(concat_y2df_2d)

# Reshape the scaled 2D arrays back to 3D
scaled_y1 = scaled_y1_2d.reshape(n_samples, n_timesteps, n_features)
scaled_y2 = scaled_y2_2d.reshape(n_samples, n_timesteps, n_features)

# Defining output variables names
output1_name = 'vol'
output2_name = 'temp'

#__________________________________________________________________________________#
#________________________________Model training____________________________________#
#__________________________________________________________________________________#

for act in activations:
    for batch in batch_sizes:
        for opt in optimizers:

            # Clear Keras session and reset model weights
            K.clear_session()
            model = Sequential()
            
            # Input layer
            inputs = Input(concat_xdf.shape[1:])
            
            # FNN layers
            x = Dense(8, activation='relu')(inputs)
            x = Dense(128, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            x = Dense(1024, activation='relu')(x)
            
            # Output layers
            output1 = Dense(1, activation=act, name=output1_name)(x)
            output2 = Dense(1, activation=act, name=output2_name)(x)
            
            model = Model(inputs=inputs, outputs=[output1, output2])
            
            model.compile(optimizer=opt, loss={'vol': 'mse', 'temp': 'mse'}, metrics={'vol': ['mse'], 'temp': ['mse']})
            
            # Train the model
            history = model.fit(
                concat_xdf,
                {output1_name: scaled_y1, output2_name: scaled_y2},
                epochs=100,
                batch_size=batch,
                callbacks=callbacks,
                validation_split=0.2,
                verbose=1
            )
            
            # Model representation
            print(model.summary())
            tf.keras.utils.plot_model(
                model,
                to_file=metrics_path + 'model_plot.png',
                show_shapes=True,
                rankdir='LR',
                expand_nested=True,
                show_layer_activations=True,
                dpi=100
            )
            
            # Plot training & validation loss values
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(str(act) + ' loss for ' + str(batch) + ' batches')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            print("Average of history.history['loss']: " + str(np.mean(history.history['loss'])))
            print("Average of history.history['val_loss']: " + str(np.mean(history.history['val_loss'])))
            
            # Plot training & validation loss values
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['vol_mse'])
            plt.plot(history.history['val_vol_mse'])
            plt.title('Volume fraction loss for ' + str(batch) + ' batches')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            print("Average of history.history['vol_mse']: " + str(np.mean(history.history['vol_mse'])))
            print("Average of history.history['val_vol_mse']: " + str(np.mean(history.history['val_vol_mse'])))
            
            # Plot training & validation loss values
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['temp_mse'])
            plt.plot(history.history['val_temp_mse'])
            plt.title('Temperature loss for ' + str(batch) + ' batches')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            print("Average of history.history['temp_mse']: " + str(np.mean(history.history['temp_mse'])))
            print("Average of history.history['val_temp_mse']: " + str(np.mean(history.history['val_temp_mse'])))
            
            # Make prediction
            test_df_list = list(test_dataframes.values())
            
            # split test data to X and Y1/Y2
            test_x_df = []
            for x in test_df_list:
                drop_cols = [0, 1]
                x = x.drop(x.columns[drop_cols], axis=1, inplace=False)
                test_x_df.append(x)
                
            test_y1_df = []
            for y1 in test_df_list:
                drop_cols = [1, 2, 3, 4]
                y1 = y1.drop(y1.columns[drop_cols], axis=1, inplace=False)
                test_y1_df.append(y1)
            
            test_y2_df = []
            for y2 in test_df_list:
                drop_cols = [0, 2, 3, 4]
                y2 = y2.drop(y2.columns[drop_cols], axis=1, inplace=False)
                test_y2_df.append(y2)
            
            # stack X/Y1/Y2 to move from list to array of dataframes
            concat_test_xdf = np.stack([df.values for df in test_x_df])
            concat_test_y1df = np.stack([df.values for df in test_y1_df])
            concat_test_y2df = np.stack([df.values for df in test_y2_df])
            
            prediction = model.predict(concat_test_xdf)
            
            # After training, make predictions and reshape them to 2d arrays for inverse transform
            n_samples_test = concat_test_xdf.shape[0]
            pred_y1_2d = prediction[0].reshape(n_samples_test * n_timesteps, n_features)
            pred_y2_2d = prediction[1].reshape(n_samples_test * n_timesteps, n_features)
            
            # Inverse transform the predictions and reshape back to 3d (2000,66,1)
            pred_y1 = scaler_y1.inverse_transform(pred_y1_2d).reshape(n_samples_test, n_timesteps, n_features)
            pred_y2 = scaler_y2.inverse_transform(pred_y2_2d).reshape(n_samples_test, n_timesteps, n_features)
            
            # Define a color palette
            colors = ['darkred', 'lightcoral', 'darkblue', 'lightskyblue', 'darkgreen', 'lightgreen', 'darkorange', 'lightyellow']

            # Sample indices
            sample_indices = random.sample(range(2000), 5)
            plt.figure(figsize=(15, 5))

            # Plot pred_y1 against test_y1 for the selected samples
            plt.subplot(1, 2, 1)
            for idx, i in enumerate(sample_indices):
                plt.plot(concat_test_y1df[i], color=colors[idx], label=f'Sample {i} True')
                plt.plot(pred_y1[i], linestyle='dashed', color=colors[idx], label=f'Sample {i} Pred')
            plt.xlabel('Pixel')
            plt.ylabel('Soot volume fraction')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title('Predicted/Actual Volume fraction')
            plt.legend()

            # Plot pred_y2 against test_y2 for the selected samples
            plt.subplot(1, 2, 2)
            for idx, i in enumerate(sample_indices):
                plt.plot(concat_test_y2df[i], color=colors[idx], label=f'Sample {i} True')
                plt.plot(pred_y2[i], linestyle='dashed', color=colors[idx], label=f'Sample {i} Pred')
            plt.xlabel('Pixel')
            plt.ylabel('Temperature')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.title('Predicted/Actual Temperature')
            plt.legend()

            plt.tight_layout()
            plt.show()
            
            #__________________________________________________________________________________#
            #__________________________________Evaluation______________________________________#
            #__________________________________________________________________________________#
            
            # Calculate overall MSE, RMSE, R2 for pred_y1
            mse_y1 = mean_squared_error(concat_test_y1df.flatten(), pred_y1.flatten())
            rmse_y1 = np.sqrt(mse_y1)
            r2_y1 = r2_score(concat_test_y1df.flatten(), pred_y1.flatten())
            adj_r2_y1 = 1 - (1 - r2_y1) * (len(concat_test_y1df.flatten()) - 1) / (len(concat_test_y1df.flatten()) - concat_test_y1df.shape[1] - 1)
            
            # Calculate overall MSE, RMSE, R2 for pred_y2
            mse_y2 = mean_squared_error(concat_test_y2df.flatten(), pred_y2.flatten())
            rmse_y2 = np.sqrt(mse_y2)
            r2_y2 = r2_score(concat_test_y2df.flatten(), pred_y2.flatten())
            adj_r2_y2 = 1 - (1 - r2_y2) * (len(concat_test_y2df.flatten()) - 1) / (len(concat_test_y2df.flatten()) - concat_test_y2df.shape[1] - 1)
            
            metrics_df = metrics_df.append({'Model': str(act) + ' outputs activation with ' + str(opt) + ' optimizer and ' + str(batch) + ' batches', 'MSE_y1': str(round(mse_y1, 3)), 'RMSE_y1': str(round(rmse_y1, 3)), 'R2_y1': str(round(r2_y1, 3)), 'Adj_R2_y1': str(round(adj_r2_y1, 3)), 'MSE_y2': str(round(mse_y2, 3)), 'RMSE_y2': str(round(rmse_y2, 3)), 'R2_y2': str(round(r2_y2, 3)), 'Adj_R2_y2': str(round(adj_r2_y1, 3))}, ignore_index=True)

metrics_df.to_csv(metrics_path, index=False)
