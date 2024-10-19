# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 08:57:52 2023

@author: elusi
"""

import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Conv1D, MaxPooling1D, Flatten, Reshape
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Defining paths
database = pd.ExcelFile(r'C:/Users/elusi/Downloads/proj/dataset/dataDifferentHeights2_121024.xlsx')
metrics_path = r'C:/Users/elusi/Downloads/proj/metrics.csv'
model_path = r'C:/Users/elusi/Downloads/proj/model/CNN/trained_model'
scaler1_path = r'C:/Users/elusi/Downloads/proj/model/CNN/scaler_y1.pkl'
scaler2_path = r'C:/Users/elusi/Downloads/proj/model/CNN/scaler_y2.pkl'
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

# Defining hyperparameters cross-validations for model fine tuning
#callbacks = [earlystop, reducelearning]
#batch_sizes = [1, 32, 10, 5]
#activations = ['linear', LeakyReLU(alpha=0.01), LeakyReLU(alpha=0.1), 'elu', 'tanh']
#optimizers = ['adagrad', 'adam', 'nadam', 'rmsprop']

# Hyperparameters that were found optimal
callbacks = [reducelearning]
batch_sizes = [5]
activations = [LeakyReLU(alpha=0.1)]
optimizers = ['rmsprop']

#__________________________________________________________________________________#
#______________________Data preprocessing and exploration__________________________#
#__________________________________________________________________________________#

# Creating 3d dataframe (10000, 66, 5) from the .xlsx database containing 10,000 worksheets containing (66,5) data each
sheets = database.sheet_names
dataframes = {}
for s in sheets:
    dataframes[s] = database.parse(s)
        
# Randomly pick 5 dataframes and plot their f/T & RGB for data exploration
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
            
            # Convolutional layers
            x = Conv1D(filters=8, kernel_size=5, activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=512, kernel_size=5, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)  
            x = Flatten()(x)  # Flatten the output of the last convolutional layer
            
            # Fully connected layers
            x = Dense(512, activation='relu')(x)
            x = Dense(1024, activation='relu')(x)

            # Output layers
            output1 = Dense(concat_xdf.shape[1], activation=act, name=output1_name)(x)
            output1 = Reshape((concat_xdf.shape[1], 1), name='reshaped_vol')(output1)
            output2 = Dense(concat_xdf.shape[1], activation=act, name=output2_name)(x)
            output2 = Reshape((concat_xdf.shape[1], 1), name='reshaped_temp')(output2)
            
            model = Model(inputs=inputs, outputs=[output1, output2])
            
            model.compile(optimizer=opt, loss={'reshaped_vol': 'mse', 'reshaped_temp': 'mse'}, metrics={'reshaped_vol': ['mse'], 'reshaped_temp': ['mse']})
            
            # Train the model
            history = model.fit(
                concat_xdf,
                {'reshaped_vol': scaled_y1, 'reshaped_temp': scaled_y2},
                epochs=100,
                batch_size=batch,
                callbacks=callbacks,
                validation_split=0.2,
                verbose=1
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

            # Plot Volume fraction training & validation loss values
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['reshaped_vol_mse'])
            plt.plot(history.history['val_reshaped_vol_mse'])
            plt.title('Volume fraction loss for ' + str(batch) + ' batches')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            print("Average of history.history['reshaped_vol_loss']: " + str(np.mean(history.history['reshaped_vol_mse'])))
            print("Average of history.history['val_reshaped_vol_loss']: " + str(np.mean(history.history['val_reshaped_vol_mse'])))
            
            # Plot Temperature training & validation loss values
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['reshaped_temp_mse'])
            plt.plot(history.history['val_reshaped_temp_mse'])
            plt.title('Temperature loss for ' + str(batch) + ' batches')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
            print("Average of history.history['reshaped_temp_loss']: " + str(np.mean(history.history['reshaped_temp_mse'])))
            print("Average of history.history['val_reshaped_temp_loss']: " + str(np.mean(history.history['val_reshaped_temp_mse'])))
            
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
            
            # Reshaping predictions to 2d arrays for inverse transform
            n_samples_test = concat_test_xdf.shape[0]
            pred_y1_2d = prediction[0].reshape(n_samples_test * n_timesteps, n_features)
            pred_y2_2d = prediction[1].reshape(n_samples_test * n_timesteps, n_features)
            
            # Inverse transform the predictions and reshape back to 3d (2000,66,1)
            pred_y1 = scaler_y1.inverse_transform(pred_y1_2d).reshape(n_samples_test, n_timesteps, 1)
            pred_y2 = scaler_y2.inverse_transform(pred_y2_2d).reshape(n_samples_test, n_timesteps, 1)
            
            # Plot few samples pred vs actual
            # Each plot takes 5 random samples from 500 different ranges of the data, i.e. between 1-500, 501-1000, 1001-1500
            # Define a color palette (adjust colors as needed)
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
            
            # Calculate MSE, RMSE, R2 for pred_y1
            mse_y1 = mean_squared_error(concat_test_y1df.flatten(), pred_y1.flatten())
            rmse_y1 = np.sqrt(mse_y1)
            r2_y1 = r2_score(concat_test_y1df.flatten(), pred_y1.flatten())
            adj_r2_y1 = 1 - (1 - r2_y1) * (len(concat_test_y1df.flatten()) - 1) / (len(concat_test_y1df.flatten()) - concat_test_y1df.shape[1] - 1)
            
            # Calculate MSE, RMSE, R2 for pred_y2
            mse_y2 = mean_squared_error(concat_test_y2df.flatten(), pred_y2.flatten())
            rmse_y2 = np.sqrt(mse_y2)
            r2_y2 = r2_score(concat_test_y2df.flatten(), pred_y2.flatten())
            adj_r2_y2 = 1 - (1 - r2_y2) * (len(concat_test_y2df.flatten()) - 1) / (len(concat_test_y2df.flatten()) - concat_test_y2df.shape[1] - 1)
            
            # Append results in metrics dataframe
            metrics_df = metrics_df.append({'Model': str(act) + ' outputs activation with ' + str(opt) + ' optimizer and ' + str(batch) + ' batches', 'MSE_y1': str(round(mse_y1, 3)), 'RMSE_y1': str(round(rmse_y1, 3)), 'R2_y1': str(round(r2_y1, 3)), 'Adj_R2_y1': str(round(adj_r2_y1, 3)), 'MSE_y2': str(round(mse_y2, 3)), 'RMSE_y2': str(round(rmse_y2, 3)), 'R2_y2': str(round(r2_y2, 3)), 'Adj_R2_y2': str(round(adj_r2_y1, 3))}, ignore_index=True)

# Saving metrics dataframe
metrics_df.to_csv(metrics_path, index=False)

# Saving the trained model
model.save(model_path + '.keras')

# Saving the scalers
joblib.dump(scaler_y1, scaler1_path)
joblib.dump(scaler_y2, scaler2_path)