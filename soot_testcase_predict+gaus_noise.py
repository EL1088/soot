# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 08:57:52 2023

@author: elusi
"""

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
import tensorflow as tf

# Defining paths
model_path = r'C:/Users/elusi/OneDrive/Desktop/soot_proj/model/CNN_dataset2differentHeights/trained_model.keras'
scaler1_path = r'C:/Users/elusi/OneDrive/Desktop/soot_proj/model/CNN_dataset2differentHeights/scaler_y1.pkl'
scaler2_path = r'C:/Users/elusi/OneDrive/Desktop/soot_proj/model/CNN_dataset2differentHeights/scaler_y2.pkl'
testcase_dir = 'C:/Users/elusi/OneDrive/Desktop/soot_proj/testcases/dataset2/'
testcase_metrics_path = r'C:/Users/elusi/OneDrive/Desktop/soot_proj/metrics_testcases.csv'
testcase_metrics_df = pd.read_csv(testcase_metrics_path)

# Load the pre trained model
custom_objects = {'LeakyReLU': LeakyReLU}
trained_model = load_model(model_path, custom_objects=custom_objects)

# Model representation
print(trained_model.summary())
tf.keras.utils.plot_model(
    trained_model,
    to_file=testcase_dir + 'model_plot.png',
    show_shapes=True,
    rankdir='LR',
    expand_nested=True,
    show_layer_activations=True,
    dpi=100
)

# Load the scalers
scaler_y1 = joblib.load(scaler1_path)
scaler_y2 = joblib.load(scaler2_path)

# Function for creating gaussian noise in RGB data
def add_gaussian_noise(rgb_matrix, std_dev):
    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, rgb_matrix.shape)

    # Add noise to the original matrix
    noisy_rgb_matrix = rgb_matrix + noise

    # Clip the values to stay within the valid RGB range [0, 5000]
    noisy_rgb_matrix = np.clip(noisy_rgb_matrix, 0, 5000)

    return noisy_rgb_matrix

testcases = []
# Iterate over files in the directory
for filename in os.listdir(testcase_dir):
    if filename.endswith(".csv"):
        # Get the testcase name
        test = os.path.splitext(filename)[0]
        
        # Read the CSV file into a DataFrame
        file_path = os.path.join(testcase_dir, filename)
        df = pd.read_csv(file_path)
        testcases.append((test, df))

# Iterate over and predict testcases
xdrops = [0, 1]
y1drops = [1, 2, 3, 4]
y2drops = [0, 2, 3, 4]
columns_for_noise = [2, 3, 4]

noise_magnitudes = [0.0025, 0.005, 0.01, 0.015, 0.02]
noisy_matrices = []
matrices = []

# Create noisy versions of the testcases and append all version into 'matrices' list
for test, testcase in testcases:
    if 'noisy' not in test:
        testcase_x = testcase.iloc[:, columns_for_noise].to_numpy()
        # Store the original matrix with metadata
        matrices.append({'matrix': testcase.copy(), 'test': test, 'noise': None})  # Original matrix
        
        # Determine max intensity for the current sample
        max_intensity = np.max(testcase_x, axis=0)
        
        for noise in noise_magnitudes:
            # Calculate absolute standard deviation based on the mean
            std = max_intensity * noise
            
            # Add Gaussian noise
            noisy_matrix = add_gaussian_noise(testcase_x.copy(), std)
            noisy_matrix = np.clip(noisy_matrix, 0, np.max(testcase_x))  # Clamp to the max of the original matrix
            
            # Create a noisy version of the original matrix
            full_noisy_matrix = testcase.copy()
            full_noisy_matrix.iloc[:, columns_for_noise] = noisy_matrix  # Replace only the relevant columns
            
            # Store the noisy matrix with metadata
            matrices.append({'matrix': full_noisy_matrix, 'test': test, 'noise': noise})

# Plot testcases RGB
for matrix in matrices:
    mat = matrix['matrix']
    test = matrix['test']
    noise = matrix['noise']
    
    plt.figure(figsize=(12, 6))
    plt.plot(mat.iloc[:, 2], label='Red', color='red', linewidth=2)        # Second column
    plt.plot(mat.iloc[:, 3], label='Green', color='limegreen', linewidth=2)  # Third column
    plt.plot(mat.iloc[:, 4], label='Blue', color='royalblue', linewidth=2)   # Fourth column
    plt.xlabel('Pixel Index', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title(f'RGB Pixel Intensities of {test}, Noise: {noise}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Iterate through all each testcase in 'matrices' list containing original and noisy version
for matrix in matrices:
    testcase = matrix['matrix']
    test = matrix['test']
    noise = matrix['noise']
    
    # Check if the number of pixels in the testcase is larger than what the model was trained on
    if testcase.shape[0] > trained_model.input_shape[1]:
        # Trim all pixels larger than what the model was trained on
        testcase = testcase.iloc[:trained_model.input_shape[1], :]
        
    # Split data to X and Y1/Y2 and reshape to arrays
    testcase_x = testcase.drop(testcase.columns[xdrops], axis=1, inplace=False).to_numpy().reshape((-1, testcase.shape[0], 3))
    testcase_y1 = testcase.drop(testcase.columns[y1drops], axis=1, inplace=False).to_numpy().reshape((-1, testcase.shape[0], 1))
    testcase_y2 = testcase.drop(testcase.columns[y2drops], axis=1, inplace=False).to_numpy().reshape((-1, testcase.shape[0], 1))
    n_samples, n_timesteps, n_features = testcase_y1.shape
    pred_testcase = trained_model.predict(testcase_x)
    
    # Reshaping predictions to 2d arrays for inverse transform
    #n_samples_testcase = testcase_x.shape[0]
    n_samples, n_timesteps, n_features = testcase_y1.shape
    pred_testcase_y1_2d = pred_testcase[0].reshape(n_samples * n_timesteps, n_features)
    pred_testcase_y2_2d = pred_testcase[1].reshape(n_samples * n_timesteps, n_features)
    
    # Inverse transform the predictions and reshape back to 3d (2000,66,1)
    pred_testcase_y1_2d = scaler_y1.inverse_transform(pred_testcase_y1_2d).reshape(n_samples, n_timesteps, 1)
    pred_testcase_y2_2d = scaler_y2.inverse_transform(pred_testcase_y2_2d).reshape(n_samples, n_timesteps, 1)
    
    #__________________________________________________________________________________#
    #__________________________________Optional______________________________________#
    #__________________________________________________________________________________#

    # Removal of all pixels from where RGB values zero-out, usually around length of 60th pixel
    # mask = (testcase_x >= 50).any(axis=2)
    # filtered_indices = mask.flatten()
    # testcase_x = testcase_x[:, filtered_indices]
    # num_indices = testcase_x.shape[1]

    # testcase_y1 = testcase_y1[:, :num_indices, :]
    # testcase_y2 = testcase_y2[:, :num_indices, :]

    # pred_testcase_y1_2d = pred_testcase_y1_2d[:, :num_indices, :]
    # pred_testcase_y2_2d = pred_testcase_y2_2d[:, :num_indices, :]
    
    #__________________________________________________________________________________#
    #_____________________________End of optional part_________________________________#
    #__________________________________________________________________________________#
        
    # Plot pred vs actual
    plt.figure(figsize=(15, 5))
    # Plot pred_y1 against test_y1 for the selected samples
    plt.subplot(1, 2, 1)
    plt.plot(testcase_y1[0], label=f'Testcase {test} True')
    plt.plot(pred_testcase_y1_2d[0], linestyle='dashed', label=f'Testcase {test} Predicted')
    plt.xlabel('Pixel')
    plt.ylabel('Soot volume fraction')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'Predicted/Actual vol. fraction for {test}, Noise: {noise}')
    # Plot pred_y2 against test_y2 for the selected samples
    plt.subplot(1, 2, 2)
    plt.plot(testcase_y2[0], label=f'Testcase {test} True')
    plt.plot(pred_testcase_y2_2d[0], linestyle='dashed', label=f'Testcase {test} Predicted')
    plt.xlabel('Pixel')
    plt.ylabel('Temperature')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f'Predicted/Actual Temp. for {test}, Noise: {noise}')
    plt.tight_layout()
    plt.show()

    #__________________________________________________________________________________#
    #__________________________________Evaluation______________________________________#
    #__________________________________________________________________________________#
        
    # Calculate overall MSE, RMSE, R2 for pred_y1
    testcase_mse_y1 = mean_squared_error(testcase_y1.flatten(), pred_testcase_y1_2d.flatten())
    testcase_rmse_y1 = np.sqrt(testcase_mse_y1)
    testcase_r2_y1 = r2_score(testcase_y1.flatten(), pred_testcase_y1_2d.flatten())
    testcase_adj_r2_y1 = 1 - (1 - testcase_r2_y1) * (len(testcase_y1.flatten()) - 1) / (len(testcase_y1.flatten()) - testcase_y1.shape[1] - 1)
    
    # Calculate overall MSE, RMSE, R2 for pred_y2
    testcase_mse_y2 = mean_squared_error(pred_testcase_y2_2d.flatten(), testcase_y2.flatten())
    testcase_rmse_y2 = np.sqrt(testcase_mse_y2)
    testcase_r2_y2 = r2_score(pred_testcase_y2_2d.flatten(), testcase_y2.flatten())
    testcase_adj_r2_y2 = 1 - (1 - testcase_r2_y2) * (len(pred_testcase_y2_2d.flatten()) - 1) / (len(pred_testcase_y2_2d.flatten()) - pred_testcase_y2_2d.shape[1] - 1)
    
    # Append results in metrics dataframe
    testcase_metrics_df = testcase_metrics_df.append({'Model' : f'Testcase {test}', 'MSE_y1': str(round(testcase_mse_y1, 3)), 'RMSE_y1': str(round(testcase_rmse_y1, 3)), 'R2_y1': str(round(testcase_r2_y1, 3)), 'Adj_R2_y1': str(round(testcase_adj_r2_y1, 3)), 'MSE_y2': str(round(testcase_mse_y2, 3)), 'RMSE_y2': str(round(testcase_rmse_y2, 3)), 'R2_y2': str(round(testcase_r2_y2, 3)), 'Adj_R2_y2': str(round(testcase_adj_r2_y1, 3))}, ignore_index=True)

# Saving metrics dataframe
testcase_metrics_df.to_csv(testcase_metrics_path, index=False)