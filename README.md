# soot
Machine Learning-based Approximation of Soot Temperature and Volume Fraction Profiles in Combustion Diagnostics

The scripts and output metrics summary are found in the main directory.

soot_proj/
- datasets/
- model/
- testcases/
- mat_to_excel.py
- metrics.csv
- metrics_testcases.csv
- soot_FNN_train.py
- soot_CNN_train.py
- soot_testcase_predict.py
		
Datasets are saved under soot_proj/datasets/ directory and it is references to in the train .py codes.

After training a model, the models .keras and scalers .pkl files are saved in soot_proj/model/ directory for later use for prediction. They are referenced to in soot_testcase_predict.py.

For data conversion from .mat to .xlsx use mat_to_excel.py code, with the data files locations referenced in the code.
The matlab data title might need to be adjusted based on how it was named on creation, this is referenced in row 13.
