#@@@ Dependencies

#++ OS tested: Mac OS and Windows

#++ Packages/Modules dependencies are listed in the scripts.

#++ Typical install time: <30 minutes

#@@@ Scripts

#+++ KLEAR_development.py		
Python script for training/testing KLEAR(XGBoost+DNN) models for ABMR and TCMR prediction

#+++ KLEAR_evaluation.py		
Python script for additional evaluation of model performance

#+++ KLEAR_run.py
Python script to generate ABMR and TCMR prediction

#@@@ Sample datasets

#+++ KLEAR_sample_dataset.csv
Sample input for KLEAR

#+++ KLEAR_ABMR_sample_output.csv
Sample output from KLEAR for ABMR prediction

#+++ KLEAR_TCMR_sample_output.csv
Sample output from KLEAR for TCMR prediction

#+++ KLEAR_ABMR_test_model_performance
Key model evaluation output for ABMR prediction

#+++ KLEAR_TCMR_test_model_performance
Key model evaluation output for TCMR predition

#@@@ Demo/Usage instructions

#+++ Expected output: risk scores (probability) ranging from 0 (low risk) to 1 (high risk)

#+++ Expected runtime: <30 seconds

#+++ Development of KLEAR models

### ABMR prediction
python KLEAR_development.py --outcome_type outcome_abmr --train_set ./train.csv --test_set ./test_data.csv
#output directory: ./outcome_abmr

### TCMR prediction 
python KLEAR_development.py --outcome_type outcome_tcmr --train_set ./train.csv --test_set ./test_data.csv
#output directory: ./outcome_tcmr

#+++ Additional evaluation of KLEAR models

### ABMR prediction
python KLEAR_evaluation.py --outcome_col outcome_abmr --model_dir ./model_directory --test_paths ./test_data.csv
#output directory: ./outcome_abmr

### TCMR prediction
python KLEAR_evaluation.py --outcome_col outcome_tcmr --model_dir ./model_directory --test_paths ./test_data.csv 
#output directory: ./outcome_tcmr

#+++ Prediction using KLEAR

### ABMR prediction
python KLEAR_run.py --outcome_col outcome_abmr --model_dir ./model_directory --test_paths ./sample_data.csv
#output directory: ./outcome_abmr

### TCMR prediction 
python KLEAR_run.py --outcome_col outcome_tcmr --model_dir ./model_directory --test_paths ./sample_data.csv 
#output directory: ./outcome_tcmr
