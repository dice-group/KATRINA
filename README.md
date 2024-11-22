
# Joint Query Generation
## installation
dowload the repository

run pip install -r requirements
## training Query Generation models
run the script train.py in the package Generator Model

check parameters in the python file parameters.py for joint and single model training.

for joint training the files for both traing datasets should be in the same folder.
## Entitiy Linking
- Train LM-model: run the script train.py in the folder Entity span prediction.
hyperparameters are defined in the script parameters.py
- predicting Entities on a benchmark: Run the script predictForFile.py the format of the input file should be in LC-QuAD json format
## run experiments
-use the script result_script.py to generate Queries for a QALD-JSON formated input file
There are multiple options, that can be configured in the hyperparameters.
Espacially, if Gold Entities should be used, or on which KB results should be predicted. The output is a file in QAD-format
