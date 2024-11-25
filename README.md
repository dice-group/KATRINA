
# UniQ-Gen
This repository contains the code for the paper UniQ-Gen: Unified Query Generation across Multiple Knowledge Graphs.
The goal is to train unified query generation models across knowledge graphs. 
To this end, this repository contains code for training query generation models based on LLMs in various setups and with different input from entity linking systems.
## Installation
- Dowload the repository
- Run pip install -r requirements
## Files
All required files including trained models can be downloaded from our FTP server: https://files.dice-research.org/projects/UniQ-Gen/EKAW/
## Training Query Generation models
Run the script train.py in the package Generator Model
check parameters in the python file parameters.py for joint and single model training, 
for editing paths to the datasets and other configurations

for joint training the files for both traing datasets should be in the same folder.
## Entity Linking
### Wikidata
- Train LM-model: run the script train.py in the folder Entity span prediction.
hyperparameters are defined in the script parameters.py
- we also rely on the GENRE framework https://github.com/facebookresearch/GENRE
- please also download the kilt titles trie and the huggingface model as described in the GENRE repository
- the code for GENRE is already included here https://github.com/dice-group/KATRINA/tree/main/flair_el/genre 
- predicting Entities on a benchmark: Run the script predictForFile.py the format of the input file should be in LC-QuAD json format
### Freebase/GrailQA
- **Entity Linking**: we rely on the entity linking framework of RNG-KBQA https://github.com/salesforce/rng-kbqa for installation please follow their setup instructions
- If you only want to run the experiments, they also provide files with linked entities for the Grail QA dataset: https://github.com/salesforce/rng-kbqa/blob/main/GrailQA/entity_linking/grailqa_el.json
- **Type and Relation Linking** We rely on the TIARA framework for this step plese follow the instructions in this repository: https://github.com/microsoft/KC/tree/main/papers/TIARA
- similar as RNG-KBQA, they also provide their results for schema linking as download, in case you only want to run the experiments
## Experiments
- **Script for experiments** use the script result_script.py to generate Queries for a QALD-JSON formated input file
There are multiple options like the target KG, the benchmarking files, if Gold Entities should be used, or on which KB results should be predicted that can be configured in the parameters. The input is a file in QALD format like https://github.com/KGQA/QALD-10 (without the queries and the answers).
The output is a file in QALD-format as well that can be uploaded into the GERBIL-QA framework https://gerbil-qa.aksw.org/gerbil/
- **Trained Models**: Our pretrained models can be found here: https://files.dice-research.org/projects/UniQ-Gen/EKAW/models/
