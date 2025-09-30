# MFFN
This repository provides the implemention for the paper Multi-scale Feature Fusion Network for the Prediction of Protein-Protein Binding Affinity Changes upon Mutations.

Please cite our paper if our datasets or code are helpful to you ~ 😊

## Requirements
* Python 3.9
* Pytorch 1.12.1
* Transformers 4.21.3
* sentencepiece 0.2.0
* openpyxl
* joblib

## Dataset
* [SKEMPI] Given in the dataset folder.
* [MPAD]  (http://compbio.clemson.edu/SAAMBE-MEM & https://web.iitm.ac.in/bioinfo2/mpad)

Processing the dataset:
```bash
cd model
python process_data.py
```

## Protein LLM Settings
* Download [pytorch_model.bin](https://drive.google.com/file/d/1ZXpWZELAmTC9IfqpMYQ16BUUUfIanYpv/view?usp=drive_link) and unzip the file in folder ./model/protein_llm/.

## Training & Evaluation for Ten-Fold-Cross-Validation
```bash
python main.py
```

## Acknowledgements
MFFEN builds upon the source code from the project.

We thank their contributors and maintainers!
