# MFFN
This repository provides the implemention for the paper Multi-scale Feature Fusion Network for the Prediction of Protein-Protein Binding Affinity Changes upon Mutations.

Please cite our paper if our datasets or code are helpful to you ~ 😊

## Requirements
* Python 3.9
* Pytorch 1.12.1
* Transformers 4.21.3
* sentencepiece 0.2.0
* numpy 1.23.1
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

## Cite
Please cite our paper if our datasets or code are helpful to you.

H. Zhang, Y. Liu, L. Yu, Z. Wang, Y. Liu and M. Guo, "Multi-Scale Feature Fusion Network for the Prediction of Protein-Protein Binding Affinity Changes upon Mutations," 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Wuhan, China, 2025, pp. 218-223, doi: 10.1109/BIBM66473.2025.11356932.

```
@INPROCEEDINGS{zhang2025mffn,
  author={Zhang, Hao and Liu, Yang and Yu, Limin and Wang, Zejie and Liu, Yifei and Guo, Maozu},
  booktitle={2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={Multi-Scale Feature Fusion Network for the Prediction of Protein-Protein Binding Affinity Changes upon Mutations}, 
  year={2025},
  volume={},
  number={},
  pages={218-223},
  doi={10.1109/BIBM66473.2025.11356932}}
```
