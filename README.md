# TRIM-AI
Official repository for AAAI 2023 paper "A Continual Pre-training Approach to Tele-Triaging Pregnant Women in Kenya."  In this paper, we develop an NLP framework, TRIM-AI, which can automatically predict the emergency level (or severity of medical condition) of a pregnant mother based on the content of their code-mixed SMS messages.

The full version of the paper is available at https://ojs.aaai.org/index.php/AAAI/article/view/26709

This project is licensed under the terms of the MIT license.

## Data 

Due to the confidentiality agreement，we cannot make the dataset public available. Instead, we will provide the a few lines of fake data to help you understand the stucture of the dataset (for example, the columns and the corresponding column names). 

## Files

- `./annotated_dataset.csv` - Fake labeled dataset with same stucture of original dataset.

- `./dataset_continual_pretraining.txt` - Fake unlabeled dataset for continual pre-taining (each line is a single code-mixed message).

- `./model_file/train_continual_model.py` - Python code to do model continual pretraining with unlabeled dataset

- `./model_file/train_model.py` -Python code to do model fine tuning

- `./model_file/eval_model.py` - Python code for making prediction on new test set

- `./model_file/pred_single.py` - Python code for making prediction on new **single messages**

  

## Requirements

- Python: 3.7
- pytorch: 1.8.0 (GPU version)
- transformers: 4.12.5
- numpy: 1.21.2
- pandas: 1.3.4
- scikit-learn: 1.0.1
- sentencepiece: 0.1.99



## Train a new model

1. Make sure GPU is available under the environment

2. Do continual pre-training on open-sourced pre-trained language model  and "dataset_continual_pretraining.txt" (can skip this step if you don't need continual pre-pertaining)

   ```python
   python ./model_file/train_continual_model.py --path_project_folder [actual project folder path]
   ```

   

3. Train the new model (with step-2) based on labeled set "annotated_dataset.csv" 

```python
python ./model_file/train_model.py --path_project_folder [actual project folder path] --pretrained_model_name [folder path for model files after continual pre-training]
```

​        Train the new model (without step-2) based on labeled set "annotated_dataset.csv" 

```
python ./model_file/train_model.py --path_project_folder [actual project folder path]
```



## Predict labels on the new dataset (CSV file)

1. Make sure GPU is available under the environment

2. Predict the new data based on existing test set "split_test_data.csv". This test data will automatically generate  from the labeled dataset during the fine tuning step.

**Note: if you want to predict label on other new dataset, the new dataset should have the same format as "split_test_data.csv"**

```python
# if predict label for existing test set (split_test_data.csv)
python ./model_file/eval_model.py --path_project_folder [actual project folder path]
```



## Predict labels on a new single sentence

1. Make sure GPU is available under the environment

```python
# if predict label for a new single sentence
python ./model_file/pred_single.py --path_project_folder [actual project folder path] --new_sentence [new sentence]
```

2. A simple example.

The path for project folder is: "/home/TRIM-AI/"

the sentence I want to predict is:"hello, how are you?"

```python
#the command for make prediction based on single sentence is :
python ./model_file/pred_single.py --path_project_folder "/home/TRIM-AI/" --new_sentence "hello, how are you?"
```

