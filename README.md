## Ventilator Pressure Competition


[**Task List**](#quickstart-colab-in-the-cloud)
| [**Preapare data**](#prepare-data)
| [**Config file**](#config-file)
| [**Run**](#run)
| [**How to contribute**](#how-to-contribute)

**Our solution to kaggle competition "Ventilator Pressure Competition"**<br>
[Link](https://www.kaggle.com/c/ventilator-pressure-prediction/leaderboard) to the competition

Authors: 

|  Name   | Group  | Email | Github |
|  ----   | ----   | ----  | ---- |
| Xinze Li  | мНОД\_ИССА\_2020 | <sli_4@edu.hse.ru> | [li-xinze][xinze] |
| Katarina Kuchuk  | мНОД\_ИССА\_2020 | <kkuchuk@edu.hse.ru> | [Kale2601][katarina] |



### Task list

- [x] Framework
	- [x] PyTorch version (branch torch)[@Xinze][xinze]
	- [x] Tensorflow version (branch tf) [@Katarina][katarina]
- [ ] Models
	- [x] LSTMv1, Transformers, ... [@Xinze][xinze]
	- [x] LSTMv2, TCN, ... [@Katarina][katarina]
	- [ ] ... 
- [x] Grouping strategy 
- [x] Error Analysis tools [@Xinze][xinze], [@Katarina][katarina]
- [x] Final submission before the dealine (693/2605 in public leaderboard)
- [ ] Late submissions

### Prepare data
Download dataset from [here](https://www.kaggle.com/c/ventilator-pressure-prediction/data) <br>
unzip it and move files to folder `data`:

```
data/
├── train.csv
├── test.csv
└── sample_submission.csv (optional)
```

### Config file
Currently supported modes in pipeline are: <br/>

`train_valid`: hyperparameter tuning <br/>
`train`: merge train and val dataset to train a model<br/>
`pred`: pred on test.csv, get submission file <br/>

A template of config file: `config/config.yaml`


### Run

You can run it on local/remote server or Colab [![Open in Colab][Colab Badge]][main Notebook]

on Colab

```
1. upload this repo to your google drive
2. click [Open in Colab] and follow the instrutions in opened file
```

In console: 

```
python main.py -c path_to_config
```
In jupyter notebook: 

```
main.ipynb
```

### How to contribute

#### 1 Add a new model
- Add your model under the folder `project/model.py`

- Register your model in `MODEL_MAP` in `project/model.py`

```

MODEL_MAP = {
	'LSTMv1': LSTMv1,
	'your_model_name': YourModel
}
```

- Use your model by indicating your model's name in config file

```
model_config:
	model: your_model_name
```

#### 2 Add a new data processor
- Add your processor in `project/data_processor.py`
- Register your processor in `DATASET_PROCESSOR_MAP` in `project/data_processor.py`

```

DATASET_PROCESSOR_MAP = {
    'v1': processor_v1,
    'your_processor_name': your_processor
}
```
- Use your data processor by indicating its name in config file
 
```
data_config:
   processor: your_processor_name
```


[Colab Badge]:          https://colab.research.google.com/assets/colab-badge.svg
[main Notebook]:        https://colab.research.google.com/github/li-xinze/ventilator_pressure_prediction/blob/master/main.ipynb
[xinze]:                https://github.com/li-xinze
[katarina]:             https://github.com/Kale2601
