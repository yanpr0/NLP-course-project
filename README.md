# Grammatical acceptability evaluation using the material of the RuCoLA corpus

It is possible to open the entire project in Google.Colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19GYXlgw51oFyFU29HVoQzV7HqDz9rGCi?usp=sharing) or you can train and use the model locally.  

## Installation  
    pip install -r requirements.txt
    ./build_datasets.sh

## Usage
### 1. Pre-train grammeme model
    ./pretrain.py
### 2. Train main model using one of the checkpoints of pre-trained model in `data/model/`
    ./main.py --checkpoint-dir data/model/<SOME_CHECKPOINT_DIR>

