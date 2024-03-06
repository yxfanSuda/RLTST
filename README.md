
# TSTRL

## Data and Pretrained Model Downloading
### Molweni
[https://aclanthology.org/2020.coling-main.238/](https://aclanthology.org/2020.coling-main.238/) 

Put the downloaded data in the ./Data/Molweni directory.

### STAC
[https://aclanthology.org/L16-1432/](https://aclanthology.org/L16-1432/) 

Put the downloaded data in the ./Data/STAC directory.

### Hu
[https://www.ijcai.org/Proceedings/2019/0696.pdf](https://www.ijcai.org/Proceedings/2019/0696.pdf) 

Put the downloaded data in the ./Data/Hu_Dataset directory.

### Ou5/Ou10/Ou15
[https://aclanthology.org/D16-1231.pdf](https://aclanthology.org/D16-1231.pdf)

Put the downloaded data in the ./Data/Ou_Dataset directory.

### bert-base-uncased
[https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)
Put the downloaded model in the ./pretrained_model directory.

## Environment
```
conda create -n RLTST python==3.8
pip install -r requirements.txt
conda activate RLTST
```

## Data Processing
```
cd Data/Ou_Dataset
python ou_process.py
cd ..
python convert_ARlabel2_Link.py
```

# Data Selection
```
sh select_data.sh
```


# TST Learning
```
sh TST.sh
```

# Test
```
sh test.sh
```
