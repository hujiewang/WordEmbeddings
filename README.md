# Predicting next word given a context of previous words

## Model

```
input->LookupTable->[Dropout->Linear->Tanh]x2->SoftMaxTree
```

## Software
* CUDA
* Torch(dp,cunn,cunnx)
* graphicsmagick
* torchx

## Usage

### Dataset preparation
* Dataset will be automatically downloaded from (http://lisaweb.iro.umontreal.ca/transfert/lisa/users/leonardn/billionwords.tar.gz)

### Training

```
th start.lua
```

## Prediction
* Uncomment the following line in main

```
--predictSingle({"I","want","to","have","a"},model,billionwords,opt)
```


