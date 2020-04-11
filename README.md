# Identity tracking

## Prepare dataset

Dataset: `https://drive.google.com/drive/folders/1hG0kJBNwGDXyJ1XtG4PK7zX4RarCQhrr?usp=sharing`

Download `train.zip` if you only care about he training process. In case you intent to edit the data, dowload `raw.zip`

## How to train?

```
python3 main.py --test tracker train
```

## Compile Edge TPU

```
python3 main.py --test tracker convert
cd models/tpu
edgetpu_compiler ohmnilabs_features_extractor_quant_postprocess.tflite
```