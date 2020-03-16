## Compile Edge TPU

```
python3 main.py --test tracker convert
cd models/tpu
edgetpu_compiler ohmnilabs_features_extractor_quant_postprocess.tflite
```