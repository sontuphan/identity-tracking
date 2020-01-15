import os
import tflite_runtime.interpreter as tflite

from utils import detect

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
LABELS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../models/tpu/coco_labels.txt")
MODELS = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../models/tpu/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")


class HumanDetection:
    def __init__(self, confidence=0.6):
        self.labels = self.load_labels()
        self.interpreter = tflite.Interpreter(
            model_path=MODELS,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB)
            ])
        self.confidence = confidence

    def load_labels(self):
        with open(LABELS, 'r', encoding='utf-8') as labels_file:
            lines = labels_file.readlines()
            if not lines:
                return {}

            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}

    def predict(self, img):
        self.interpreter.allocate_tensors()
        detect.set_input(self.interpreter, img)
        self.interpreter.invoke()
        objs = detect.get_output(self.interpreter, self.confidence, True)
        return objs
