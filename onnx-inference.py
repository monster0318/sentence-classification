"""
---------------------------------------------------
Run ONNX inference runtime on some sample sentences
for checking models onnx runtime and performance.
---------------------------------------------------

Author: Muhammad Faizan

python onnx-inference.py
-------------------------
"""

# add dependencies
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
import torch

from dataset import Dataset
from utils import timing


# ONNX predictor definition
class ONNXPredictor:
    def __init__(self, model_path) -> None:
        """
        ONNX predictor 
        -------------
        Parameters
        ----------
        model_path: str
        """
        self.ort_session = ort.InferenceSession(model_path, providers= ["AzureExecutionProvider", "CPUExecutionProvider"])
        self.processor = Dataset()
        self.labels = ["unacceptable", "acceptable"]

    # predict
    @timing
    def predict(self, text):
        """
        predict the text as acceptable or unacceptable
        ----------------------------------------------

        Parameters
        ----------
        text: str

        """
        inference_example = {'sentence': text}
        processed = self.processor.tokenize(inference_example)
        ort_input = {'input_ids': np.expand_dims(processed['input_ids'], axis=0).astype(np.int64), 
                     'attention_mask': np.expand_dims(processed['attention_mask'], axis= 0).astype(np.int64)}
        # run the ort inference
        ort_outputs = self.ort_session.run(None, ort_input)
        scores = softmax(ort_outputs[0])[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == '__main__':
    # single sentence
    sentence = 'He eating is apple' # WARNING: grammatically uncorrect but model is making it correct.
    predictor = ONNXPredictor('models/model.onnx')
    print(predictor.predict(sentence))

    # for a list of sentences
    sentences = ['Mission impossible is my favourite movie'] * 3
    for sentence in sentences:
        print(predictor.predict(sentence))
    

