
import torch
from model import colaModel
from dataset import Dataset

class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = colaModel.load_from_checkpoint(self.model_path)
        self.model.eval()
        self.model.freeze()
        self.process = Dataset()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]
    
    # predict sentece labels as acceptable or non acceptable
    def predict(self, sentence):
        inference_example = {"sentence": sentence}
        processed = self.process.tokenize(inference_example)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        preditions = []
        for score, label in zip(scores, self.labels):
            preditions.append({'labels': label, 'scores': score})
        return preditions

# to test trained model..
if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog."
    checkpoint_path = ""
    predictor = ColaPredictor(checkpoint_path)
    print(predictor.predict(sentence))