
import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

from server.demo_model import DemoModel
from server.lru_cache import LRUCache


DISTILBERT_MODEL = 'distilbert-base-uncased'
DISTILMBERT_MODEL = 'distilbert-base-multilingual-cased'


class DistilBertPredictor():
    def __init__(self,
                 model_name: str = DISTILMBERT_MODEL) -> None:
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForMaskedLM.from_pretrained(model_name)

    def predict_json(self, inputs: dict) -> dict:
        query = inputs["sentence"] + ' [MASK].'
        topk = inputs.get("topk", 5)

        logits, mask_token_ind = self._predict(query)
        best_logits, best_indices = logits.topk(topk)
        best_words = [[' ' + self.tokenizer.decode([idx.item()])
                       for idx in best_indices[i]]
                      for i in range(len(best_indices))]
        print(best_words)

        probs = torch.nn.functional.softmax(logits[mask_token_ind])
        best_probs = probs[best_indices[mask_token_ind]].tolist()

        return {
            "logits": best_logits[mask_token_ind].tolist(),
            "probabilities": best_probs,
            "words": [best_words[mask_token_ind]]
        }

    def _predict(self, query: str) -> torch.Tensor:
        input_ids = torch.tensor(
            self.tokenizer.encode(query, add_special_tokens=True)).unsqueeze(0)
        print([self.tokenizer.decode(i) for i in input_ids[0].tolist()])
        mask_token_ind = input_ids[0].tolist().index(103)
        with torch.no_grad():
            output = self.model(input_ids)
        logits = output[0][0]
        return logits, mask_token_ind

    def __getitem__(self, index: int) -> str:
        return self.tokenizer.decode([index])


class DistilBertDemoModel(DemoModel):
    def predictor(self):
        return DistilBertPredictor()
