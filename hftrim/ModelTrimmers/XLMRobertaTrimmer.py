import torch
from .RobertaTrimmer import RobertaTrimmer

class XLMRobertaTrimmer(RobertaTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='XLMRobertaModel':
            from transformers import XLMRobertaModel
            model = XLMRobertaModel(self.config)
            changed_params = [
                'embeddings.word_embeddings.weight'
            ]
        elif arch=='XLMRobertaForSequenceClassification':
            from transformers import XLMRobertaForSequenceClassification
            model = XLMRobertaForSequenceClassification(self.config)
            changed_params = [
                'roberta.embeddings.word_embeddings.weight'
            ]
        else:
            raise NotImplementedError("ERROR: XLMRobertaTrimmer does not support this architecture!")

        self.trimmed_model = model
        self.changed_params = changed_params