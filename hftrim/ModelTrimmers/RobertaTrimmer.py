import torch
from .BaseTrimmer import BaseTrimmer

class RobertaTrimmer(BaseTrimmer):
    def __init__(self, model, config, tokenizer):
        super().__init__(model, config, tokenizer)

    def trim_weights(self):
        # embedding matrix
        if 'embeddings.word_embeddings.weight' in self.model.state_dict():
            em = self.model.embeddings.word_embeddings.weight.detach().numpy()
            self.trimmed_weights['embeds'] = em[self.trimmed_vocab_ids, :]
        if 'roberta.embeddings.word_embeddings.weight' in self.model.state_dict():
            em = self.model.roberta.embeddings.word_embeddings.weight.detach().numpy()
            self.trimmed_weights['embeds'] = em[self.trimmed_vocab_ids, :]

    def initialize_new_model(self):
        # arch = self.config.architectures[0]
        arch = self.model.__class__.__name__
        if arch=='RobertaModel':
            from transformers import RobertaModel
            model = RobertaModel(self.config)
            changed_params = [
                'embeddings.word_embeddings.weight'
            ]
        elif arch=='RobertaForSequenceClassification':
            from transformers import RobertaForSequenceClassification
            model = RobertaForSequenceClassification(self.config)
            changed_params = [
                'roberta.embeddings.word_embeddings.weight'
            ]
        else:
            raise NotImplementedError("ERROR: RobertaTrimmer does not support this architecture!")

        self.trimmed_model = model
        self.changed_params = changed_params

    def trim_model(self):
        # copy unchanged params over from the old model
        for param in self.model.state_dict().keys():
            if param in self.changed_params:
                continue
            self.trimmed_model.state_dict()[param].copy_(self.model.state_dict()[param])

        prunedEmbeddingMatrix = torch.nn.Embedding.from_pretrained(torch.Tensor(self.trimmed_weights['embeds']), 
                                                                   freeze=False, 
                                                                   padding_idx=self.tokenizer.pad_token_id)
        self.trimmed_model.set_input_embeddings(prunedEmbeddingMatrix)

        # tie weights as described in model config
        self.trimmed_model.tie_weights()