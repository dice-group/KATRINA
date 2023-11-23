from transformers import Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import torch

class GenerationTrainer(Trainer):

    # def compute_loss(self, model, inputs):
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs, use_cache=False)
    #     logits = outputs[0]
    #     return self._compute_loss(logits, labels, ignore_index=model.config.pad_token_id)

    # def _compute_loss(self, logits, labels, ignore_index):
    #     if self.args.label_smoothing == 0:
    #         # Same behavior as modeling_bart.py
    #         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    #         assert logits.shape[-1] == self.model.config.vocab_size
    #         loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    #     else:
    #         lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    #         loss, nll_loss = label_smoothed_nll_loss(
    #             lprobs, labels, self.args.label_smoothing, ignore_index=ignore_index
    #         )
    #     return loss
 


    def _pad_tensors_to_max_len(self, tensor, max_length, pad_token_id):
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=max(self.args.warmup_steps, self.args.warmup_ratio * num_training_steps),
                num_training_steps=num_training_steps
            )