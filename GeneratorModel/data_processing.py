import json
import torch

from torch.utils.data import Dataset
from transformers import T5PreTrainedModel


class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)

class Dataprocessor():
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args=args

    def read_ds_to_list(self,path_to_ds):
        return []
    def process_training_ds(self,data):
        samples = self.read_ds_to_list(data)
        dataset=ListDataset([])
        for sample in samples:
            dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def process_sample(self,input,label=None):
        pass
class Dataprocessor_test(Dataprocessor):

    def process_sample(self,input,label=None):
        encoding = self.tokenizer(text=input,text_target=label, return_tensors="pt",
                                  )
        return encoding

class Dataprocessor_QALD(Dataprocessor):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data["questions"]:
            sample={}
            for lang in question["question"]:
                if lang["language"]=="en":
                    sample["input"]=lang["string"]
            sample["label"]=question["query"]["sparql"]
            samples.append(sample)
        return samples
    def process_sample(self,input,label=None):
        encoding = self.tokenizer.prepare_seq2seq_batch(src_texts=[input],text_target=[label], return_tensors="pt",
                                  max_length=self.args["max_target_length"],
                                  max_target_length=self.args["max_target_length"]

                                  )
        input=encoding.data["input_ids"]
        padded_input_tensor = self.tokenizer.pad_token_id * torch.ones(
            (input.shape[0], self.args["max_input_length"]), dtype=input.dtype, device=input.device
        )
        padded_input_tensor[:, : input.shape[-1]] = input
        encoding.data["input_ids"]=torch.flatten(padded_input_tensor)

        attention_mask = encoding.data["attention_mask"]
        padded_attention_mask = self.tokenizer.pad_token_id * torch.ones(
            (attention_mask.shape[0], self.args["max_input_length"]), dtype=attention_mask.dtype, device=attention_mask.device
        )
        padded_attention_mask[:, : attention_mask.shape[-1]] = attention_mask
        encoding.data["attention_mask"] = torch.flatten(padded_attention_mask)

        target = encoding.data["labels"]
        padded_target_tensor = self.tokenizer.pad_token_id * torch.ones(
            (target.shape[0], self.args["max_target_length"]), dtype=target.dtype, device=target.device
        )
        padded_target_tensor[:, : target.shape[-1]] = target
        encoding.data["labels"]=torch.flatten(padded_target_tensor)

        #out["decoder_input_ids"]=T5PreTrainedModel._shift_right(input_ids=out["labels"])
        return encoding.data
