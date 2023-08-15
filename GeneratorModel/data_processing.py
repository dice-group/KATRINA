from torch.utils.data import Dataset
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
    def process_training_ds(self,data):
        dataset=ListDataset([])
        for sample in data:
            dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def process_sample(self,input,label=None):
        pass
class Dataprocessor_test(Dataprocessor):

    def process_sample(self,input,label=None):
        encoding = self.tokenizer(text=input,text_target=label, return_tensors="pt")
        return encoding
