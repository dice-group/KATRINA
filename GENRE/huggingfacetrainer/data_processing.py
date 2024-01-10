import json
import torch

from torch.utils.data import Dataset
from transformers import T5PreTrainedModel
import  rdflib.plugins.sparql as sparql

from rdflib.plugins.sparql import algebra
import pickle
from tqdm import tqdm

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
        print("tokenize data")
        for sample in tqdm(samples):
            dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def process_sample(self,input,label=None):
        pass
class Dataprocessor_test(Dataprocessor):

    def process_sample(self,input,label=None):
        encoding = self.tokenizer(text=input,text_target=label, return_tensors="pt",
                                  )
        return encoding

class Dataprocessor_KBQA_basic(Dataprocessor):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
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





class Dataprocessor_Combined_simple(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, path_to_ds):
        prefixes = """
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wds: <http://www.wikidata.org/entity/statement/>
                PREFIX wdv: <http://www.wikidata.org/value/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX p: <http://www.wikidata.org/prop/>
                PREFIX ps: <http://www.wikidata.org/prop/statement/>
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX bd: <http://www.bigdata.com/rdf#>
                """
        lcquad_data = json.load(open(path_to_ds+"/lcquad.json"))
        entitylabels_alt = pickle.load(open("../GENRE/alt_dict_wikidata.pkl", "rb"))
        samples=[]
        for question in tqdm(lcquad_data):
            # print(question)
            if "entities" in question and "relations" in question and question["question"] is not None:
                question_str = question["question"]
                query = question["sparql_wikidata"]
                for ent in question["entities"]:
                    key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                    if key in entitylabels_alt:
                        #ent_str+=entitylabels[key]+", "
                        sample = {"input": question_str+"[START_ENT]"+entitylabels_alt[key]["label"]+
                                           "[END_ENT] [SEP]target_wikidata", "label": entitylabels_alt[key]["label"]}
                        samples.append(sample)
                    if "alternatives" in entitylabels_alt[key]:
                        for alt in entitylabels_alt[key]["alternatives"]:
                            sample = {"input": question_str +
                                               "[START_ENT]"+alt +
                                               "[END_ENT] [SEP]target_wikidata" , "label": entitylabels_alt[key]["label"]}
                            samples.append(sample)
        grail_qa = json.load(open(path_to_ds+"/grail.json"))
        entitylabels = pickle.load(open("../GENRE/label_dict_freebase.pkl", "rb"))
        entitylabels_alt = pickle.load(open("../GENRE/alt_dict_freebase.pkl", "rb"))
        #print(list(entitylabels.keys()))
        for el in tqdm(grail_qa):
            question_str = el["question"]
            nodes = el["graph_query"]["nodes"]
            for n in nodes:
                if not n["node_type"] == "literal":
                    sample = {"input": question_str+"[START_ENT]"+entitylabels_alt[n["id"]]["label"]+
                                           "[END_ENT] [SEP]target_freebase", "label": entitylabels[n["id"]]}
                    samples.append(sample)
                    if "alternatives" in entitylabels_alt[n["id"]]:
                        for alt in entitylabels_alt[n["id"]]["alternatives"]:
                            sample = {"input": question_str +
                                               "[START_ENT]" + alt +
                                               "[END_ENT] [SEP]target_freebase", "label": entitylabels[n["id"]]}
                            samples.append(sample)

        return samples
class Dataprocessor_QALD(Dataprocessor_KBQA_basic):
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






