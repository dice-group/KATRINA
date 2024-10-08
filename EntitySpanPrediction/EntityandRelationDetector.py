from parameters import KATRINAParser
from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_processing import Dataprocessor_test
from PrefixTrie import Trie_not_recursive
import json
import torch
import pickle
from tqdm import tqdm
class EntityAndRelationDetector():
    def __init__(self,params):
        self.device = "cuda:"+str(params["cuda_device_id"]) if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(params["tokenizer_name"]if params["tokenizer_name"]
                                                     else params["model_name"])
        self.dp = Dataprocessor_test(self.tokenizer, "")

        self.model = T5ForConditionalGeneration.from_pretrained(params["parameter_path_prefix"]
                                                                +params["pretrained_model_path"])
        self.model.to(self.device)
        with open(params["parameter_path_prefix"]+params["entity_trie_file"], "rb") as f:
            self.trie_entities = pickle.load(f)
        with open(params["parameter_path_prefix"]+params["relation_trie_file"], "rb") as f:
            self.trie_relations = pickle.load(f)
        self.relation_pre=self.tokenizer.encode("relations:")[0:-1]
        self.relations=pickle.load(open(params["parameter_path_prefix"]+params["relation_dict_file"],"rb"))
        self.entities=pickle.load(open(params["parameter_path_prefix"]+params["entity_dict_file"],"rb"))


    def prefix_allowed_tokens_fn(self,ar1, ar2):
        re = ",".join([str(el) for el in self.relation_pre])
        curr_state = ar2.tolist()
        curr_state_str = ",".join([str(el) for el in curr_state])
        if curr_state[len(curr_state) - 1] == 6306 or curr_state[len(curr_state) - 1] == 14920 and curr_state[
            len(curr_state) - 2] == 6306:
            return list(self.tokenizer.get_vocab().values())
        if 908 in curr_state:
            an_index = len(curr_state) - 1 - curr_state[::-1].index(908)
            prefix = curr_state[an_index - 3:an_index + 1]
            if prefix == [784, 279, 8579, 908]:
                ent_seq = curr_state[an_index + 1:]
                if re in curr_state_str:
                    return self.trie_relations.get(ent_seq)
                else:
                    return self.trie_entities.get(ent_seq)
        return list(self.tokenizer.get_vocab().values())

    def predict(self,sentence):
        i = self.dp.process_sample(sentence+"[SEP]target_wikidata").input_ids
        out = self.model.generate(input_ids=i.to(self.device), max_length=650)
        #out = self.model.generate(input_ids=i.to(self.device), max_length=650,
        #                          prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn)
        out_str=self.tokenizer.decode(out[0], skip_special_tokens=True)
        # print(out_str)
        try:
            entstr=out_str[out_str.index("entities: ")+len("entities: "):out_str.index(", relations")]
            all_ents=[el.replace("[BEG]","").replace("[END]","")for el in entstr.split("[END], [BEG]")]
            relations_str = out_str[out_str.index("relations: ") + len("relations: "):-1]
            all_relations = [el.replace("[BEG]","").replace("[END]","")for el in relations_str.split("[END], [BEG]")]
        except ValueError:
            all_ents=[]
            all_relations=[]
        return [{"label":el,"uri":self.entities[el]}for el in all_ents if el in self.entities],\
               [{"label":el,"uri":self.relations[el]}for el in all_relations if el in self.relations]