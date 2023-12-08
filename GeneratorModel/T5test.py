from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
import json
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dp=Dataprocessor_test(T5Tokenizer.from_pretrained("t5-large"),"")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large-baseline")
model.to(device)
data=json.load(open("../qa-data/combined/test/lcquad.json","r",encoding="utf-8"))

'''
input="How many chancellors did Germany have?"
#print(input)
#print(ques["sparql_wikidata"])
labels="Das Haus ist schön"
sample = dp.process_sample(input,labels)
i=dp.process_sample(input).input_ids
l=dp.process_sample(labels).input_ids
# the forward function automatically creates the correct decoder_input_ids
out = model.generate(input_ids = i,max_length=650)
'''
#print(tokenizer.decode(out[0], skip_special_tokens=True))


class token_control():
    def __init__(self,tokenizer,entities):
        self.tokenizer=tokenizer
        self.entity_start_tag_ids = tokenizer.encode("wd:Q")[0:-1]
        self.curr_ent_trie=None
        self.end_id = 3
    def compute_ent_prefix_trie(self,entity_list):
        trie={}
        for ent_ment in entity_list:
            ent=ent_ment["uri"].replace("http://www.wikidata.org/entity/","")
            token_ids=tokenizer.encode(ent)[1:-1]
            #tokens = tokenizer.tokenize(ent)
            curr_trie=trie
            for tk in token_ids:
                if tk in trie:
                    curr_trie=trie[tk]
                else:
                    curr_trie[tk]={}
                    curr_trie=curr_trie[tk]
        self.curr_ent_trie=trie

    def test_prefix_allowed_tokens_funktion(self,ar1,ar2):
        vo=tokenizer.get_vocab().values()
        curr_state=ar2.tolist()
        if self.end_id in curr_state:
            l_ind_end_start=len(curr_state) - 1 - curr_state[::-1].index(self.end_id)
            tail_list=curr_state[l_ind_end_start:len(curr_state)]
            in_ent=tail_list[0:len(self.entity_start_tag_ids)]==self.entity_start_tag_ids
            if in_ent:
                ent_state=tail_list[len(self.entity_start_tag_ids):len(tail_list)]
                trie=self.curr_ent_trie
                for tk in ent_state:
                    trie=trie[tk]
                allowed_tokens=list(trie.keys())
                if len(allowed_tokens)==0:
                    return [self.end_id]
                else:
                    return allowed_tokens
        return list(vo)

token_controller=token_control(tokenizer,[])

for ques in data:
    if "question" in ques:
        input=ques["question"]+"[SEP]target_wikidata"
        print(input)
        token_controller.compute_ent_prefix_trie(ques["entities"])
        print(ques["sparql_wikidata"])
        labels="Das Haus ist schön"
        sample = dp.process_sample(input,labels)
        i=dp.process_sample(input).input_ids
        l=dp.process_sample(labels).input_ids
        # the forward function automatically creates the correct decoder_input_ids
        out = model.generate(input_ids = i.to(device),max_length=650,prefix_allowed_tokens_fn=token_controller.test_prefix_allowed_tokens_funktion)
        entity_start_tags = tokenizer.tokenize("wd:Q")
        entity_start_tag_ids = tokenizer.encode("wd:Q")
        seq=tokenizer.decode(out[0], skip_special_tokens=True)
        print(seq)

