from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
import json
import torch
device = "cuda:1" if torch.cuda.is_available() else "cpu"
dp=Dataprocessor_test(T5Tokenizer.from_pretrained("t5-base"),"")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("span_prediction_model")
data=json.load(open("../qa-data/QALD/QALD_10.json","r",encoding="utf-8"))
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

for ques in data["questions"]:
    if "question" in ques:
        for lang in ques["question"]:
            if lang["language"]=="en":
                input=lang["string"]
        #input=ques["question"]
        print(input)
        #print(ques["sparql_wikidata"])
        labels="Das Haus ist schön"
        sample = dp.process_sample(input,labels)
        #print(sample)
        i=dp.process_sample(input).input_ids
        l=dp.process_sample(labels).input_ids
        # the forward function automatically creates the correct decoder_input_ids
        out = model.generate(input_ids = i,max_length=650)

        print(tokenizer.decode(out[0], skip_special_tokens=True)+"\n")
