from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
import json

dp=Dataprocessor_test(T5Tokenizer.from_pretrained("out"),"")
tokenizer = T5Tokenizer.from_pretrained("out")
model = T5ForConditionalGeneration.from_pretrained("")
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

for ques in data:
    if "question" in ques:
        input=ques["question"]
        print(input)
        print(ques["sparql_wikidata"])
        labels="Das Haus ist schön"
        sample = dp.process_sample(input,labels)
        i=dp.process_sample(input).input_ids
        l=dp.process_sample(labels).input_ids
        # the forward function automatically creates the correct decoder_input_ids
        out = model.generate(input_ids = i,max_length=650)

        print(tokenizer.decode(out[0], skip_special_tokens=True)+"\n")
