from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
import json

dp=Dataprocessor_test(T5Tokenizer.from_pretrained("out"),"")
tokenizer = T5Tokenizer.from_pretrained("out")
model = T5ForConditionalGeneration.from_pretrained("out-combined-simple/checkpoint-103000")
data=json.load(open("../qa-data/combined/test/lcquad.json","r",encoding="utf-8"))

#print(tokenizer.decode(out[0], skip_special_tokens=True))
input="semaphore railway line is on the rail network named what?[SEP]target_wikipedia"
print(input)
print("SELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n?x0 :type.object.type :rail.rail_network . \nVALUES ?x1 { :m.03qcvdj } \n?x0 :rail.rail_network.railways ?x1 . \nFILTER ( ?x0 != ?x1  )\n}\n}")
labels="Label"
sample = dp.process_sample(input,labels)
i=dp.process_sample(input).input_ids
l=dp.process_sample(labels).input_ids
out = model.generate(input_ids = i,max_length=650)
print(tokenizer.decode(out[0], skip_special_tokens=True)+"\n")

        # the forward function automatically creates the correct decoder_input_ids
out = model.generate(input_ids = i,max_length=650)



