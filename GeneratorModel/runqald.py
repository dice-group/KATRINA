from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_processing import Dataprocessor_test
import json
from SPARQLWrapper import SPARQLWrapper,JSON
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dp=Dataprocessor_test(T5Tokenizer.from_pretrained("t5-large"),"")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("out-combined-simple-limtest/checkpoint-213500")
model.to(device)
data=json.load(open("../qa-data/LCQUAD/lcquad-test-quald.json","r",encoding="utf-8"))
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
def get_answers(query_str):
    try:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(query_str)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if not "boolean" in results:
            results["head"]["vars"]=results["head"]["vars"][0]
        return results

    except:
        print("FAILED"+query_str)
        return {'head': {'vars': 'result'}, 'results': {'bindings': []}}

for ques in data["questions"]:
        input=ques["question"][0]["string"]+"[SEP]target_wikidata"
        print(input)
        print(ques["query"]["sparql"])
        labels="emp"
        sample = dp.process_sample(input,labels)
        i=dp.process_sample(input).input_ids
        #l=dp.process_sample(labels).input_ids
        # the forward function automatically creates the correct decoder_input_ids
        out = model.generate(input_ids = i.to(device),max_length=650)
        query = tokenizer.decode(out[0], skip_special_tokens=True)+"\n"
        query=query.replace("_result_","?result")
        query=query.replace("_var_", "?var")
        query=query.replace("_cbo_", "{")
        query=query.replace("_cbc_", "}")
        print(query)
        answers=get_answers(query)
        print(answers)
        ques["answers"]=answers
json.dump(data,open("../qa-data/LCQUAD/lcquad-test-output-t5large.json","w",encoding="utf-8"))