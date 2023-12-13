from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_processing import Dataprocessor_test
import json
from SPARQLWrapper import SPARQLWrapper,JSON
import torch
ent_data=json.load(open("../qa-data/combined/test/grail.json","r",encoding="utf-8"))
enitity_map={}
for ques in ent_data:
    id=ques["qid"]
    entities=ques["graph_query"]["nodes"]
    enitity_map[id]=entities
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dp=Dataprocessor_test(T5Tokenizer.from_pretrained("t5-large"),"")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("/data/KATRINA/out-simple-entities")
model.to(device)

data=json.load(open("../qa-data/GrailQA_v1.0/grailqa_dev_qald.json","r",encoding="utf-8"))
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
    prefixes="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> "
    try:
        sparql = SPARQLWrapper("https://freebase.data.dice-research.org/sparql")
        sparql.setQuery(prefixes+query_str)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if not "boolean" in results:
            results["head"]["vars"]=results["head"]["vars"][0]
        return results

    except:
        print("FAILED"+query_str)
        return {'head': {'vars': 'result'}, 'results': {'bindings': []}}

for ques in data["questions"]:
        input=ques["question"][0]["string"]+"[SEP] "
        if ques["id"] in enitity_map:
            nodes = enitity_map[ques["id"]]
            for n in nodes:
                if not n["node_type"] == "literal":
                    input += n["friendly_name"] + " : " + n["id"] + " , "
        input += "[SEP]target_freebase"
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
json.dump(data,open("../qa-data/GrailQA_v1.0/grail-dev-output-entities-t5large.json","w",encoding="utf-8"))