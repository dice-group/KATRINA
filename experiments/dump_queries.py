from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_processing import Dataprocessor_test
import json
from SPARQLWrapper import SPARQLWrapper,JSON
import torch
from parameters import KATRINAParser
import pickle
parser = KATRINAParser(add_model_args=True,add_training_args=True)
parser.add_model_args()
parser.add_inference_args()

args = parser.parse_args()
print(args)
params = args.__dict__

device = "cuda:"+str(params["cuda_device_id"]) if torch.cuda.is_available() else "cpu"

tokenizer = T5Tokenizer.from_pretrained(params["tokenizer_name"]if params["tokenizer_name"]
                                                     else params["model_name"])
dp=Dataprocessor_test(tokenizer,"")
model = T5ForConditionalGeneration.from_pretrained(params["pretrained_model_path"])
#model = T5ForConditionalGeneration.from_pretrained("t5-large-baseline")
model.to(device)
data = json.load(open(params["predict_file"], "r", encoding="utf-8"))
#data=json.load(open("../qa-data/QALD/qald_10.json","r",encoding="utf-8"))


#print(tokenizer.decode(out[0], skip_special_tokens=True))
def get_answer_generator(params):

    def get_answers_wikidata(query_str):
        try:
            sparql = SPARQLWrapper(params["wikidata_sparql_endpoint"])
            sparql.setQuery(query_str)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            if not "boolean" in results:
                results["head"]["vars"]=[results["head"]["vars"][0]]
            return results

        except:
            print("FAILED"+query_str)
            return {'head': {'vars': ['result']}, 'results': {'bindings': []}}
    def get_answers_freebase(query_str):
        prefixes="PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> "
        try:
            sparql = SPARQLWrapper(params["freebase_sparql_endpoint"])
            sparql.setQuery(prefixes+query_str)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            if not "boolean" in results:
                results["head"]["vars"]=[results["head"]["vars"][0]]
            return results

        except:
            print("FAILED"+query_str)
        return {'head': {'vars': ['result']}, 'results': {'bindings': []}}

    if params["benchmark_KG"]=="wikidata":
        return get_answers_wikidata
    else:
        return get_answers_freebase

def freebase_resource_generator(add_entities=True,add_relations=True,use_gold_resources=False):

    if use_gold_resources:
        with(open(params["gold_resource_benchmark"]))as file:
            ent_data=json.load(file)
            enitity_map = {}
            relation_map= {}
            for ques in ent_data:
                id = ques["qid"]
                entities = ques["graph_query"]["nodes"]
                enitity_map[id] = entities
                relations=ques["graph_query"]["edges"]
                relation_map[id] = relations
    else:
        with(open(params["freebase_qa_schema_file"], "r")) as file:
            ent_schema = {}
            for ln in file:
                el = json.loads(ln)
                ent_schema[el["qid"]] = el
        with(open(params["freebase_qa_entity_file"], "r")) as file:
            grail_ent = json.load(file)
        with(open(params["freebase_type_dict"], "rb")) as file:
            freebase_types = pickle.load(file)
        with(open(params["freebase_relation_dict"], "rb")) as file:
            freebase_relations = pickle.load(file)
    def add_resources_freebase(question):
        input = question["question"][0]["string"] + "[SEP] "
        question_id = question["id"]
        if add_entities and str(question_id) in grail_ent:
            entities = grail_ent[str(question_id)]["entities"]
            input += "entities: "
            for ent in list(entities.keys()):
                input += entities[ent]["friendly_name"] + " : " + ent + " , "
            if question_id in ent_schema:
                nodes = ent_schema[question_id]["classes"][:5]
                for n in nodes:
                    input +=freebase_types[n] + " : " + n + " , "
        if add_relations and question_id in ent_schema:
            input += "relations: "
            if question_id in ent_schema:
                nodes = ent_schema[question_id]["relations"][:5]
                for n in nodes:
                    input +=freebase_relations[n] + " : " + n + " , "
        return input+"[SEP]target_freebase"
    def add_resources_freebase_gold(question):
        input = question["question"][0]["string"] + "[SEP] "
        question_id = question["id"]

        if add_entities and question_id in enitity_map:
            input += "entities: "
            nodes = enitity_map[question_id]
            for n in nodes:
                if not n["node_type"] == "literal":
                    input += n["friendly_name"] + " : " + n["id"] + " , "

        if add_relations and  question_id in relation_map:
            input += "relations: "
            edges = relation_map[question_id]
            for e in edges:
                input += e["friendly_name"] + " : " + e["relation"] + " , "
        return input+"[SEP]target_freebase"
    if use_gold_resources:
        return add_resources_freebase_gold
    else:
        return add_resources_freebase

def wikidata_resource_generater(add_entities=True,add_relations=True):
    ent_data = json.load(open(params["wikidata_bechmark_entities"], "r", encoding="utf-8"))
    enitity_map = {}
    for ques in ent_data:
        if "entities" in ques:
            id = ques["uid"]
            entities = ques["entities"]
            relations= ques["relations"]
            enitity_map[id] = {"entities":entities,"relations":relations}

    def add_resources_wikidata(question):
        input = question["question"][0]["string"] + "[SEP] "
        question_id=question["id"]
        if add_entities and question_id in enitity_map:
            entities = enitity_map[question_id]["entities"]
            input += "entities: "
            for ent in entities:
                input +=ent["label"] + " : " + ent["uri"].replace("http://www.wikidata.org/entity/", "") + " , "
        if add_relations and question_id in enitity_map:
            relations = enitity_map[question_id]["relations"]
            input += "relations: "
            for rel in relations:
                input +=rel["label"] + " : " + rel["uri"].replace("http://www.wikidata.org/prop/direct/", "") + " , "
        return input+"[SEP]target_wikidata"
    return add_resources_wikidata


if params["benchmark_KG"]=="wikidata":
    resource_predicor = wikidata_resource_generater(params["use_entities"], params["use_relations"])
else:
    if params["use_gold_res_freebase"]:
        resource_predicor = freebase_resource_generator(params["use_entities"], params["use_relations"],True)
    else:
        resource_predicor = freebase_resource_generator(params["use_entities"], params["use_relations"])
answer_generator =get_answer_generator(params)
query_dump={}
for ques in data["questions"]:
    if ques["question"][0]["string"] is not None:
            input=resource_predicor(ques)
            print(input)
            print(ques["query"]["sparql"])
            labels="emp"
            sample = dp.process_sample(input,labels)
            i=dp.process_sample(input).input_ids
            #l=dp.process_sample(labels).input_ids
            # the forward function automatically creates the correct decoder_input_ids
            out = model.generate(input_ids = i.to(device),max_length=650)
            query = tokenizer.decode(out[0], skip_special_tokens=True)+"\n"
            id=ques["id"]
            print(query)
            query_dump[id]={"candidates":[{"logical_form":query}]}


json.dump(query_dump,open(params["output_file"],"w",encoding="utf-8"),ensure_ascii=False,indent=4)