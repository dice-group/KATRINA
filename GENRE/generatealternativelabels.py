import json
import pickle
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper,JSON
def get_alternatives_freebase(entity_str):
    prefix="PREFIX : <http://rdf.freebase.com/ns/>"
    sparql = SPARQLWrapper("https://freebase.data.dice-research.org/sparql")
    sparql.setQuery(prefix+"select ?ent ?lab where{?ent <http://rdf.freebase.com/ns/common.topic.alias> ?lab FILTER (lang(?lab) = 'en')VALUES ?ent {"+entity_str+"} "
                        "VALUES ?rel{<http://rdf.freebase.com/ns/common.topic.alias> <http://rdf.freebase.com/key/wikipedia.en>}}")

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    alternatives = {}
    for b in results["results"]["bindings"]:
        key = b["ent"]["value"].replace("http://rdf.freebase.com/ns/", "")
        if not key in alternatives:
            alternatives[key] = []
        curr = set(alternatives[key])
        curr.add(b["lab"]["value"])
        alternatives[key]=list(curr)

    return alternatives
def get_alternatives_wikidata(entity_str):

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery("select ?ent ?lab where {?ent skos:altLabel ?lab FILTER (lang(?lab) = 'en')VALUES ?ent {"+entity_str+"}}")

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    alternatives={}
    for b in results["results"]["bindings"]:
        key=b["ent"]["value"].replace("http://www.wikidata.org/entity/","")
        if not key in alternatives:
            alternatives[key]=[]
        curr=set(alternatives[key])
        curr.add(b["lab"]["value"])
        alternatives[key].append(list(curr))

    return alternatives

'''
lcquad_data = json.load(open("../qa-data/combined/train/lcquad.json"))
entitylabels = pickle.load(open("../precomputed/wikidata_labels.sav", "rb"))
entities=set()
for question in tqdm(lcquad_data):
    # print(question)
    if "entities" in question and "relations" in question and question["question"] is not None:
        for ent in question["entities"]:
            key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
            entities.add(key)

lcquad_data = json.load(open("../qa-data/combined/test/lcquad.json"))
for question in tqdm(lcquad_data):
    # print(question)
    if "entities" in question and "relations" in question and question["question"] is not None:
        for ent in question["entities"]:
            key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
            entities.add(key)
ind=0
entities=list(entities)
alt_dict={}
while ind<len(entities):
    print(ind)
    if ind+50<len(entities):
        to_query=entities[ind:ind+50]
    else:
        to_query = entities[ind:len(entities)]
    query_str=" wd:".join(to_query)
    query_str="wd:"+query_str
    alt=get_alternatives_wikidata(query_str)
    for ent in entities[ind:ind+50]:
        alt_dict[ent]={"label":entitylabels[ent]}
        if ent in alt:
            alt_dict[ent]["alternatives"]=alt[ent]
    ind+=50
pickle.dump(alt_dict,open("alt_dict_wikidata.pkl","wb"))
'''
grail_data = json.load(open("../qa-data/combined/train/grail.json"))
#entitylabels = pickle.load(open("../precomputed/wikidata_labels.sav", "rb"))
alt_dict={}
for question in tqdm(grail_data):
    nodes = question["graph_query"]["nodes"]
    for n in nodes:
        if not n["node_type"] == "literal":
            alt_dict[n["id"]] = {"label":n["friendly_name"]}

lcquad_data = json.load(open("../qa-data/combined/test/grail.json"))
for question in tqdm(lcquad_data):
    nodes = question["graph_query"]["nodes"]
    for n in nodes:
        if not n["node_type"] == "literal":
            alt_dict[n["id"]] = {"label":n["friendly_name"]}
ind=0
entities=list(alt_dict.keys())
while ind<len(entities):
    print(ind)
    if ind+50<len(entities):
        to_query=entities[ind:ind+50]
    else:
        to_query = entities[ind:len(entities)]
    query_str=" :".join(to_query)
    query_str=":"+query_str
    alt=get_alternatives_freebase(query_str)
    for ent in entities[ind:ind+50]:
        if ent in alt:
            alt_dict[ent]["alternatives"]=alt[ent]
    ind+=50
pickle.dump(alt_dict,open("alt_dict_freebase.pkl","wb"))



