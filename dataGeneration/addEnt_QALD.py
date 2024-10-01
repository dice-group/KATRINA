import json
import re
import pickle
from SPARQLWrapper import SPARQLWrapper,JSON

def queryEntities(entities):
    offset = 0
    values_str=" ".join(["wd:"+en for en in entities])

    sparql = SPARQLWrapper(
            "https://query.wikidata.org/bigdata/namespace/wdq/sparql",
            agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')

    sparql.setQuery('''
            SELECT ?ent ?label WHERE {

        ?ent rdfs:label ?label .
        VALUES ?ent {'''+values_str+'''}
        FILTER (langMatches( lang(?label), "EN" ) )
            }
        ''')
    sparql.setReturnFormat(JSON)
    offset = offset + 1000
    res = sparql.queryAndConvert()
    label_dict={}
    for el in res["results"]["bindings"]:
        if el["label"]["xml:lang"]=="en":
            label_dict[el["ent"]["value"].replace("http://www.wikidata.org/entity/","")]=el["label"]["value"]
    return label_dict


annotated=json.load(open("../qa-data/combined_qald/test/qald.json",encoding="utf-8"))
qald=json.load(open("../qa-data/QALD/qald_10.json",encoding="utf-8"))
org_strs=[]
for el in qald["questions"]:
    for lg in el["question"]:
        if lg["language"]=="en":
            org_strs.append(lg["string"])
#id_to_title  = {v: k for k, v in pickle.load(open("../GENRE/text_to_wikidata_id","rb")).items()}
id_to_title  = pickle.load(open("wikidata_labels_update.pkl","rb"))
relation_ids=pickle.load(open("relation_labels.sav","rb"))
relation_ids["P2522"]="victory"
relation_ids['P1196']="manner of death"
relation_ids['P2512']="has spin-off"
relation_ids['P2521']="female form of label"
missing=set()
for i in range(len(annotated)):
    query=annotated[i]["sparql_wikidata"]
    entitiy_pattern=r"wd:(Q[0-9]+)"
    annotated[i]["question"]=org_strs[i]
    entities=re.findall(entitiy_pattern,query)
    en_dict=[]
    for ent in entities:
        en_dict.append({"uri":ent,"label":id_to_title[ent]})
    annotated[i]["entities"]=en_dict
    print(entities)
    relation_pattern = r"(P[0-9]+)"
    relations = re.findall(relation_pattern, query)
    rel_dict=[]
    for rel in relations:
        if rel in relation_ids:
            rel_dict.append({"uri":rel,"label":relation_ids[rel]})
        else:
            missing.add(rel)
    annotated[i]["relations"]=rel_dict
print("missing")

#add=queryEntities(missing)
#id_to_title.update(add)
json.dump(annotated,open("../qa-data/combined_qald/test/qald_updated.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
pickle.dump(relation_ids,open("relation_ids.pkl","wb"))
