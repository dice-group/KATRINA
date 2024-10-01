import json

#annotated=json.load(open("../qa-data/combined_qald/train/qald_updated.json",encoding="utf-8"))
annotated_pred=json.load(open("../qa-data/combined/test/pred_test_exp.json",encoding="utf-8"))

for i in range(len(annotated_pred)):
    ent_pred=annotated_pred[i]["entities"]
    entities=[]
    ent_found=set()
    for el in ent_pred:
        uri=el["uri"].replace('http://www.wikidata.org/entity/','')
        if not uri in ent_found:
            entities.append({"uri":uri,"label":el["label"]})
            ent_found.add(uri)
    annotated_pred[i]["entities"]=entities

    ent_pred = annotated_pred[i]["relations"]

    relations = []
    rel_found = set()

    for el in ent_pred:
        uri = el["uri"].replace('http://www.wikidata.org/prop/direct/', '').replace("http://www.wikidata.org/prop/qualifier/", "") \
        .replace("http://www.wikidata.org/prop/statement/", "").replace("http://www.wikidata.org/prop/","")
        if not uri in rel_found:
            relations.append({"uri": uri, "label": el["label"]})
            rel_found.add(uri)
    annotated_pred[i]["relations"] = relations
json.dump(annotated_pred,open("../qa-data/combined/test/lcquad_pred.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)