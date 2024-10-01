import json

annotated=json.load(open("../qa-data/combined_qald/train/qald_updated.json",encoding="utf-8"))
annotated_pred=json.load(open("../qa-data/combined_qald/train/pred_train_qald_exp.json",encoding="utf-8"))

for i in range(len(annotated)):
    if "entities" in annotated[i]:
        ent_gd=annotated[i]["entities"]
    else:
        ent_gd=[]
    ent_pred=annotated_pred[i]["entities"]
    ent_gd.extend(ent_pred)
    entities=[]
    ent_found=set()
    for el in ent_gd:
        uri=el["uri"].replace('http://www.wikidata.org/entity/','')
        if not uri in ent_found:
            entities.append({"uri":uri,"label":el["label"]})
            ent_found.add(uri)
    annotated[i]["entities"]=entities

    if "relations" in annotated[i]:
        ent_gd = annotated[i]["relations"]
    else:
        ent_gd = []
    ent_pred = annotated_pred[i]["relations"]

    ent_gd.extend(ent_pred)
    relations = []
    rel_found = set()

    for el in ent_gd:
        uri = el["uri"].replace('http://www.wikidata.org/prop/direct/', '').replace("http://www.wikidata.org/prop/qualifier/", "") \
        .replace("http://www.wikidata.org/prop/statement/", "").replace("http://www.wikidata.org/prop/","")
        if not uri in rel_found:
            relations.append({"uri": uri, "label": el["label"]})
            rel_found.add(uri)
    annotated[i]["relations"] = relations
json.dump(annotated,open("../qa-data/combined_qald/train/qald_el.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)