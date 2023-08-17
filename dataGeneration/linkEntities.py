import json
import pickle
entitylabels=pickle.load(open("../precomputed/wikidata_labels.sav","rb"))
relationlabels=pickle.load(open("../precomputed/relation_labels.sav","rb"))


data=json.load(open("../qa-data/LCQUAD/train-with-resources.json","r",encoding="utf-8"))
missingrelations=set()
for question in data:

    if "entities" in question:
        entities = []
        for entity in question["entities"]:
            ent_str=entity.replace("http://www.wikidata.org/entity/","")

            if ent_str in entitylabels:
                entity={"uri":entity,"label":entitylabels[ent_str]}
                entities.append(entity)
        question["entities"] = entities
    if "relations" in question:
        relations=[]
        for entity in question["relations"]:
            ent_str=entity.replace("http://www.wikidata.org/prop/direct/","")
            ent_str=ent_str.replace("http://www.wikidata.org/prop/qualifier/","")
            ent_str = ent_str.replace("http://www.wikidata.org/prop/statement/", "")
            ent_str = ent_str.replace("http://www.wikidata.org/entity/", "")
            ent_str = ent_str.replace("http://www.wikidata.org/prop/", "")
            if ent_str in relationlabels:
                entity={"uri":entity,"label":relationlabels[ent_str]}
            else:
               missingrelations.add(ent_str)
            relations.append(entity)
        question["relations"]=relations
pickle.dump(missingrelations,open("missing_rel","wb"))
json.dump(data, open("../qa-data/LCQUAD/train-with-resources-labels.json", "w", encoding="utf-8"))



    #for relation in question["relations"]: