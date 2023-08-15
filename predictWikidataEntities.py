import requests
import json

def query(query_str):
    resp=requests.post('http://neamt.cs.upb.de:6100/custom-pipeline',headers={'Content-Type': 'application/x-www-form-urlencoded'}, data='components=babelscape_ner, mgenre_el&query='+query_str)
    #resp = requests.post('http://neamt.cs.upb.de:6100/custom-pipeline',
    #                     headers={'Content-Type': 'application/x-www-form-urlencoded'},
    #                     data='components=mgenre_el&'+ query_str)
    return resp.json()
files=[
    "qa-data/LCQUAD/train_with_freebase_ent.json",
    #"dataset_updates/data_qald_test.json",
    #"dataset_updates/data_qald_train.json",
    #"dataset_updates/data_train.json"
    ]

for file in files:
        with open(file, "r", encoding="utf-8") as data:
                data = json.load(data)
                for sample in data:
                    question=sample["question"]
                    ent_list = []
                    if question is not None:
                        try:
                            entities=query(question)


                            print(entities)

                            for ent in entities["ent_mentions"]:
                                if "link" in ent:
                                    ent_list.append({"mention":question[ent["start"]:ent["end"]],"id":ent["link"],"friendly_name":ent["surface_form"]})
                        except:
                            print("query failed")
                        sample["wikidata_entities"]=ent_list

                json.dump(data, open("qa-data/LCQUAD/train_with_freebase_and_wikidata_ent.json", "w", encoding="utf-8"))