import requests
import json
from EntityLinkingFreebase import surface_index_memory
from EntityLinkingFreebase.bert_entity_linker import BertEntityLinker
import json
surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "EntityLinking/data/entity_list_file_freebase_complete_all_mention", "EntityLinking/data/surface_map_file_freebase_complete_all_mention",
        "../freebase_complete_all_mention")
entity_linker = BertEntityLinker(surface_index, model_path="/NER/BERT_NER/trained_ner_model/")
def query(query_str):
    resp=requests.post('http://neamt.cs.upb.de:6100/custom-pipeline',headers={'Content-Type': 'application/x-www-form-urlencoded'}, data='components=babelscape_ner, mgenre_el&query='+query_str)
    #resp = requests.post('http://neamt.cs.upb.de:6100/custom-pipeline',
    #                     headers={'Content-Type': 'application/x-www-form-urlencoded'},
    #                     data='components=mgenre_el&'+ query_str)
    return resp.json()
files=[
    "qa-data/GrailQA_v1.0/grailqa_v1.0_dev.json",
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
                        '''
                        try:
                            entities=query(question)


                            print(entities)

                            for ent in entities["ent_mentions"]:
                                if "link" in ent:
                                    ent_list.append({"mention":question[ent["start"]:ent["end"]],"id":ent["link"],"friendly_name":ent["surface_form"]})
                        except:
                            print("query failed")
                        
                        sample["wikidata_entities"]=ent_list
                        '''
                        entities = entity_linker.identify_entities(question)
                        print(entities)
                        ent_list=[]
                        for ent in entities:
                            ent_list.append(
                                {"mention": ent.mention, "id": ent.entity.id, "friendly_name": ent.entity.name})
                        sample["freebase_entities"] = ent_list
                json.dump(data, open("qa-data/GrailQA_v1.0/grailqa_v1.0_dev_with_entities.json", "w", encoding="utf-8"))