from EntityLinking import surface_index_memory
from EntityLinking.bert_entity_linker import BertEntityLinker
import json
surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "EntityLinking/data/entity_list_file_freebase_complete_all_mention", "EntityLinking/data/surface_map_file_freebase_complete_all_mention",
        "../freebase_complete_all_mention")
entity_linker = BertEntityLinker(surface_index, model_path="/NER/BERT_NER/trained_ner_model/")
files=[
    "qa-data/LCQUAD/train.json",
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
                        entities=entity_linker.identify_entities(question)
                        print(entities)

                        for ent in entities:
                            ent_list.append({"mention":ent.mention,"id":ent.entity.id,"friendly_name":ent.entity.name})
                        sample["freebase_entities"]=ent_list
                    print(question)
                    print(ent_list)
        json.dump(data,open("qa-data/LCQUAD/train_with_freebase_ent.json","w",encoding="utf-8"))
