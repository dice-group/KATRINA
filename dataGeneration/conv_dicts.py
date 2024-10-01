import pickle

id_to_title  = pickle.load(open("wikidata_labels_update.pkl","rb"))
relation_ids=pickle.load(open("relation_labels.sav","rb"))

inv_map_ent = {v: k for k, v in id_to_title.items()}
pickle.dump(inv_map_ent,open("label_to_entity_wk.pkl","wb"))

inv_map_rel = {v: k for k, v in relation_ids.items()}
#pickle.dump(inv_map_ent,open("label_to_entity_wk.pkl"))
pickle.dump(inv_map_rel,open("label_to_relation_wk.pkl","wb"))

