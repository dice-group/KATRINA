import json
data_lcquad_train=json.load(open("../qa-data/combined/train/lcquad.json",encoding="utf-8"))
data_lcquad_test=json.load(open("../qa-data/combined/test/lcquad.json",encoding="utf-8"))
data_lcquad_train.extend(data_lcquad_test)
data_qald_train=json.load(open("../qa-data/combined_qald/train/qald.json",encoding="utf-8"))
data_lcquad_train.extend(data_qald_train)
data_qald_test=json.load(open("../qa-data/combined_qald/test/qald.json",encoding="utf-8"))
data_lcquad_train.extend(data_qald_test)
#ent_pred=json.load(open("../qa-data/LCQUAD/train_pred_resource.json",encoding="utf-8"))
print("")
import pickle
wkgenre=pickle.load(open("../GENRE/text_to_wikidata_id","rb"))
not_found=set()
addition={}

for i in range(len(data_lcquad_train)):
    if "entities"in data_lcquad_train[i]:
        ent_gold={el["label"]:el["uri"].replace("http://www.wikidata.org/entity/","") for el in data_lcquad_train[i]["entities"]}
    else:
        ent_gold= {}
    #ent_found=set([el["uri"] for el in ent_pred[i]["entities"]])
    #ent_found_label = set([el["label"] for el in ent_pred[i]["entities"]])
    for l in ent_gold.keys():
        if l in wkgenre:
            print(wkgenre[l])
        else:
            addition[l]=ent_gold[l]
labels_to_wikidata_id = {**wkgenre, **addition}
wikidata_id_to_label = dict((v,k) for k,v in labels_to_wikidata_id.items())
pickle.dump(labels_to_wikidata_id,open("../precomputed/EntityLinking/labels_to_wikidata_id.pkl","wb"))
pickle.dump(wikidata_id_to_label,open("../precomputed/EntityLinking/wikidata_id_to_label.pkl","wb"))

