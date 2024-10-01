import json

annotated_pred=json.load(open("../qa-data/combined_qald/test/qald_pred.json","r",encoding="utf-8"))
for i in range(len(annotated_pred)):
    annotated_pred[i]["uid"]=i

json.dump(annotated_pred,open("../qa-data/combined_qald/test/qald_pred_ids.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

