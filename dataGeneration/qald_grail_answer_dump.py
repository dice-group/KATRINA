import json

grail_ans=json.load(open("../grail_arcane_dev_processed.json"))
grail_qald=json.load(open("../qa-data/GrailQA_v1.0/grail_dev_qald_corrected.json"))

for qe in grail_qald["questions"]:
    answers=grail_ans[str(qe["id"])]["answer"]
    if len(qe["answers"][0]["results"]["bindings"])>0:
        ans_dict=qe["answers"][0]["results"]["bindings"][0]
    else:
        ans_dict={'value': {'type': 'uri', 'value': 'http://rdf.freebase.com/ns/dm'}}
    bindings=[]
    for a in answers:
        an={'value': {'type': 'uri', 'value': "wrong"}}
        if ans_dict["value"]["type"]=="uri":
            an = {'value': {'type': 'uri', 'value': "wrong"}}
            an["value"]["value"]='http://rdf.freebase.com/ns/'+a
        else:
            an["value"]["type"]=ans_dict["value"]["type"]
            an["value"]["datatype"] = ans_dict["value"]["datatype"]
            an["value"]["value"]=a
        print(an)
        bindings.append(an)
    qe["answers"][0]["results"]["bindings"]=bindings

json.dump(grail_qald,open("qald_out_arcane.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)