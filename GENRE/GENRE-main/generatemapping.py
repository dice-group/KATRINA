import json
import pickle

title_map={}
with open("C:/Users/danvo/Desktop/EntityLinkingGenerative/data/kilt_knowledgesource.json","r",encoding="utf-8")as file:
    for ln in file:
        data=json.loads(ln)
        if "wikidata_info" in data:
            title_map[data["wikipedia_title"]]=data["wikidata_info"]["wikidata_id"]
        else:
            print(data["wikipedia_title"])
pickle.dump(title_map,open("text_to_wikidata_id","wb"))

