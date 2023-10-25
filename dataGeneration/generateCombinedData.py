
import json
lcquad_data=json.load(open("../qa-data/LCQUAD/train-with-resources-labels-update.json"))
samples=[]
for question in lcquad_data:
    #print(question)
    if "entities"in question and "relations" in question:
        question_str = question["question"]
        query=question["sparql_wikidata"]

        for ent in question["entities"]:
            key=ent["uri"].replace("http://www.wikidata.org/entity/","")
            query=query.replace(key,ent["label"])
        for rel in question["relations"]:
            key=rel["uri"].replace("http://www.wikidata.org/prop/direct/","")
            query=query.replace(key,rel["label"])
        sample={"source":question_str,"target":query}

        samples.append(sample)

grail_qa=json.load(open("../qa-data/GrailQA_v1.0/grailqa_v1.0_train.json"))
for el in grail_qa:
    question_str=el["question"]
    query=el["sparql_query"]
    nodes=el["graph_query"]["nodes"]
    edges = el["graph_query"]["edges"]
    en = algebra.translateQuery(parsed_query)
    for e in edges:
        query=query.replace(e["relation"],e["friendly_name"])
    for n in nodes:
        query=query.replace(n["id"],n["friendly_name"])


    query = query.replace("\n", "")
    query=query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>","")
    query=query.replace(
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>","")
    query=query.replace(
        "PREFIX : <http://rdf.freebase.com/ns/> ","")
    sample = {"source": question_str, "target": query}
    samples.append(sample)
json.dump(samples,open("../qa-data/combined_train_replaced_uris","w",encoding="utf-8"))
