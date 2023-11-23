import json
from SPARQLWrapper import SPARQLWrapper,JSON
data= {"dataset":{"id":  "lcquad"},"questions":[]}
lcquad = json.load(open("../qa-data/LCQUAD/test.json"))

def get_answers(query_str):
    try:
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(query_str)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if not "boolean" in results:
            results["head"]["vars"]=results["head"]["vars"][0]
        return results

    except:
        print("FAILED"+query_str)
        return {'head': {'vars': 'result'}, 'results': {'bindings': []}}


for el in lcquad:
    id = el["uid"]
    print(id)
    question=el["question"]
    query=el["sparql_wikidata"]
    answers=get_answers(query)
    data["questions"].append({"id":id,"question":[{"language":"en","string":question}],"query":{"sparql":query},"answers":answers})
json.dump(data,open("../qa-data/LCQUAD/lcquad-test-quald.json","w",encoding="utf-8"))

