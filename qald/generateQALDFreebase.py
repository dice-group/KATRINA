import json
from SPARQLWrapper import SPARQLWrapper,JSON
data= {"dataset":{"id":  "lcquad"},"questions":[]}
dataset = json.load(open("../qa-data/GrailQA_v1.0/grailqa_v1.0_dev.json"))

def get_answers(query_str):
    try:
        sparql = SPARQLWrapper("https://freebase.data.dice-research.org/sparql")
        sparql.setQuery(query_str)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if not "boolean" in results:
            results["head"]["vars"]=results["head"]["vars"][0]
        return results
        '''
        if "boolean"in results:
            return results["boolean"]
        print(results)
        res=[]
        '''
    except:
        print("FAILED"+query_str)
        return {'head': {'vars': 'result'}, 'results': {'bindings': []}}
    return {"results"}


for el in dataset:
    id = el["qid"]
    print(id)
    question=el["question"]
    query=el["sparql_query"]
    answers=get_answers(query)
    data["questions"].append({"id":id,"question":[{"language":"en","string":question}],"query":{"sparql":query},"answers":answers})
json.dump(data,open("../qa-data/LCQUAD/lcquad-test-quald.json","w",encoding="utf-8"))

