from SPARQLWrapper import SPARQLWrapper, JSON
import pickle


def queryRelations():
    relations = {}
    found = True
    offset = 0
    while found:
        print(offset)
        sparql = SPARQLWrapper(
            "https://query.wikidata.org/bigdata/namespace/wdq/sparql",
            agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')

        sparql.setQuery('''
            SELECT ?property ?propertyLabel ?instance WHERE {

                ?property a wikibase:Property .
                SERVICE wikibase:label {
                bd:serviceParam wikibase:language "en" .
                }
            }

            LIMIT 1000 OFFSET ''' + str(offset))
        sparql.setReturnFormat(JSON)
        offset = offset + 1000
        res = sparql.queryAndConvert()
        if len(res['results']["bindings"]) < 1000:
            found = False
        for el in res['results']["bindings"]:
            relations.update({el["property"]["value"].replace("http://www.wikidata.org/entity/",""): el["propertyLabel"]["value"]})
    return relations

def querymissingRelation(rel):
    print(rel)
    sparql = SPARQLWrapper(
        "https://query.wikidata.org/bigdata/namespace/wdq/sparql",
            agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')

    sparql.setQuery('''
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX wd: <http://www.wikidata.org/entity/> 
        SELECT  *
        WHERE {
            wd:'''+rel+''' rdfs:label ?label .
            FILTER (langMatches( lang(?label), "EN" ) )
        } ''')
    sparql.setReturnFormat(JSON)
    res = sparql.queryAndConvert()
    if len(res['results']["bindings"])>0:
        return {rel.replace("http://www.wikidata.org/entity/",""): res['results']["bindings"][0]["label"]["value"]}

    else:
        return None



relations=pickle.load(open("../precomputed/relation_labels.sav","rb"))
print(len(relations))
missing=pickle.load(open("missing_rel","rb"))
for rel in missing:
    update=querymissingRelation(rel)
    print(update)
    if update is not None:
        relations.update(update)
pickle.dump(relations,open("../precomputed/relation_labels.sav","wb"))


