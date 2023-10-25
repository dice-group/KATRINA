#import SPARQL_parser.SPARQL_parser as SP
import rdflib
import  rdflib.plugins.sparql as sparql

from rdflib.plugins.sparql.parserutils import Expr, CompValue
from rdflib.paths import Path
from rdflib.plugins.sparql import algebra
import json

class Node_Tiny:
    def __init__(self,node_name):
        self.node_value=node_name
        self.children=[]

class Leave:
    def __init__(self,node_name,type):
        self.node_value=node_name
        self.type=type
class Edge:
    def __init__(self,src,dest,label=None):
        self.src=src
        self.dest=dest
        self.label=label
from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
import json

dp=Dataprocessor_test(T5Tokenizer.from_pretrained("out"),"")
tokenizer = T5Tokenizer.from_pretrained("out")
model = T5ForConditionalGeneration.from_pretrained("out-wikidata-labels/checkpoint-150000")

def graph_to_dict(graph,nodes,edges,leaves,parent_id=None):

    for node in graph.children:
        if node.__class__==Node_Tiny and len(node.children)>0:
            n={"value":node.node_value,"id":len(nodes)}
            nodes.append(n)
            if parent_id is not None:
                edges.append({"src":n["id"],"dest":parent_id})
            graph_to_dict(node,nodes,edges,leaves,n["id"])
        elif node.__class__==Leave:
            if not str(node.node_value) in leaves:
                n={"value":str(node.node_value),"type":node.type,"id":len(nodes)}
                leaves.update({n["value"]:n["id"]})
                nodes.append(n)
                edges.append({"src":n["id"],"dest":parent_id})
            else:
                edges.append({"src":leaves[str(node.node_value)],"dest":parent_id})
    '''
    node2Id={}
    max_node_id=0
    node_list=[]
    edge_list=[]
    for node in nodes:
        node2Id.update({str(node):max_node_id})
        node_list.append({"id": max_node_id,"node_type":node.node_type,"node_value":node.node_value})
        max_node_id+=1
    for edge in edges:
        edge_list.append({"src":node2Id[str(edge.src)],"dest":node2Id[str(edge.dest)],"label":str(edge.label)})
    '''
    return {"nodes":nodes,"edges":edges}


def process_query(query_str):
    query_triples=[]
    query_patterns=[]
    parsed_query=sparql.parser.parseQuery(query_str)
    entities=set()
    relations=set()
    en = algebra.translateQuery(parsed_query)
    #en.algebra["PV"]=list(en.algebra["_vars"])
    #en.algebra.name="SelectQuery"
    #endpoint=CompValue(name="DataClause")
    #endpoint["default"]="http://20230607-truthy.wikidata.data.dice-research.org/"

    #en.algebra["datasetClause"]=endpoint
    g = rdflib.Graph()

    res=g.query(en)
    if len(res)>0:
        triples = gentree(en)
        for t in triples:
            query_pattern=[str(t[0]),str(t[1]),str(t[2])]
            if isinstance(t[0],rdflib.Variable):
                query_pattern[0]="_var"
            if isinstance(t[2],rdflib.Variable):
                query_pattern[2]="_var"
            query_patterns.append(query_pattern)
        vars=res.vars
        var_map={}
        for i in range(0,len(vars)):
            var_map[str(vars[i])]=i
        for r in res:
            for triple in triples:
                if isinstance(triple[0],rdflib.Variable):
                    s=str(r[var_map[str(triple[0])]])
                else:
                    s=str(triple[0])
                p=str(triple[1])
                if isinstance(triple[2],rdflib.Variable):
                    o=str(r[var_map[str(triple[2])]])
                else:
                    o=str(triple[2])
                query_triples.append([s,p,o])
            #print(r)
        #print(res)

    return query_triples,query_patterns

def gentree(q):
    triples=[]
    def pp(p, parent):
        # if isinstance(p, list):
        #     print "[ "
        #     for x in p: pp(x,ind)
        #     print "%s ]"%ind
        #     return
        from rdflib.plugins.sparql.parser import TriplesBlock
        if p is None:
            return
        if "triples" in p:
            triples.extend(p["triples"])
        if isinstance(p, (rdflib.URIRef,rdflib.Variable,rdflib.Literal,str,int)):

            n=Leave(p,str(p.__class__))
            parent.children.append(n)
            #print(p)
            return
        if isinstance(p,list):
            n=Node_Tiny("list")
        elif isinstance(p,set):
            n=Node_Tiny("set")
        elif isinstance(p,tuple):
            n=Node_Tiny("tuple")
        elif isinstance(p,rdflib.paths.MulPath):
            n=Node_Tiny("MulPath"+p.mod)
            p=[p.path]
        elif isinstance(p,rdflib.paths.AlternativePath):
            n=Node_Tiny("AlternativePath")
            p=p.args
        elif isinstance(p,rdflib.paths.SequencePath):
            n=Node_Tiny("SequencePath")
            p=p.args
        elif isinstance(p,rdflib.paths.InvPath):
            n=Node_Tiny("InvPath")
            p=[p.arg]
        else:
            if hasattr(p, 'name'):
                n=Node_Tiny(p.name)
            else:
                n=Node_Tiny("dict")
        parent.children.append(n)
        for k in p:
            '''
            print(
                "%s%s ="
                % (
                    ind,
                    k,
                ),
                end=" ",
            )'''
            if isinstance(p, (list,set,tuple)):
                pp(k,n)
            else:
                pp(p[k], n)
        #print("%s)" % ind)

    #try:
    n=Node_Tiny("root")
    pp(q.algebra,n)
    return triples



prefixes="""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wds: <http://www.wikidata.org/entity/statement/>
PREFIX wdv: <http://www.wikidata.org/value/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX bd: <http://www.bigdata.com/rdf#>
"""
'''
query="SELECT (COUNT(?obj) AS ?value ) { wd:Q190679 wdt:P1080 ?obj }"

query="Select * where " + query[query.find("{"):len(query)]
query=query.replace("{","{ SERVICE <https://20230607-truthy.wikidata.data.dice-research.org/sparql> {",1)
lb=query.rfind("}")
query=query[0:lb]+"}"+query[lb:len(query)]
"SERVICE{ < https: // 20230607 - truthy.wikidata.data.dice - research .org / sparql >"
process_query(prefixes+query)
'''
'''
data=json.load(open("../qa-data/LCQUAD/test-with-resources-labels.json","r",encoding="utf-8"))
for question in data:
    try:
        print(question["uid"])
        query=question["sparql_wikidata"]
        query="Select * where " + query[query.find("{"):len(query)]
        query=query.replace("{","{ SERVICE <https://20230607-truthy.wikidata.data.dice-research.org/sparql> {",1)
        lb=query.rfind("}")
        query=query[0:lb]+"}"+query[lb:len(query)]

        query_triples=process_query(prefixes+query)
        question["triples"]=query_triples
        #entities,relations=process_query(prefixes+query)
        #question["entities"]=list(entities)
        #question["relations"]=list(relations)
    except:
        print(question["sparql_wikidata"]+" failed")
json.dump(data,open("../qa-data/LCQUAD/test-with-triples.json","w",encoding="utf-8"))
'''
data=json.load(open("../qa-data/LCQUAD/train-with-resources-labels.json","r",encoding="utf-8"))
for question in data:
    try:
        print(question["uid"])
        query=question["sparql_wikidata"]
        query="Select * where " + query[query.find("{"):len(query)]
        query=query.replace("{","{ SERVICE <https://20230607-truthy.wikidata.data.dice-research.org/sparql> {",1)
        lb=query.rfind("}")
        query=query[0:lb]+"}"+query[lb:len(query)]

        query_triples,query_patterns=process_query(prefixes+query)
        question["triples"]=query_triples
        question["patterns"] = query_patterns
        #entities,relations=process_query(prefixes+query)
        #question["entities"]=list(entities)
        #question["relations"]=list(relations)
    except:
        print(question["sparql_wikidata"]+" failed")
json.dump(data,open("../qa-data/LCQUAD/LCQUAD_triples_patterns.json","w",encoding="utf-8"))


'''
with open("../qald-9-train-multilingual.json","r",encoding="utf-8")as queries:
    queries=json.load(queries)
    for el in queries["questions"]:
        process_query(el["query"]["sparql"])
'''