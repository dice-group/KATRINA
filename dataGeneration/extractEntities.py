#import SPARQL_parser.SPARQL_parser as SP
import rdflib
import  rdflib.plugins.sparql as sparql

from rdflib.plugins.sparql.parserutils import Expr, CompValue
from rdflib.paths import Path
from rdflib.plugins.sparql import algebra
import json
import pickle
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
    parsed_query=sparql.parser.parseQuery(query_str)
    entities=set()
    relations=set()
    en = algebra.translateQuery(parsed_query)
    graph=gentree(en)
    #process_path(parsed_query.algebra["p"],nodes,edges)
    #triples=get_path(parsed_query.algebra)
    graph=graph_to_dict(graph,[],[],{},None)
     #out={"query_str":query_str,"graph":graph}
    for node in graph["nodes"]:
        if "type"in node and node["type"]=="<class 'rdflib.term.URIRef'>":
            if "P" in node["value"]:
                relations.add(node["value"])
            else:
                entities.add(node["value"])
    return entities,relations




def gentree(q):

    def pp(p, parent):
        # if isinstance(p, list):
        #     print "[ "
        #     for x in p: pp(x,ind)
        #     print "%s ]"%ind
        #     return
        from rdflib.plugins.sparql.parser import TriplesBlock
        if p is None:
            return
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
    return n

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
data=json.load(open("../qa-data/LCQUAD/test.json","r",encoding="utf-8"))
for question in data:
    try:
        process_query(prefixes+question["sparql_wikidata"])
        entities,relations=process_query(prefixes+question["sparql_wikidata"])
        question["entities"]=list(entities)
        question["relations"]=list(relations)
    except:
        print(question["sparql_wikidata"]+" failed")
json.dump(data,open("../qa-data/LCQUAD/test-with-resources.json","w",encoding="utf-8"))
'''
labels=pickle.load(open("../precomputed/labels_wikipedia.sav","rb"))
labels_relations=pickle.load(open("../precomputed/relation_labels.sav","rb"))
processed=[]
with open("../qa-data/QALD/qald_10.json", "r", encoding="utf-8")as data:
    queries=json.load(data)["questions"]
    for el in queries:
        question_str=""
        for q in el["question"]:
            if q["language"]=="en":
                question_str=q["string"]
            query=el["query"]["sparql"]

        print(question_str)
        print()
        entities,relations=process_query(query)
        query=query.replace("PREFIX bd: <http://www.bigdata.com/rdf#> PREFIX dct: <http://purl.org/dc/terms/> PREFIX geo: <http://www.opengis.net/ont/geosparql#> PREFIX p: <http://www.wikidata.org/prop/> PREFIX pq: <http://www.wikidata.org/prop/qualifier/> PREFIX ps: <http://www.wikidata.org/prop/statement/> PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wds: <http://www.wikidata.org/entity/statement/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wdv: <http://www.wikidata.org/value/> PREFIX wikibase: <http://wikiba.se/ontology#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> ","")
        print(entities)
        print(relations)
        entities=list(entities)
        ent_list=[]
        for en in entities:
            if en.replace("http://www.wikidata.org/entity/","") in labels:
                ent_list.append({"uri": en, "label": labels[en.replace("http://www.wikidata.org/entity/", "")]})

        relation_list=[]
        for en in relations:
            if en.replace("http://www.wikidata.org/prop/direct/","") in labels_relations:
                relation_list.append({"uri": en, "label": labels_relations[en.replace("http://www.wikidata.org/prop/direct/","")]})
        processed.append({"uid":el["id"],"question":question_str, "sparql_wikidata":query, "entities":ent_list,"relations":relation_list})
json.dump(processed, open("../qa-data/QALD/qald_10.json", "w", encoding="utf-8"), indent=4)

