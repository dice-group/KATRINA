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

data=json.load(open("../qa-data/LCQUAD/train.json","r",encoding="utf-8"))
for question in data:
    try:
        process_query(prefixes+question["sparql_wikidata"])
        entities,relations=process_query(prefixes+question["sparql_wikidata"])
        question["entities"]=list(entities)
        question["relations"]=list(relations)
    except:
        print(question["sparql_wikidata"]+" failed")
json.dump(data,open("../qa-data/LCQUAD/train-with-resources.json","w",encoding="utf-8"))


'''
with open("../qald-9-train-multilingual.json","r",encoding="utf-8")as queries:
    queries=json.load(queries)
    for el in queries["questions"]:
        process_query(el["query"]["sparql"])
'''