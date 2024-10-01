from srtk.retrieve import Retriever,KnowledgeGraphTraverser,Scorer, get_knowledge_graph
import torch
from typing import Dict,Any
import heapq
from collections import namedtuple
END_REL = 'END OF HOP'
from srtk_subgraph.wikidata_update import WikidataUpdate
# Path collects the information at each traversal step
# - prev_relations stores the relations that have been traversed
# - score stores the score of the relation path
Path = namedtuple('Path', ['prev_relations', 'score'], defaults=[(), 0])




class SRTKRetriever:
    def __init__(self,args,device):
        '''
        kg = WikidataUpdate(args.sparql_endpoint,
                                 prepend_prefixes=not args.omit_prefixes,
                                 exclude_qualifiers=not args.include_qualifiers)
        '''
        kg = get_knowledge_graph(args.knowledge_graph, args.sparql_endpoint,
                                 prepend_prefixes=not args.omit_prefixes,
                                 exclude_qualifiers=not args.include_qualifiers)

        scorer=Scorer(args.scorer_model_path, device)
        #self.retriever = RetreiverRestObj(kg, scorer, args.beam_width, args.max_depth)
        self.retriever = Retriever(kg, scorer, args.beam_width, args.max_depth)

    def retrieve(self,sample):
        triplets = self.retriever.retrieve_subgraph_triplets(sample)
        return triplets



import argparse
from srtk_subgraph import SRTKRetriever
import os
os.environ['HF_HOME'] = '/data/test_remote/cache/'
import flair
from pathlib import Path
flair.cache_root = Path("/data/test_remote/.flair")
from entity_linking import EL_model
import torch

def _add_arguments(parser):
    """Add retrieve arguments to the parser in place."""
    parser.description = '''Retrieve subgraphs with a trained model on a dataset that entities are linked.
    This command can also be used to evaluate a trained retriever when the answer entities are known.

    Provide a JSON file as input, where each JSON object must contain at least the 'question' and 'question_entities' fields.
    When ``--evaluate`` is set, the input JSON file must also contain the 'answer_entities' field.

    The output JSONL file will include an added 'triplets' field, based on the input JSONL file. This field consists of a list of triplets,
    with each triplet representing a (head, relation, tail) tuple.
    When ``--evaluate`` is set, a metric file will also be saved to the same directory as the output JSONL file.
    '''
    parser.add_argument('-i', '--input', type=str,default="../data/docleaderboard-queries.tsv", help='path to input jsonl file. it should contain at least \
                        ``question`` and ``question_entities`` fields.')
    parser.add_argument('-o', '--output',default=None, type=str, help='output file path for storing retrieved triplets.')
    parser.add_argument('-e', '--sparql-endpoint',default="https://20230607-truthy.wikidata.data.dice-research.org/sparql/", type=str, help='SPARQL endpoint for Wikidata or Freebase services.')
    parser.add_argument('-kg', '--knowledge-graph',default="wikidata", type=str, choices=('freebase', 'wikidata', 'dbpedia'),
                        help='choose the knowledge graph: currently supports ``freebase``, ``wikidata`` and ``dbpedia``.')
    parser.add_argument('-m', '--scorer-model-path',default="drt/srtk-scorer", type=str,  help='Path to the scorer model, containing \
                        both the saved model and its tokenizer in the Huggingface models format.\
                        Such a model is saved automatically when using the ``srtk_subgraph train`` command.\
                        Alternatively, provide a pre-trained model name from the Hugging Face model hub.\
                        In practice it supports any Huggingface transformers encoder model, though models that do not use [CLS] \
                        tokens may require modifications on similarity function.')
    parser.add_argument('--beam-width', type=int, default=5, help='beam width for beam search (default: 10).')
    parser.add_argument('--max-depth', type=int, default=1, help='maximum depth for beam search (default: 2).')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the retriever model. When the answer \
                        entities are known, the recall can be evluated as the number of samples that any of the \
                        answer entities are retrieved in the subgraph by the number of all samples. This equires \
                        `answer_entities` field in the input jsonl.')
    parser.add_argument('--include-qualifiers', action='store_true', help='Include qualifiers from the retrieved triplets. \
                        Qualifiers are informations represented in non-entity form, like date, count etc.\
                        This is only relevant for Wikidata.')
    parser.add_argument('--omit-prefixes', action='store_true', help='Whether to omit prefixes when passing SPARQLs \
                        to the endpoints. This can potentially save some bandwidth, but may cause errors when the \
                        prefixes are not defined in the endpoint.')

class Subgraph_linker():
    def __init__(self,device):
        parser = argparse.ArgumentParser()
        _add_arguments(parser)
        self.args = parser.parse_args()
        self.srtk_retriever=SRTKRetriever.SRTKRetriever(self.args,device)
    def get_subgraph(self,text):

        return self.srtk_retriever.retrieve(srtk_input)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
linker=Subgraph_linker(device)
input=linker.srtk_retriever.retriever
with open(linker.args.input,"r",encoding="utf-8")as input:
    processed_num=0
    for ln in input:
        if processed_num>1:
            cols=ln.split("\t")
            subgraph = linker.get_subgraph(cols[1])
            print(len(subgraph))
        processed_num+=1
'''
inv_wikidata_dict = {v: k for k, v in  linker.el_model.wikidata_dict.items()}
graph=(linker.get_subgraph("Merkel is a human born in Germany"))
for kn in graph:
   print(kn)
   if(kn[0] in inv_wikidata_dict):
       print(inv_wikidata_dict[kn[0]])
   if (kn[2] in inv_wikidata_dict):
       print(inv_wikidata_dict[kn[2]])


srtk_input={"question":text,"question_entities":[],"spans":[],"entity_names":[]}
        for el in res:
            srtk_input["question_entities"].append(el["link"])
            srtk_input["spans"].append(el["inds"])
            srtk_input["entity_names"].append(el["name"])
        print(srtk_input)
'''



