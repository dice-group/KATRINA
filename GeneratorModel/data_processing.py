import json
import torch
from random import shuffle
from torch.utils.data import Dataset
import  rdflib.plugins.sparql as sparql

from rdflib.plugins.sparql import algebra
import pickle
from tqdm import tqdm

class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)

class Dataprocessor():
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args=args

    def read_ds_to_list(self,path_to_ds):
        return []
    def process_training_ds(self,data):
        samples = self.read_ds_to_list(data)
        shuffle(samples)
        dataset=ListDataset([])
        print("tokenize data")
        for sample in tqdm(samples):
            dataset.examples.append(self.process_sample(sample["input"],sample["label"]))
        return dataset

    def process_sample(self,input,label=None):
        pass
class Dataprocessor_test(Dataprocessor):

    def process_sample(self,input,label=None):
        encoding = self.tokenizer(text=input,text_target=label, return_tensors="pt",
                                  )
        return encoding

class Dataprocessor_KBQA_basic(Dataprocessor):

    def process_sample(self,input,label=None):
        encoding = self.tokenizer.prepare_seq2seq_batch(src_texts=[input],text_target=[label], return_tensors="pt",
                        max_length=self.args["max_target_length"],
                        max_target_length=self.args["max_target_length"]

                                      )
        input=encoding.data["input_ids"]
        padded_input_tensor = self.tokenizer.pad_token_id * torch.ones(
            (input.shape[0], self.args["max_input_length"]), dtype=input.dtype, device=input.device
        )
        padded_input_tensor[:, : input.shape[-1]] = input
        encoding.data["input_ids"]=torch.flatten(padded_input_tensor)

        attention_mask = encoding.data["attention_mask"]
        padded_attention_mask = self.tokenizer.pad_token_id * torch.ones(
            (attention_mask.shape[0], self.args["max_input_length"]), dtype=attention_mask.dtype, device=attention_mask.device
        )
        padded_attention_mask[:, : attention_mask.shape[-1]] = attention_mask
        encoding.data["attention_mask"] = torch.flatten(padded_attention_mask)

        target = encoding.data["labels"]
        padded_target_tensor = self.tokenizer.pad_token_id * torch.ones(
            (target.shape[0], self.args["max_target_length"]), dtype=target.dtype, device=target.device
        )
        padded_target_tensor[:, : target.shape[-1]] = target
        encoding.data["labels"]=torch.flatten(padded_target_tensor)

        #out["decoder_input_ids"]=T5PreTrainedModel._shift_right(input_ids=out["labels"])


        return encoding.data



class Dataprocessor_Combined_simple(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, path_to_ds):
        prefixes = """
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
        lcquad_data = json.load(open(path_to_ds+"/lcquad.json"))
        samples = []
        for question in tqdm(lcquad_data):
            # print(question)
            if "entities" in question and "relations" in question and question["question"] is not None:
                question_str = question["question"]
                query = question["sparql_wikidata"]
                parsed_query = sparql.parser.parseQuery(prefixes+query)
                en = algebra.translateQuery(parsed_query)
                '''
                for ent in question["entities"]:
                    key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                    query = query.replace(key, ent["label"])
                for rel in question["relations"]:
                    key = rel["uri"].replace("http://www.wikidata.org/prop/direct/", "")
                    query = query.replace(key, rel["label"])
                '''
                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")
                sample = {"input": question_str+"[SEP]target_wikidata", "label": query}
                samples.append(sample)

        grail_qa = json.load(open(path_to_ds+"/grail.json"))

        for el in tqdm(grail_qa):
            question_str = el["question"]


            def preprocessfreebasequery(query_str):
                processed_query=""
                #print(query_str)
                query_split=query_str.split("\n")
                values={}
                for el in query_split:
                    if "VALUES" in el:
                        key=el[len("VALUES "):el.index("{")-1]
                        value=el[el.index("{")+2:el.index("}")+-1]
                        values[key]=value
                    elif not "FILTER" in el:
                        processed_query+=el
                for k in values.keys():
                    processed_query = processed_query.replace(k,values[k])
                return processed_query

            query = el["sparql_query"]
            nodes = el["graph_query"]["nodes"]
            edges = el["graph_query"]["edges"]
            query=preprocessfreebasequery(query)
            parsed_query = sparql.parser.parseQuery(query)

            en = algebra.translateQuery(parsed_query)

            '''
            for e in edges:
                query = query.replace(e["relation"], e["friendly_name"])
            for n in nodes:
                query = query.replace(n["id"], n["friendly_name"])
            '''
            query = query.replace("\n", "")
            query = query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>", "")
            query = query.replace(
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>", "")
            query = query.replace(
                "PREFIX : <http://rdf.freebase.com/ns/> ", "")

            res_vars = en.algebra["PV"]
            vars = en.algebra["_vars"]
            it = 0

            for el in res_vars:
                query = query.replace("?" + el, "_result_" + str(it))
                it += 1
            it = 0
            for el in vars:
                query = query.replace("?" + el, "_var_" + str(it))
                it += 1
            query = query.replace("{", "_cbo_")
            query = query.replace("}", "_cbc_")
            sample = {"input": question_str+"[SEP]target_freebase", "label": query}

            samples.append(sample)

        return samples

class Dataprocessor_Combined_entities(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, path_to_ds):
        prefixes = """
                PREFIX bd: <http://www.bigdata.com/rdf#> 
                PREFIX dct: <http://purl.org/dc/terms/> 
                PREFIX geo: <http://www.opengis.net/ont/geosparql#> 
                PREFIX p: <http://www.wikidata.org/prop/> 
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/> 
                PREFIX ps: <http://www.wikidata.org/prop/statement/> 
                PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wds: <http://www.wikidata.org/entity/statement/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX wdv: <http://www.wikidata.org/value/> 
                PREFIX wikibase: <http://wikiba.se/ontology#> 
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
                """
        lcquad_data = json.load(open(path_to_ds+"/lcquad.json"))
        samples = []
        for question in tqdm(lcquad_data):
            # print(question)
            if "entities" in question  and question["question"] is not None:
                question_str = question["question"]+"[SEP] "
                entities=question["entities"]
                for ent in entities:
                    question_str+=ent["label"]+" : "+ent["uri"].replace("http://www.wikidata.org/entity/","")+" , "
                query = question["sparql_wikidata"]
                parsed_query = sparql.parser.parseQuery(prefixes+query)
                en = algebra.translateQuery(parsed_query)
                '''
                for ent in question["entities"]:
                    key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                    query = query.replace(key, ent["label"])
                for rel in question["relations"]:
                    key = rel["uri"].replace("http://www.wikidata.org/prop/direct/", "")
                    query = query.replace(key, rel["label"])
                '''
                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")
                sample = {"input": question_str+"[SEP]target_wikidata", "label": query}
                samples.append(sample)

        grail_qa = json.load(open(path_to_ds+"/grail.json"))
        num_samples_wk=len(samples)
        for el in tqdm(grail_qa):
            question_str = el["question"]+"[SEP] "
            nodes=el["graph_query"]["nodes"]
            for n in nodes:
                if not n["node_type"] == "literal":
                    question_str += n["friendly_name"] + " : " + n["id"]+ " , "



            def preprocessfreebasequery(query_str):
                processed_query=""
                #print(query_str)
                query_split=query_str.split("\n")
                values={}
                for el in query_split:
                    if "VALUES" in el:
                        key=el[len("VALUES "):el.index("{")-1]
                        value=el[el.index("{")+2:el.index("}")+-1]
                        values[key]=value
                    elif not "FILTER" in el:
                        processed_query+=el
                for k in values.keys():
                    processed_query = processed_query.replace(k,values[k])
                return processed_query

            query = el["sparql_query"]
            nodes = el["graph_query"]["nodes"]
            edges = el["graph_query"]["edges"]
            #query=preprocessfreebasequery(query)
            parsed_query = sparql.parser.parseQuery(query)

            en = algebra.translateQuery(parsed_query)

            '''
            for e in edges:
                query = query.replace(e["relation"], e["friendly_name"])
            for n in nodes:
                query = query.replace(n["id"], n["friendly_name"])
            '''
            query = query.replace("\n", "")
            query = query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>", "")
            query = query.replace(
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>", "")
            query = query.replace(
                "PREFIX : <http://rdf.freebase.com/ns/> ", "")

            res_vars = en.algebra["PV"]
            vars = en.algebra["_vars"]
            it = 0

            for el in res_vars:
                query = query.replace("?" + el, "_result_" + str(it))
                it += 1
            it = 0
            for el in vars:
                query = query.replace("?" + el, "_var_" + str(it))
                it += 1
            query = query.replace("{", "_cbo_")
            query = query.replace("}", "_cbc_")
            sample = {"input": question_str+"[SEP]target_freebase", "label": query}

            samples.append(sample)
        return samples

class LC_Quad_Processor(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, filename, add_entities=True, add_relations=True):
        prefixes = """
                PREFIX bd: <http://www.bigdata.com/rdf#> 
                PREFIX dct: <http://purl.org/dc/terms/> 
                PREFIX geo: <http://www.opengis.net/ont/geosparql#> 
                PREFIX p: <http://www.wikidata.org/prop/> 
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/> 
                PREFIX ps: <http://www.wikidata.org/prop/statement/> 
                PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wds: <http://www.wikidata.org/entity/statement/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX wdv: <http://www.wikidata.org/value/> 
                PREFIX wikibase: <http://wikiba.se/ontology#> 
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
                """
        lcquad_data = json.load(open(filename))
        samples = []
        for question in tqdm(lcquad_data):
            # print(question)
            if "entities" in question  and question["question"] is not None:
                question_str = question["question"]+"[SEP] "
                entities=question["entities"]
                if add_entities:
                    for ent in entities:
                        question_str+=ent["label"]+" : "+ent["uri"].replace("http://www.wikidata.org/entity/","")+" , "
                    question_str=question_str+"[SEP]"
                if add_relations:
                    relations = question["relations"]
                    for rel in relations:
                        question_str+=rel["label"]+ " : "+ rel["uri"].replace("http://www.wikidata.org/prop/direct/","") + " , "
                query = question["sparql_wikidata"]
                parsed_query = sparql.parser.parseQuery(prefixes+query)
                en = algebra.translateQuery(parsed_query)
                '''
                for ent in question["entities"]:
                    key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                    query = query.replace(key, ent["label"])
                for rel in question["relations"]:
                    key = rel["uri"].replace("http://www.wikidata.org/prop/direct/", "")
                    query = query.replace(key, rel["label"])
                '''
                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")
                sample = {"input": question_str+"[SEP]target_wikidata", "label": query}
                samples.append(sample)
        return samples


class Grail_QA_processor(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, filename,path_to_schema_links=None,max_entities=0, max_relations=0):
        samples=[]
        grail_qa = json.load(open(filename))
        types = pickle.load(open("../precomputed/Generator/type_dict_freebase.pkl", "rb"))
        relation_labels = pickle.load(open("../precomputed/Generator/relation_labels.pkl", "rb"))
        schema = {}
        if path_to_schema_links is not None:
            with(open(path_to_schema_links, "r")) as file:
                schema = {}
                for ln in file:
                    el = json.loads(ln)
                    schema[el["qid"]] = el
        for el in tqdm(grail_qa[:500]):
            question_str = el["question"]
            if max_entities >0:
                question_str +="[SEP] entities: "

                nodes = el["graph_query"]["nodes"]
                classes = set()
                nodes_to_add = []
                for n in nodes:
                    if not n["node_type"] == "literal":
                        if n["node_type"] == "class":
                            classes.add(n["id"])
                        nodes_to_add.append(n)
                while len(classes) < max_entities:
                    if el["qid"] in schema:
                        for cl in schema[el["qid"]]["classes"]:
                            if not cl in classes:
                                classes.add(cl)
                                nodes_to_add.append({"id": cl, "friendly_name": types[cl]})
                            if len(classes) == max_entities:
                                break
                #                    question_str += n["friendly_name"] + " : " + n["id"]+ " , "
                shuffle(nodes_to_add)
                for n in nodes_to_add:
                    question_str += n["friendly_name"] + " : " + n["id"] + " , "
            if max_entities > 0:
                question_str += "[SEP] relations: "
                edges = el["graph_query"]["edges"]
                relations = set()
                relations_to_add = []
                for e in edges:
                    relations.add(e["relation"])
                    relations_to_add.append(e)
                while len(relations) < max_relations:
                    if el["qid"] in schema:
                        for rel in schema[el["qid"]]["relations"]:
                            if not rel in relations:
                                relations.add(rel)
                                relations_to_add.append({"relation": rel, "friendly_name": relation_labels[rel]})
                            if len(relations) == max_relations:
                                break
                shuffle(relations_to_add)
                question_str = question_str + "relations: "
                for e in relations_to_add:
                    question_str += e["friendly_name"] + " : " + e["relation"] + " , "
                question_str += "[SEP]"

            query = el["sparql_query"]
            # query = el["s_expression"]
            #nodes = el["graph_query"]["nodes"]
            #edges = el["graph_query"]["edges"]
            # query=preprocessfreebasequery(query)

            parsed_query = sparql.parser.parseQuery(query)

            en = algebra.translateQuery(parsed_query)

            query = query.replace("\n", "")
            query = query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>", "")
            query = query.replace(
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>", "")
            query = query.replace(
                "PREFIX : <http://rdf.freebase.com/ns/> ", "")

            res_vars = en.algebra["PV"]
            vars = en.algebra["_vars"]
            it = 0

            for el in res_vars:
                query = query.replace("?" + el, "_result_" + str(it))
                it += 1
            it = 0
            for el in vars:
                query = query.replace("?" + el, "_var_" + str(it))
                it += 1
            query = query.replace("{", "_cbo_")
            query = query.replace("}", "_cbc_")

            sample = {"input": question_str + "[SEP]target_freebase", "label": query}

            samples.append(sample)
        return samples

class Combined_Processor(Dataprocessor_KBQA_basic):
    def __init__(self,tokenizer,args):
        self.lc_quad=LC_Quad_Processor(tokenizer,args)
        self.grail = Grail_QA_processor(tokenizer,args)
        self.tokenizer=tokenizer
        self.args=args

    def read_ds_to_list(self,folder):
        if self.args["use_wikidata"]:
            samples=self.lc_quad.read_ds_to_list(folder+"/"+self.args["lc_quad_file"], add_entities=True, add_relations=True)
        if self.args["use_freebase"]:
            if self.args["add_entities"]:
                samples_grail=self.grail.read_ds_to_list(folder+"/"+self.args["grail_qa_file"],
                                                         path_to_schema_links=self.args["schema_links"],max_entities=5, max_relations=5)
            else:
                samples_grail=self.grail.read_ds_to_list(folder+"/"+self.args["grail_qa_file"])
            if "qald" in folder:
                samples_grail=samples_grail[:len(samples)]
        samples.extend(samples_grail)
        return samples



class Dataprocessor_Combined_entities_relations(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, path_to_ds):
        prefixes = """
                PREFIX bd: <http://www.bigdata.com/rdf#> 
                PREFIX dct: <http://purl.org/dc/terms/> 
                PREFIX geo: <http://www.opengis.net/ont/geosparql#> 
                PREFIX p: <http://www.wikidata.org/prop/> 
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/> 
                PREFIX ps: <http://www.wikidata.org/prop/statement/> 
                PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wds: <http://www.wikidata.org/entity/statement/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX wdv: <http://www.wikidata.org/value/> 
                PREFIX wikibase: <http://wikiba.se/ontology#> 
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
                """
        lcquad_data = json.load(open(path_to_ds+"/lcquad.json"))
        samples = []
        for question in tqdm(lcquad_data):
            # print(question)
            if "entities" in question  and question["question"] is not None:
                question_str = question["question"]+"[SEP] "
                entities=question["entities"]
                for ent in entities:
                    question_str+=ent["label"]+" : "+ent["uri"].replace("http://www.wikidata.org/entity/","")+" , "
                question_str=question_str+"[SEP]"
                relations = question["relations"]
                for rel in relations:
                    question_str+=rel["label"]+ " : "+ rel["uri"].replace("http://www.wikidata.org/prop/direct/","") + " , "
                query = question["sparql_wikidata"]
                parsed_query = sparql.parser.parseQuery(prefixes+query)
                en = algebra.translateQuery(parsed_query)
                '''
                for ent in question["entities"]:
                    key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                    query = query.replace(key, ent["label"])
                for rel in question["relations"]:
                    key = rel["uri"].replace("http://www.wikidata.org/prop/direct/", "")
                    query = query.replace(key, rel["label"])
                '''
                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")
                sample = {"input": question_str+"[SEP]target_wikidata", "label": query}
                samples.append(sample)

        grail_qa = json.load(open(path_to_ds+"/grail.json"))
        for el in tqdm(grail_qa):
            question_str = el["question"]+"[SEP] "
            nodes=el["graph_query"]["nodes"]
            for n in nodes:
                if not n["node_type"] == "literal":
                    question_str += n["friendly_name"] + " : " + n["id"]+ " , "
            edges = el["graph_query"]["edges"]
            question_str=question_str+"[SEP]"
            for e in edges:
                question_str += e["friendly_name"] + " : " + e["relation"] + " , "
            def preprocessfreebasequery(query_str):
                processed_query=""
                #print(query_str)
                query_split=query_str.split("\n")
                values={}
                for el in query_split:
                    if "VALUES" in el:
                        key=el[len("VALUES "):el.index("{")-1]
                        value=el[el.index("{")+2:el.index("}")+-1]
                        values[key]=value
                    elif not "FILTER" in el:
                        processed_query+=el
                for k in values.keys():
                    processed_query = processed_query.replace(k,values[k])
                return processed_query

            query = el["sparql_query"]
            nodes = el["graph_query"]["nodes"]
            edges = el["graph_query"]["edges"]
            #query=preprocessfreebasequery(query)
            parsed_query = sparql.parser.parseQuery(query)

            en = algebra.translateQuery(parsed_query)

            '''
            for e in edges:
                query = query.replace(e["relation"], e["friendly_name"])
            for n in nodes:
                query = query.replace(n["id"], n["friendly_name"])
            '''
            query = query.replace("\n", "")
            query = query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>", "")
            query = query.replace(
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>", "")
            query = query.replace(
                "PREFIX : <http://rdf.freebase.com/ns/> ", "")

            res_vars = en.algebra["PV"]
            vars = en.algebra["_vars"]
            it = 0

            for el in res_vars:
                query = query.replace("?" + el, "_result_" + str(it))
                it += 1
            it = 0
            for el in vars:
                query = query.replace("?" + el, "_var_" + str(it))
                it += 1
            query = query.replace("{", "_cbo_")
            query = query.replace("}", "_cbc_")
            sample = {"input": question_str+"[SEP]target_freebase", "label": query}

            samples.append(sample)
        return samples


class Dataprocessor_Combined_predicted_resources(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, path_to_ds):
        prefixes = """
                PREFIX bd: <http://www.bigdata.com/rdf#> 
                PREFIX dct: <http://purl.org/dc/terms/> 
                PREFIX geo: <http://www.opengis.net/ont/geosparql#> 
                PREFIX p: <http://www.wikidata.org/prop/> 
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/> 
                PREFIX ps: <http://www.wikidata.org/prop/statement/> 
                PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wds: <http://www.wikidata.org/entity/statement/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX wdv: <http://www.wikidata.org/value/> 
                PREFIX wikibase: <http://wikiba.se/ontology#> 
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
                """
        lcquad_data = json.load(open(path_to_ds+"/lcquad_el.json"))
        samples = []
        if self.args["use_wikidata"]:
            for question in tqdm(lcquad_data)[:20]:
                # print(question)
                if "entities" in question  and question["question"] is not None:
                    question_str = question["question"]+"[SEP] entities: "
                    entities=question["entities"]
                    for ent in entities:
                        question_str+=ent["label"]+" : "+ent["uri"].replace("http://www.wikidata.org/entity/","")+" , "
                    question_str=question_str+"relations: "
                    relations = question["relations"]
                    for rel in relations:
                        question_str+=rel["label"]+ " : "+ rel["uri"].replace("http://www.wikidata.org/prop/direct/","") + " , "
                    query = question["sparql_wikidata"]
                    try:
                        parsed_query = sparql.parser.parseQuery(prefixes+query)


                        en = algebra.translateQuery(parsed_query)
                        '''
                
                        for ent in question["entities"]:
                            key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                            query = query.replace(key, ent["label"])
                        for rel in question["relations"]:
                            key = rel["uri"].replace("http://www.wikidata.org/prop/direct/", "")
                            query = query.replace(key, rel["label"])
                        '''
                        res_vars = en.algebra["PV"]
                        vars = en.algebra["_vars"]
                        it = 0

                        for el in res_vars:
                            query = query.replace("?" + el, "_result_" + str(it))
                            it += 1
                        it = 0
                        for el in vars:
                            query = query.replace("?" + el, "_var_" + str(it))
                            it += 1
                        query = query.replace("{", "_cbo_")
                        query = query.replace("}", "_cbc_")
                        sample = {"input": question_str+"[SEP]target_wikidata", "label": query}
                        samples.append(sample)
                    except:
                        print("failed: "+query)
        if self.args["use_freebase"]:
            grail_qa = json.load(open(path_to_ds+"/grail.json"))
            types=pickle.load(open("../precomputed/Generator/type_dict_freebase.pkl","rb"))
            relation_labels = pickle.load(open("../precomputed/Generator/relation_labels.pkl", "rb"))
            with(open(path_to_ds+"/dense_retrieval_grailqa.jsonl", "r")) as file:
                schema = {}
                for ln in file:
                    el = json.loads(ln)
                    schema[el["qid"]] = el
            for el in tqdm(grail_qa[:20]):
                question_str = el["question"]+"[SEP] entities: "
                nodes=el["graph_query"]["nodes"]
                classes=set()
                nodes_to_add=[]
                for n in nodes:
                    if not n["node_type"] == "literal":
                        if n["node_type"]=="class":
                            classes.add(n["id"])
                        nodes_to_add.append(n)
                while len(classes)<5:
                    for cl in schema[el["qid"]]["classes"]:
                        if not cl in classes:
                            classes.add(cl)
                            nodes_to_add.append({"id":cl,"friendly_name":types[cl]})
                        if len(classes)==5:
                            break
    #                    question_str += n["friendly_name"] + " : " + n["id"]+ " , "
                shuffle(nodes_to_add)
                for n in nodes_to_add:
                    question_str += n["friendly_name"] + " : " + n["id"]+ " , "
                edges = el["graph_query"]["edges"]
                relations=set()
                relations_to_add=[]
                for e in edges:
                    relations.add(e["relation"])
                    relations_to_add.append(e)
                while len(relations)<5:
                    for rel in schema[el["qid"]]["relations"]:
                        if not rel in relations:
                            relations.add(rel)
                            relations_to_add.append({"relation":rel,"friendly_name":relation_labels[rel]})
                        if len(relations)==5:
                            break
                shuffle(relations_to_add)
                question_str = question_str + "relations: "
                for e in relations_to_add:
                    question_str += e["friendly_name"] + " : " + e["relation"] + " , "

                query = el["sparql_query"]
                #query = el["s_expression"]
                nodes = el["graph_query"]["nodes"]
                edges = el["graph_query"]["edges"]
                #query=preprocessfreebasequery(query)

                parsed_query = sparql.parser.parseQuery(query)

                en = algebra.translateQuery(parsed_query)

                
                query = query.replace("\n", "")
                query = query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>", "")
                query = query.replace(
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>", "")
                query = query.replace(
                    "PREFIX : <http://rdf.freebase.com/ns/> ", "")

                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")

                sample = {"input": question_str+"[SEP]target_freebase", "label": query}

                samples.append(sample)
        return samples


class Dataprocessor_Combined_QALD(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self, path_to_ds):
        prefixes = """
                PREFIX bd: <http://www.bigdata.com/rdf#> 
                PREFIX dct: <http://purl.org/dc/terms/> 
                PREFIX geo: <http://www.opengis.net/ont/geosparql#> 
                PREFIX p: <http://www.wikidata.org/prop/> 
                PREFIX pq: <http://www.wikidata.org/prop/qualifier/> 
                PREFIX ps: <http://www.wikidata.org/prop/statement/> 
                PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/> 
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
                PREFIX wd: <http://www.wikidata.org/entity/> 
                PREFIX wds: <http://www.wikidata.org/entity/statement/> 
                PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
                PREFIX wdv: <http://www.wikidata.org/value/> 
                PREFIX wikibase: <http://wikiba.se/ontology#> 
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
                """
        lcquad_data = json.load(open(path_to_ds + "/qald.json"))
        samples = []
        len_qald=len(lcquad_data)
        for question in tqdm(lcquad_data):
            # print(question)
            if "entities" in question and question["question"] is not None:
                question_str = question["question"] + "[SEP] entities: "
                entities = question["entities"]
                for ent in entities:
                    question_str += ent["label"] + " : " + ent["uri"].replace("http://www.wikidata.org/entity/",
                                                                              "") + " , "
                question_str = question_str + "relations: "
                relations = question["relations"]
                for rel in relations:
                    question_str += rel["label"] + " : " + rel["uri"].replace("http://www.wikidata.org/prop/direct/",
                                                                              "") + " , "
                query = question["sparql_wikidata"]
                print(query)
                parsed_query = sparql.parser.parseQuery(prefixes + query)
                en = algebra.translateQuery(parsed_query)
                '''

                for ent in question["entities"]:
                    key = ent["uri"].replace("http://www.wikidata.org/entity/", "")
                    query = query.replace(key, ent["label"])
                for rel in question["relations"]:
                    key = rel["uri"].replace("http://www.wikidata.org/prop/direct/", "")
                    query = query.replace(key, rel["label"])
                '''
                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")
                sample = {"input": question_str + "[SEP]target_wikidata", "label": query}
                samples.append(sample)
        if self.args["use_freebase"]:
            grail_qa = json.load(open(path_to_ds + "/grail.json"))
            types = pickle.load(open("../precomputed/Generator/type_dict_freebase.pkl", "rb"))
            relation_labels = pickle.load(open("../precomputed/Generator/relation_labels.pkl", "rb"))
            with(open(path_to_ds + "/dense_retrieval_grailqa.jsonl", "r")) as file:
                schema = {}
                for ln in file:
                    el = json.loads(ln)
                    schema[el["qid"]] = el
            for el in tqdm(grail_qa[:len_qald]):
                question_str = el["question"] + "[SEP] entities: "
                nodes = el["graph_query"]["nodes"]
                classes = set()
                nodes_to_add = []
                for n in nodes:
                    if not n["node_type"] == "literal":
                        if n["node_type"] == "class":
                            classes.add(n["id"])
                        nodes_to_add.append(n)
                while len(classes) < 3:
                    for cl in schema[el["qid"]]["classes"]:
                        if not cl in classes:
                            classes.add(cl)
                            nodes_to_add.append({"id": cl, "friendly_name": types[cl]})
                        if len(classes) == 3:
                            break
                #                    question_str += n["friendly_name"] + " : " + n["id"]+ " , "
                shuffle(nodes_to_add)
                for n in nodes_to_add:
                    question_str += n["friendly_name"] + " : " + n["id"] + " , "
                edges = el["graph_query"]["edges"]
                relations = set()
                relations_to_add = []
                for e in edges:
                    relations.add(e["relation"])
                    relations_to_add.append(e)
                while len(relations) < 3:
                    for rel in schema[el["qid"]]["relations"]:
                        if not rel in relations:
                            relations.add(rel)
                            relations_to_add.append({"relation": rel, "friendly_name": relation_labels[rel]})
                        if len(relations) == 3:
                            break
                shuffle(relations_to_add)
                question_str = question_str + "relations: "
                for e in relations_to_add:
                    question_str += e["friendly_name"] + " : " + e["relation"] + " , "

                query = el["sparql_query"]
                nodes = el["graph_query"]["nodes"]
                edges = el["graph_query"]["edges"]
                # query=preprocessfreebasequery(query)
                parsed_query = sparql.parser.parseQuery(query)

                en = algebra.translateQuery(parsed_query)

                '''
                for e in edges:
                    query = query.replace(e["relation"], e["friendly_name"])
                for n in nodes:
                    query = query.replace(n["id"], n["friendly_name"])
                '''
                query = query.replace("\n", "")
                query = query.replace("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>", "")
                query = query.replace(
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>", "")
                query = query.replace(
                    "PREFIX : <http://rdf.freebase.com/ns/> ", "")

                res_vars = en.algebra["PV"]
                vars = en.algebra["_vars"]
                it = 0

                for el in res_vars:
                    query = query.replace("?" + el, "_result_" + str(it))
                    it += 1
                it = 0
                for el in vars:
                    query = query.replace("?" + el, "_var_" + str(it))
                    it += 1
                query = query.replace("{", "_cbo_")
                query = query.replace("}", "_cbc_")
                sample = {"input": question_str + "[SEP]target_freebase", "label": query}

                samples.append(sample)
        return samples

class Dataprocessor_QALD(Dataprocessor_KBQA_basic):
    def read_ds_to_list(self,path_to_ds):
        samples=[]
        data=json.load(open(path_to_ds,"r",encoding="utf-8"))
        for question in data["questions"]:
            sample={}
            for lang in question["question"]:
                if lang["language"]=="en":
                    sample["input"]=lang["string"]
            sample["label"]=question["query"]["sparql"]
            samples.append(sample)
        return samples






