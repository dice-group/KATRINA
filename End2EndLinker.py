from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
import json
from SPARQLWrapper import SPARQLWrapper,JSON
from GENRE import GenreLinker
from GENRE import genre
import torch
class End2End_Model:
    def __init__(self):
        #Query GenerationModel
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dp_query=Dataprocessor_test(T5Tokenizer.from_pretrained("t5-large"),"")
        self.tokenizer_query = T5Tokenizer.from_pretrained("t5-large")
        self.query_model = T5ForConditionalGeneration.from_pretrained("GeneratorModel/complete")
        self.query_model.to(self.device)

        #NER
        self.dp_ner = Dataprocessor_test(T5Tokenizer.from_pretrained("t5-base"), "")
        self.tokenizer_ner = T5Tokenizer.from_pretrained("t5-base")
        self.ner_model = T5ForConditionalGeneration.from_pretrained("EntitySpanPrediction/span_prediction_model")
        self.ner_model.to(self.device)

        #Linking Model
        self.linker=GenreLinker.GenreLinker()
    def predict(self,question_str):
        print("start prediction")
        #sample = self.dp_ner.process_sample(input, "")
        i = self.dp_ner.process_sample(question_str).input_ids
        out = self.ner_model.generate(input_ids=i.to(self.device), max_length=650)
        ner_prediction = self.tokenizer_ner.decode(out[0], skip_special_tokens=True)
        print(ner_prediction)
        ent=ner_prediction[ner_prediction.index("entities:")+len("entities:"):-1].split(", ")
        genre_strs=[]
        for el in ent:
            genre_strs.append(question_str+" [START_ENT]"+el+"[END_ENT]")
        el_out=self.linker.link_entities(texts_with_marked_entities=genre_strs)
        input = question_str + "[SEP] "
        for ent in el_out:
                input += ent[0]["text"] + " : " + ent[0]["wikidata_id"] + " , "
        input += "[SEP]target_wikidata"
        print(input)
        i = self.dp_query.process_sample(input).input_ids
        # l=dp.process_sample(labels).input_ids
        # the forward function automatically creates the correct decoder_input_ids
        out = self.query_model.generate(input_ids=i.to(self.device), max_length=650)
        query = self.tokenizer_query.decode(out[0], skip_special_tokens=True) + "\n"
        query = query.replace("_result_", "?result")
        query = query.replace("_var_", "?var")
        query = query.replace("_cbo_", "{")
        query = query.replace("_cbc_", "}")
        print(query)
        print("end_prediction")
qa_model=End2End_Model()
qa_model.predict("What is the end time for Whitehaven resident Jonathan Swift?")

