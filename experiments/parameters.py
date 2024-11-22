import argparse
import os

class KATRINAParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_training_args:
        (default False) initializes the default arguments for model training
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    :param add_inference_args:
        (default False) initializes the default arguments for inference,
        including prefix tries.
    """

    def __init__(
        self, add_training_args=False, add_model_args=False,add_inference_args=False,
        description=' ',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
        )

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_model_args:
            self.add_model_args()
        if add_inference_args:
            self.add_inference_args()
    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )

        parser.add_argument(
            "--model_name",
            default="t5-base",
            type=str,
            help="path or name of model",
        )


    def add_inference_args(self,args=None):
        parser = self.add_argument_group("Model Arguments")

        parser.add_argument(
            "--pretrained_model_path",
            # default"models/combined_model"
            type=str,
            help="Path to pretrained model"
        )
        parser.add_argument(
            "--cuda_device_id",
            default=0,
            type=int,
            help="id of cuda device to use"
        )
        parser.add_argument(
            "--parameter_path_prefix",
            default="",
            type=str,
            help="optional to add for each path parameter"
        )
        parser.add_argument(
            "--predict_file",
            #default="../qa-data/combined/test/grail.json",
            type=str,
            help="LCQuAD of grail qa formated file in case of LCQuAD the file should also contain predicted entities"
        )
        parser.add_argument(
            "--output_file",
            default="output_file",
            type=str,
            help="file_to_write"
        )
        parser.add_argument(
            "--benchmark_KG",
            default="freebase",
            type=str,
            help="use freebase or wikidata KG for benchmarking in case of freebase, the script expects a file in grail qa-format, else in LC-QuAD format"
        )

        parser.add_argument(
            "--use_entities",
            default=True,
            type=bool,
            help="add entities and types in input"
        )

        parser.add_argument(
            "--use_relations",
            default=True,
            type=bool,
            help="add relations in input"
        )

        parser.add_argument(
            "--wikidata_sparql_endpoint",
            default="https://query.wikidata.org/sparql",
            type=str,
            help="wikidata sparql endpoint"
        )

        parser.add_argument(
            "--freebase_sparql_endpoint",
            default="https://freebase.data.dice-research.org/sparql",
            type=str,
            help="freebase sparql endpoint"
        )

        parser.add_argument(
            "--freebase_qa_schema_file",
            default="../qa-data/GrailQA_v1.0/dense_retrieval_grailqa_dev.jsonl",
            type=str,
            help="file with schema linking results"
        )

        parser.add_argument(
            "--freebase_qa_entity_file",
            default="../qa-data/GrailQA_v1.0/grailqa_el.json",
            type=str,
            help="file with entity linking results for e.g grail qa"
        )

        parser.add_argument(
            "--freebase_type_dict",
            default="../precomputed/Generator/type_dict_freebase.pkl",
            type=str,
            help="freebase type dict"
        )
        parser.add_argument(
            "--use_gold_res_freebase",
            default=False,
            type=bool,
            help="configure if gold resources should be used in freebase evaluation"
        )
        parser.add_argument(
            "--gold_resource_benchmark",
            default="../qa-data/combined/test/grail.json",
            type=str,
            help="benchmark with gold entities"
        )

        parser.add_argument(
            "--freebase_relation_dict",
            default="../precomputed/Generator/relation_labels.pkl",
            type=str,
            help="freebase relation dict"
        )

        parser.add_argument(
            "--wikidata_bechmark_entities",
            default="../qa-data/combined/test/lcquad.json",
            type=str,
            help="lcquad fomated benchmark with entities"
        )
