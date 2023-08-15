import argparse
import os

class KATRINAParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_blink_args:
        (default True) initializes the default arguments for BLINK package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_training_args=False, add_model_args=False,
        description=' ',
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler='resolve',
            formatter_class=argparse.HelpFormatter,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ['BLINK_HOME'] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_model_args:
            self.add_model_args()
            if add_training_args:
                self.add_training_args()
    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_input_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
                 "Sequences longer than this will be truncated, and sequences shorter \n"
                 "than this will be padded.",
        )
        parser.add_argument(
            "--config_name",
            default=None,
            type=str,
            help="Pretrained config name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default='hfcache',
            type=str,
            help="Where do you want to store the pretrained models downloaded from s3",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--train_data_file",
            default=None,
            type=str,
            help="File with Training data",
        )
        parser.add_argument(
            "--eval_data_file",
            default=None,
            type=str,
            help="File with Training data",
        )
        parser.add_argument(
            "--model_name",
            default=None,
            type=str,
            help="path or name of model",
        )
        parser.add_argument(
            "--eval_beams",
            default=None,
            type=str,
            help="# num_beams to use for evaluation.",
        )


    def add_training_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--warmup_ratio",
            default=0.0,
            type=float,
            help="The warmup ratio",
        )
        parser.add_argument(
            "--label_smoothing",
            default=256,
            type=float,
            help="The label smoothing epsilon to apply (if not zero).",
        )
        parser.add_argument(
            "--sortish_sampler",
            default=None,
            type=bool,
            help="Whether to SortishSamler or not.",
        )
        parser.add_argument(
            "--predict_with_generate",
            default='hfcache',
            type=bool,
            help="Whether to use generate to calculate generative metrics (ROUGE, BLEU).",
        )
        parser.add_argument(
            "--train_model",
            default=True,
            type=bool,
            help="Whether to train a model",
        )
        parser.add_argument(
            "--output_dir",
            default=True,
            type=bool,
            help="Whether to train a model",
        )
