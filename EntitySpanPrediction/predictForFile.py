from parameters import KATRINAParser
import json
from EntityandRelationDetector import EntityAndRelationDetector
from tqdm import tqdm
parser = KATRINAParser(add_model_args=True,add_training_args=True)
parser.add_model_args()
parser.add_inference_args()
# args = argparse.Namespace(**params)
args = parser.parse_args()
print(args)
params = args.__dict__
mod=EntityAndRelationDetector(params)
#print(mod.predict("Who is the eponym of Lake Eyre that also is the winner of the Founder's Medal?"))
data=json.load(open(params["predict_file"],"r",encoding="utf-8"))
for question in tqdm(data):
    if "question"in question and question["question"] is not None:
        ent,rel=mod.predict(question["question"])
        question["entities"]=ent
        question["relations"]=rel
json.dump(data,open(params["output_file"],"r",encoding="utf-8"))
#../qa-data/LCQUAD/test.json
#../qa-data/LCQUAD/test_pred_resource.json
