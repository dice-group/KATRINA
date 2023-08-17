from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
dp=Dataprocessor_test(T5Tokenizer.from_pretrained("out"),"")
tokenizer = T5Tokenizer.from_pretrained("out")
model = T5ForConditionalGeneration.from_pretrained("out/checkpoint-500")
input="What is alma mata of Angela Merkel?"
labels="Das Haus ist schön"
sample = dp.process_sample(input,labels)
i=dp.process_sample(input).input_ids
l=dp.process_sample(labels).input_ids
# the forward function automatically creates the correct decoder_input_ids
out = model.generate(input_ids = i,max_length=650)

print(tokenizer.decode(out[0], skip_special_tokens=True))
