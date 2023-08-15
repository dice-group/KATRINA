from transformers import T5Tokenizer, T5ForConditionalGeneration
from GeneratorModel.data_processing import Dataprocessor_test
dp=Dataprocessor_test(T5Tokenizer.from_pretrained("t5-small"),"")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
input="translate English to German: The house is wonderful."
labels="Das Haus ist schön"
sample = dp.process_sample(input,labels)
i=dp.process_sample(input).input_ids
l=dp.process_sample(labels).input_ids
# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids = i, labels=l).loss
print(loss.item())
