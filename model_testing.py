from transformers import BertConfig, BertModel, BertTokenizer


# This will output the most basic - randomly configured model. While this is technically usable in this state - it's output will be random until it is trained. To avoid duplicating training effore - take advantage of the "from_pretrained" method that the model exposes
# config = BertConfig()
# model = BertModel(config)

model = BertModel.from_pretrained("bert-base-cased")
print(model)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
print(tokenizer("Using a tokenizer is simple!!"))

## You can save either the mdoel or the tokenizer to your machine for more efficient running down the line. Commenting these out, but this would look like:
# model.save_pretrained("directory on your machine") || tokenizer.save_pretrained("directory on your machine")










