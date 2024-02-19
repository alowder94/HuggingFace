from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification

import torch


# classifier = pipeline("sentiment-analysis")
# print(classifier(["I love this!", "I hate this!"]))

# classifier = pipeline("zero-shot-classification")
# print(classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# ))

## Translation example - good first feature for AI UI/API pet project
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# translator("Ce cours est produit par Hugging Face.")

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)

print(model(**inputs))

outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs['logits'], dim=-1)
print(predictions)

modelLabels = model.config.id2label # Shows what position means what in the model's response







