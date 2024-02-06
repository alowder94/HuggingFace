from transformers import pipeline


classifier = pipeline("sentiment-analysis")
print(classifier(["I love this!", "I hate this!"]))

# classifier = pipeline("zero-shot-classification")
# print(classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# ))




