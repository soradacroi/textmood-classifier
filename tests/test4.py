from textmood import TextMood, load_data

# 1. Configuration
stop_words={"the", "is", "very"}, 
negations={"not", "no", "never"},
ngram_range=(1, 2)


# Training data: (text, label)
data = [("I love this", "pos"), ("not good", "neg")]
model = TextMood(stop_words=stop_words, negations=negations, ngram_range=ngram_range)
model.train(data, labels=["pos", "neg"])

# Predict
print(model.predict("The movie not good"))