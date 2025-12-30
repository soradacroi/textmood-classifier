from textmood import TextMood, load_data

# 1. Configuration
target_labels = ["positive", "negative"]
stop_words = {"the", "a", "is", "and", "it", "to", "this", "in", "of"}
negations = {"not", "no", "never", "didnt", "isnt", "wasnt"}

# 2. Load the data 
formated_dataset = load_data("dataset1.csv", amount=100, labels=target_labels, separator=",")

# 3. Initialize the model
model = TextMood(stop_words=stop_words, negations=negations)

# 4. train
model.train(formated_dataset, target_labels)

# 5. Test it
while True:
    test_phrase = input("Input: ")
    if test_phrase == "quit":
        break
    prediction = model.predict(test_phrase)

    print(f"Prediction: {prediction}")
