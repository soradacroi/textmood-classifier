from textmood import TextMood, load_data


target_labels = ["toxic", "normal"]
stop_words = {"the", "a", "is", "and", "it", "to", "this", "in", "of"}
negations = {"not", "no", "never", "didnt", "isnt", "wasnt"}
MODEL_FILE = "tests/mod.pkl"

def main():
    dataset = load_data("tests\dataset2.csv", amount=150, labels=target_labels, separator=",")

    model = TextMood(stop_words=stop_words, negations=negations, ngram_range=(1, 2))
    
    print(f"--- Training on {len(dataset)} samples ---")
    model.train(dataset, target_labels)
    

    print(f"--- Saving model to {MODEL_FILE} ---")
    model.save_model(MODEL_FILE)
    print("Training and saving complete!")

if __name__ == "__main__":
    main()