from textmood import TextMood

MODEL_FILE = "mod.pkl"

def main():
    model = TextMood()

    model.load_model(MODEL_FILE)
    print("Model loaded successfully")
    while True:
        try:
            user_input = input("\ninput: ")
            if user_input.lower() == "quit":
                break
            if not user_input.strip():
                continue
            prediction = model.predict(user_input)
            print(f"Prediction: {prediction.upper()}")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()