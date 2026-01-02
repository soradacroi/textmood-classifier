  multinomial naive bayes classifier  body { max-width: 40em; padding: 2em 10%; margin: 0 auto; font: 1.2em/1.62 sans-serif; color: #444; background-color: rgb(241, 240, 236); } h1, h2, h3 { line-height: 1.2; margin-bottom: 1.5rem; color: #222; } p { margin-top: 20px; margin-bottom: 50px; text-align: justify; hyphens: auto; } aside { color: rgb(121, 121, 121); font-size: 0.8em; margin-bottom: 1em; } a { text-decoration: underline; color: rgb(51, 59, 97); } pre\[class\*="language-"\] { background: rgba(255, 255, 255, 0.7) !important; border: 1px solid rgba(0, 0, 0, 0.05) !important; border-left: 5px solid rgb(51, 59, 97) !important; padding: 1.5em !important; margin: 2em 0 !important; border-radius: 4px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); overflow: auto; } code\[class\*="language-"\] { font-family: 'Consolas', 'Monaco', 'Andale Mono', monospace !important; font-size: 0.9em !important; text-shadow: none !important; } .token.keyword { color: #859900; font-weight: bold; } .token.string { color: #2aa198; } .token.comment { color: #93a1a1; } .token.function { color: #268bd2; } .output-block { background-color: rgba(39, 7, 7, 0.026); border: 1px dashed #ccc; color: #666; padding: 1.2em; margin: 1em 0 2em 0; font-size: 0.9em; word-break: break-all; line-height: 1.4; border-radius: 4px; } .output-label { font-size: 0.9em; font-weight: bold; color: #999; margin-bottom: 5px; display: block; }

Textmood Classifier
===================

Dec 2025

I recently started building a tool to detect emotion in text ([github/ai-emotion-detection-py](https://github.com/soradacroi/ai-emotion-detection-py)). As I was looking into implementation I stumbled upon **[naive bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)**. While making the emotion detection I thought to myself why not make a library out of it, I have never made a proper library, but i tried this time... idk what i want to write...  
So after _finishing_ I made this library. (LIBRARY???????????????)  
  
More resources:

*   [video by StatQuest with Josh Starmer](https://youtu.be/O2L2Uv9pdDA?si=24xVgaPAKTjpbXCt)
*   [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

Source code:  
[github/textmood-classifier](https://github.com/soradacroi/textmood-classifier)  
  

Negation and Stopwords
----------------------

    
        for t in tokens:
            if t in self.stop_words:
                continue
            if t in self.negations:
                negate = True
                continue
            if negate:
                processed_tokens.append("NOT_" + t)
                negate = False 
            else:
                processed_tokens.append(t)
        

so this "libary" does like how do i explain this...  
So this library tracks negation words and transform the next word as like an unique word, for example a sentence "hello i am not happy" will get transform like "hello i am NOT\_happy", that way it have some context as multinomial naive bayes is not context aware. (the n-gram (later) does the same thing but because i am the best human (no i am not) what i am doing is always better wahahahahaha)  
And well the stop words help with the negation like,  
Let "i am not very happy" be a data, let "very" be a stop word so, after transformation it will be "i am NOT\_happy" very got removed and made "not very happy" mean sad so it can now have more complex data. ig i hope

N-gram Generation
-----------------

          
        all_features = []
        min_n, max_n = self.ngram_range
        
        n_tokens = len(processed_tokens)
        
        for n in range(min_n, max_n + 1):
            if n == 1:
                all_features.extend(processed_tokens)
            else:
                for i in range(n_tokens - n + 1):
                    gram = "_".join(processed_tokens[i : i + n])
                    all_features.append(gram)

The n-gram generation works like making two (more depends on your given depth, or just one if u do (1, 1)) one word or token,for example:  
"i am not very happy" (using the example from _#Negation and Stopwords_)  
the Negation and Stopwords will generate \[i, am, NOT\_happy\] then,  
N-gram generator will output something like (if depth is (1, 2)):  
\[i, am, NOT\_happy, i\_am, am\_NOT\_happy\] increasing the tokens and context. While increasing the depth will increase the context, it will also increase the size of the self.vocab set and the model.pkl file. Also the problem of [overfitting](https://en.wikipedia.org/wiki/Overfitting) comes up, they require significantly larger datasets to avoid overfitting. An ngram\_range of (1, 2) or (1, 3) provides the best balance between understanding context and remaining generalized. _(for me atleast)_

Basic set-up
------------

    
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
        

You can also load data from your dataset (a csv, text or anything just need to have separators and same labels as your targeted labels) by using the load\_data function.

    
    data = load_data(dataset_path, amount, labels, separator)
        

Save and Load
-------------

    
    model.save_model("my_model.pkl")
    
    # later... 
    model = TextMood()
    model.load_model("my_model.pkl")
    print(model.predict(input()))
        

Laplace Smoothing
-----------------

    
    # From your predict function:
    p_feat_given_class = (feat_count_in_class + smooth) / (total_feat_in_class + (vocab_size * smooth))
        

If you try to predict a sentence with a word the model has never seen before, the probability for that word becomes 0. Since we multiply probabilities (and it never saw that word so probability of that word is ZERO), a single zero would destroy the entire calculation.  
**Laplace Smoothing** (controlled by the `smooth` parameter) adds a tiny number to every count. This ensures every word has at least a _tiny_ chance of existing in every category, preventing the model from failing on new data.

  

Limitations and all
-------------------

*   If you train your model with 1000 positive reviews but only 100 negative reviews, the priors will get skewed, so try to have same numbers of label counts, use the amount from the load\_data to get equal data for all the labels.
*   A sentence like "Oh great, another delay, just what I wanted!" contains the words "great" and "wanted." The model will likely see these as highly positive features and guess "positive" missing the sarcasm
*   Currently, the model ignores symbols (like ! or ?) and also numbers, so it might lose the _"intensity"_ of the emotion.

Thank you.