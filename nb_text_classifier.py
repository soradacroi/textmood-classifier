# TextMood-Classifier

import math
import random
import pickle
import re

class TextMood:
    def __init__(self, stop_words=None, negations=None, ngram_range=(1, 3)):
        self.stop_words = stop_words if stop_words else set()
        self.negations = negations if negations else set()
        # parameter: (min_n, max_n). Default is (1, 3) for uni, bi, and trigrams
        self.ngram_range = ngram_range
        
        # Model parameters
        self.counts = {}
        self.total_feat_count = {}
        self.docs_per_class = {}
        self.vocab = set()
        self.priors = {}
        self.labels = []

    def _get_features(self, text):
        # Clean text: keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text.lower())
        tokens = text.split()
        
        processed_tokens = []
        negate = False
        
        # Handle Negations and Stopwords
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
                
        # N-gram Generation
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
                    
        return all_features

    def train(self, train_data, labels, verbose=True, print_vocab = False):
        self.labels = labels
        for label in labels:
            self.counts[label] = {}
            self.total_feat_count[label] = 0
            self.docs_per_class[label] = 0

        for text, label in train_data:
            self.docs_per_class[label] += 1
            features = self._get_features(text)
            for feat in features:
                self.counts[label][feat] = self.counts[label].get(feat, 0) + 1
                self.total_feat_count[label] += 1
                self.vocab.add(feat)

        # Calculate log priors: log(P(Class))
        total_docs = sum(self.docs_per_class.values())
        for label in labels:
            if total_docs > 0 and self.docs_per_class[label] > 0:
                self.priors[label] = math.log(self.docs_per_class[label] / total_docs)
            else:
                self.priors[label] = -1e9
        if print_vocab == True:
            print(self.vocab)
        if verbose:
            print(f"Training complete. Vocab size: {len(self.vocab)}")

    def predict(self, text, smooth=0.1):
        features = self._get_features(text)
        vocab_size = len(self.vocab)
        scores = {}

        for label in self.labels:
            scores[label] = self.priors[label]
            total_feat_in_class = self.total_feat_count[label]
            
            for feat in features:
                if feat not in self.vocab:
                    continue
                    
                feat_count_in_class = self.counts[label].get(feat, 0)
                
                # Laplace Smoothing formula:
                # P(Feature|Class) = (count + smooth) / (total_class_count + vocab_size * smooth)
                p_feat_given_class = (feat_count_in_class + smooth) / (total_feat_in_class + (vocab_size * smooth))
                scores[label] += math.log(p_feat_given_class)

        return max(scores, key=scores.get)

    def save_model(self, file_path="model.pkl"):
        state = {
            # Save the logic settings
            'config': {
                'stop_words': self.stop_words,
                'negations': self.negations,
                'ngram_range': self.ngram_range
            },
            # Save the learned data
            'data': {
                'counts': self.counts,
                'total_feat_count': self.total_feat_count,
                'docs_per_class': self.docs_per_class,
                'vocab': self.vocab,
                'priors': self.priors,
                'labels': self.labels
            }
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)

    def load_model(self, file_path="model.pkl"):
        with open(file_path, "rb") as f:
            state = pickle.load(f)
            # Restore config
            self.stop_words = state['config']['stop_words']
            self.negations = state['config']['negations']
            self.ngram_range = state['config']['ngram_range']
            # Restore data
            d = state['data']
            self.counts = d['counts']
            self.total_feat_count = d['total_feat_count']
            self.docs_per_class = d['docs_per_class']
            self.vocab = d['vocab']
            self.priors = d['priors']
            self.labels = d['labels']   

def load_data(data_file_path: str, amount: int, labels: list, separator: str):
    all_samples = []
    labels_set = set(labels)
    counts = {label: 0 for label in labels}
    
    with open(data_file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.split(separator, 1)
            if len(parts) < 2: continue
                
            label = parts[0].strip()
            content = parts[1].strip()
            
            if label in labels_set and counts[label] < amount:
                all_samples.append((content, label))
                counts[label] += 1
            
            if all(c >= amount for c in counts.values()):
                break
    return all_samples