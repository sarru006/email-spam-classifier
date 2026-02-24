import pickle
import numpy as np

def debug_model():
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("nb_model.pkl", "rb") as f:
            model = pickle.load(f)
        
        feature_names = vectorizer.get_feature_names_out()
        # Log probability of features given a class
        # NB log probability: log P(x_i | class)
        # model.feature_log_prob_[0] is ham, [1] is spam
        
        ham_probs = model.feature_log_prob_[0]
        spam_probs = model.feature_log_prob_[1]
        
        # Difference in log probs gives the most discriminative words
        diff = spam_probs - ham_probs
        top_spam_indices = np.argsort(diff)[-20:]
        top_ham_indices = np.argsort(diff)[:20]
        
        print("Top 20 SPAM indicators:")
        for i in reversed(top_spam_indices):
            print(f"  {feature_names[i]}: {diff[i]:.4f}")
            
        print("\nTop 20 HAM indicators:")
        for i in top_ham_indices:
            print(f"  {feature_names[i]}: {diff[i]:.4f}")

        # Check a known spam message
        from preprocessor import preprocess
        test_msg = "WINNER! You have won a 1000 cash prize. Call 09061701461 to claim your gift."
        clean = preprocess(test_msg)
        print(f"\nTest Message: {test_msg}")
        print(f"Cleaned: {clean}")
        
        X = vectorizer.transform([clean])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        print(f"Prediction: {'SPAM' if pred == 1 else 'HAM'}")
        print(f"Confidence: Spam={prob[1]:.4f}, Ham={prob[0]:.4f}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_model()
