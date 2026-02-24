from preprocessor import preprocess

def test():
    texts = [
        "Call 09061701461 to claim your prize.",
        "Visit http://spam.com today!",
        "Email me at spam@spam.com"
    ]
    for text in texts:
        print(f"Original: {text}")
        print(f"Cleaned:  '{preprocess(text)}'")

if __name__ == "__main__":
    test()
