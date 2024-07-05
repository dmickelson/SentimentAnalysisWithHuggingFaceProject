import pandas as pd
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the pre-trained tokenizer and model from HuggingFace
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

if __name__ == '__main__':
    # Transformer tokenizer and model from HuggingFace
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")

    # Create the sentiment analysis pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # Load the sample data file
    df = pd.read_csv('./data/Dropbox.csv')

    # For demo purposes, limit the sample size to 100 entries
    df = df.sample(100)

    # Get the reviews from the 'content' column
    texts = list(df.content.values)

    # Perform sentiment analysis on the reviews
    results = nlp(texts)

    # Extract the sentiment labels from the analysis results
    df['sentiment'] = [r['label'] for r in results]

    # Print the first 10 entries of the dataframe to verify the results
    print(df.head(10))

    # Uncomment the following block if you want to print each review with its analysis result and score
    # Use list zip the result, f-score and print
    # zip(texts, results, df.score.values)
    # for text, result, score in zip(texts, results, df.score.values):
    #     print(f'Text: {text}')
    #     print(f'Result: {result}')
    #     print(f'Score: {score}')
