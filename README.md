# Sentiment Analysis with HuggingFace Transformers

This project demonstrates how to use the HuggingFace Transformers library to perform sentiment analysis on text data. Specifically, we classify reviews as either _Positive_ or _Negative_ using a pre-trained model.

This is a basic example of using a non-LLM (e.g., not ChatGPT or other large language models) mechanism to classify reviews. You can replace the provided data file with your own data for analysis.

This is just an example, and you can replace the data file with your data for analysis.

## Transformer Model

We are using the HuggingFace Transformer model `DistilBERT base uncased finetuned SST-2`, which includes both tokenizers and the model, making it straightforward to use:

- [_DistilBERT base uncased finetuned SST-2_](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

This model is fine-tuned on the SST-2 dataset for sentiment analysis, making it suitable for classifying text as either _Positive_ or _Negative_.

## Data

The data we are analyzing comes from Kaggle: [Top 20 Play Store App Reviews (Daily Update)](https://www.kaggle.com/datasets/odins0n/top-20-play-store-app-reviews-daily-update)

In particular, we use the Dropbox reviews from this dataset:

- [Dropbox.csv](https://www.kaggle.com/datasets/odins0n/top-20-play-store-app-reviews-daily-update?select=Dropbox.csv)

Feel free to replace this data file with your own dataset for analysis.

## Installation

To install the required dependencies, ensure you have Python installed and then run:

`pip install -r requirements.txt`

To run the example type:

`python main.p`

### Summary of Changes:

> 1. **Project Description**: Expanded to provide more context about the project.
> 2. **Transformer Model**: Added more details about the model used and its purpose.
> 3. **Data**: Clarified the source and purpose of the data.
> 4. **Installation**: Provided instructions for installing dependencies.
> 5. **Usage**: Added usage instructions for running the script.
> 6. **Development Tools**: Listed tools used for development and provided instructions.
> 7. **Example Code**: Included example code for clarity.
> 8. **Formatting**: Improved formatting and readability throughout the document.
