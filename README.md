# Text_Feature_Extraction-Bag_Of_Words-TF-IDF

# Toxic Comments Classifier

This Python script provides functionality for building and evaluating a toxic comments classifier using machine learning techniques. The classifier is trained on a dataset of comments and their corresponding toxicity labels (0 for non-toxic and 1 for toxic). The implementation uses natural language processing (NLP) techniques and scikit-learn library for text vectorization and logistic regression classification.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

- numpy
- pandas
- scikit-learn
- spacy

You can install these libraries using the following command:

```bash
pip install numpy pandas scikit-learn spacy
```

## Usage

1. **Install Required Libraries:**
   Make sure to install the necessary Python libraries as mentioned in the 'Prerequisites' section.

2. **Download the Dataset:**
   The script expects the input data to be in a CSV file format. Ensure the dataset (for example, "toxic_comments_500.csv") is present in the same directory as the script.

3. **Run the Script:**
   Execute the script in your Python environment. It will preprocess the text data, train two classifiers using different vectorization techniques (Count Vectorization and TF-IDF), and evaluate the classifiers' performance using logistic regression.

   ```bash
   python toxic_comments_classifier.py
   ```

4. **Interpreting the Output:**
   The script will output the accuracy, precision, and recall scores for both classifiers on the test dataset. These metrics provide an understanding of how well the classifier is performing in terms of correctly identifying toxic comments.

## Customization

- **Tokenizer Function:**
  You can customize the `spacy_tokenizer` function in the script to modify the text preprocessing steps according to your requirements. For example, you can add more stopwords or perform additional text cleaning operations.

- **Classifier and Vectorizer:**
  The script uses logistic regression as the classification algorithm and provides two vectorization options: Count Vectorization and TF-IDF. If you want to experiment with different classifiers or vectorization techniques, you can modify the `classifier` and `vectorizer` variables in the script.
