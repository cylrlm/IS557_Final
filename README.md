# IS557_Final
This repository contains code for a text classification project that distinguishes between Democratic and Republican political texts.

## Project Structure

The repository consists of ??? main Python notebooks:

- **dataConcat.ipynb**: Loads and merges Democratic and Republican datasets into unified train and test sets.
- **explorer.ipynb**: Performs exploratory data analysis, including class distribution, text length analysis, vocabulary analysis, and political marker detection.
- **training.ipynb**: Trains and evaluates various machine learning models for political text classification.
- **distill_bert.ipynb**: Full pipeline to train, test and analyze with DistilBERT.
## Dataset
- **gpt directory**:  
1. split_test_data.py: Split dataset into 10 equal parts.
2. classify_with_gpt.py: Call OpenAI API to classify splitted test data.
3. analyze_results.py: Analyze result.

The dataset consists of political texts labeled as either "democratic" or "republican". The data is split into training and testing sets with the following files:
- `democratic_only.train.en.parquet`
- `republican_only.train.en.parquet`
- `democratic_only.test.en.parquet`
- `republican_only.test.en.parquet`

After data preparation, the processed data is saved as:
- `clean_train.csv`
- `clean_test.csv`