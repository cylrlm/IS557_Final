# IS557_Final
This repository contains code for a text classification project that distinguishes between Democratic and Republican political texts.

## Project Structure

The repository consists of ??? main Python notebooks:

- **dataConcat.ipynb**: Loads and merges Democratic and Republican datasets into unified train and test sets.
- **explorer.ipynb**: Performs exploratory data analysis, including class distribution, text length analysis, vocabulary analysis, and political marker detection.
- **training.ipynb**: Trains and evaluates various machine learning models for political text classification.

## Dataset

The dataset consists of political texts labeled as either "democratic" or "republican". The data is split into training and testing sets with the following files:
- `democratic_only.train.en.parquet`
- `republican_only.train.en.parquet`
- `democratic_only.test.en.parquet`
- `republican_only.test.en.parquet`

After data preparation, the processed data is saved as:
- `clean_train.csv`
- `clean_test.csv`