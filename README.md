# HTRC Indexing and Analysis with Python

## Overview
This project performs indexing and analysis on the HathiTrust Research Center (HTRC) dataset using Python. The script processes data, creates indices for efficient searching, and allows users to analyze large textual datasets. The project is built for handling large-scale text data, leveraging indexing techniques to improve search efficiency.

## Features
- **Index Creation**: Efficiently indexes large datasets for faster searching and retrieval.
- **Data Processing**: Preprocesses the text data for analysis by handling tokenization, cleaning, and normalization.
- **Text Analysis**: Provides basic text analysis functions such as term frequency, word counts, and more.

## Project Structure
- **`htrc-indexing.py`**: The main Python script responsible for indexing and analyzing the HTRC dataset.

## How to Run the Project

### Prerequisites
- **Python 3.x**: Ensure Python is installed.
- **Required Libraries**: Install the following libraries using `pip`:
    ```bash
    pip install pandas numpy nltk
    ```

### Running the Script

1. Clone the repository:
    ```bash
    git clone https://github.com/arnold-shakirov/HTRC-Indexing.git
    ```

2. Navigate to the project directory:
    ```bash
    cd htrc-indexing
    ```

3. Run the Python script:
    ```bash
    python htrc-indexing.py
    ```

### Output
The script will output:
- **Indexed Data**: Displays the progress and result of indexing the dataset.
- **Text Analysis Results**: Outputs results from the text analysis such as term frequency and word counts.

#### Sample Output:
```text
Indexing complete: 10000 documents processed.
Top 10 most frequent terms:
1. the
2. and
3. to
...
```
### Dataset
The dataset for this project is the HathiTrust Research Center (HTRC) dataset. The data should consist of large text files that need to be indexed for efficient analysis.

### Requirements
Python 3.x or higher
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
NLTK: For natural language processing tasks such as tokenization and text cleaning.

### Installing Libraries
To install the required libraries, use:
pip install pandas numpy nltk

### Contact
For any questions or suggestions, feel free to reach out to me at [ashakirov@stetson.edu].
