# Data Extraction and Natural Language Processing (NLP) Project

## **Objective**
The goal of this project is to extract textual content from articles available at the provided URLs and perform text analysis to compute several defined metrics, including:

- **Sentiment Analysis**: Positive Score, Negative Score, Polarity Score, and Subjectivity Score.
- **Readability Metrics**: Average Sentence Length, Percentage of Complex Words, Fog Index, and Average Number of Words per Sentence.
- **Other Metrics**: Complex Word Count, Word Count, Syllables per Word, Personal Pronouns, and Average Word Length.

The project automates the extraction, cleaning, and analysis of data using Python and outputs the results as an Excel file.

---

## **Data Extraction**
- To accomplish text extraction, I have used Python's requests and BeautifulSoup library to fetch the HTML content from URLs listed in the input.xlsx file. The HTML content is then parsed using BeautifulSoup with the lxml parser to efficiently extract the article's title and body text, while avoiding unnecessary elements like headers, footers, and ads. The relevant content is saved into individual text files named with the unique URL_ID for each article. This approach ensures that only the necessary article information is extracted and stored for further processing.

---

## **Natural Language Processing and Data Analysis**
- To perform textual analysis on the extracted article texts and compute the specified variables, I used a structured approach encapsulated in the TextPreprocessing class. Below I have mentioned summary of the steps I performed to accomplish:

---	Data Preparation and Cleaning and Analysis:  The ‘get_stopwords_get_data’ method combines multiple stopword files into one and reads text files from a specified directory. It then preprocesses these texts by removing stopwords, special characters, punctuation, and digits. This cleaned data is returned in a structured format as a DataFrame.

--- Performs **word tokenization and lemmatization using NLTK** to enhance the quality of text processing.

---	The ‘preprocess_analyze’ method performs further text cleaning by tokenizing content into sentences and words, removing stopwords, and ensuring proper formatting. 

--- -	The ‘get_masterdic_files’ method extracts and processes positive and negative word dictionaries from a zip file. It decodes the files, removes stopwords, and stores the clean lists of positive and negative words as Instance attributes. I have used a try and except block to handle any potential issues that may arise during the process.

--- -	The ‘compute_variables’ method performs several text analysis tasks to compute various readability and sentiment metrics for the extracted data. This is the final method in the process and includes the following steps:     
