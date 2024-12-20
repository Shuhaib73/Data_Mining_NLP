# Import necessary libraries
import os
import re 
import pandas as pd 
import requests
from bs4 import BeautifulSoup
import logging
import zipfile

# Import NLTK components
import nltk 
from nltk.corpus import stopwords                        # For accessing common stop words
from nltk.tokenize import word_tokenize, sent_tokenize   # For tokenizing text into words or sentences
import syllapy                                             # For counting syllables in words


# Download the necessary NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


class ArticleScraper:
    def __init__(self, input_file, log_file):
        self.df_in = pd.read_excel(input_file)
        self.log_file = log_file

        """
        Initializes the ArticleScraper class with an input Excel file and a log file.
        
        Args:
            input_file (str): Path to the input Excel file containing URLs.
            log_file (str): Path to the log file for recording events/logs.
        """

        # Configuring logging to log messages to the specified file with a timestamped format
        logging.basicConfig(
            filename=self.log_file, 
            level=logging.INFO, 
            format='%(asctime)s - %(message)s')

    def scrape_article(self, url_id, url):
        self.url_id = url_id
        self.url = url

        """
        Scrapes the content of an article from a given URL and saves it to a text file.
        
        Args:
            url_id (str): A unique identifier for the URL (used as the filename).
            url (str): The URL of the article to scrape.
        """

        # Initializing a list to hold the scraped content
        f_contents = []

        try:
            # Send an HTTP GET request to the URL
            source = requests.get(self.url)
            source.raise_for_status()                     # Raise an error if the request was unsuccessful
            
            # Parse the HTML content using BeautifulSoup with the 'lxml' parser
            soup = BeautifulSoup(source.content, 'lxml')

            # Extract the article's title if available
            title = soup.title.string if soup.title else ""

            # Find the main content area of the article 
            contents = soup.find('div', class_='td-post-content tagdiv-type')

            if contents:
                # Extract paragraphs and list items from the content
                p_content = contents.find_all('p')
                li_content = contents.find_all('li')

                # Combining all tags into a single list
                content_tag_combined = p_content + li_content

                # Adding the title to the content list
                f_contents.append(title)

                # Processing each tag and filter out unwanted elements
                for tag in content_tag_combined:
                    # Skip tags with certain unwanted elements or short text
                    if not tag.find(['a', 'strong', 'span', 'br/']) and len(tag.text) > 40:
                        # Filtering out the below sentences as they appear in every page to avoid data redundancy  
                        if tag.text.strip().startswith("This solution was designed") or tag.text.strip().startswith("This project was done"):
                            continue
                        else:
                            f_contents.append(tag.text.strip())

                # Defined the directory to save the scraped text files
                output_dir = "scraped_txt_files"
                os.makedirs(output_dir, exist_ok=True)
                
                # Creating dynamic full file path for saving the article
                file_path = os.path.join(output_dir, f"{self.url_id}.txt")

                # Write the scraped content to a text file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write('\n'.join(f_contents))

                # Log the success message
                logging.info(f"Article from {self.url_id} saved successfully.")

        except Exception as e:
            # Log the error message
            logging.error(f"Error while processing URL: {url}: {e}")

        
    def scrape_all_articles(self):
        """
        Iterates through all rows in the input DataFrame and scrapes articles for each URL.
            
            The method extracts the 'URL_ID' and 'URL' columns from the DataFrame 
            and calls the `scrape_article` method for each entry (scrape_article will store the extracted data in separate .txt files).
        """
        # Loop through each row in the DataFrame
        for i in range(len(self.df_in)):
            self.url_id = self.df_in['URL_ID'][i]
            self.url = self.df_in['URL'][i]

            # Call the `scrape_article` method to process the current URL (and url_id to save the .txt file with dynamic name)
            self.scrape_article(self.url_id, self.url)

    
    
class TextPreprocessing:

    def get_stopwords_get_data(self, text_file_path, stop_words_out, stp_file_path):
        self.text_file_path = text_file_path
        self.stop_words_out = stop_words_out

        """
        Combines multiple stopwords files into one and loads text files into a dataset.

        Args:
            text_file_path (str): Path to the directory containing text files.
            stop_words_out (str): Path to the output file where combined stopwords will be saved.
            stp_file_path (str): Path to the directory containing stopwords files.

        Returns:
            pd.DataFrame: A DataFrame with filenames and their corresponding file contents.
        """

        # Get the list of all stopwords files in the specified directory
        stopwords_files = os.listdir(stp_file_path)

        # Open the output file in append mode to combine stopwords
        with open(stop_words_out, 'a') as outfile:
            for file_name in stopwords_files:
                # Construct the full path of the current stopwords file
                file_path = os.path.join(stp_file_path, file_name)

                if os.path.exists(file_path):
                    with open(os.path.join(stp_file_path, file_name), 'r') as infile:
                        # Append the content of the current stopwords file to the output file
                        outfile.write(infile.read())
                else:
                    print(f'Warning: {file_name} not found!')

        print(f"Files combined into {stop_words_out}")

        # Load all text files from the text_file_path directory into a dataset & List to store file data
        scraped_files = os.listdir(self.text_file_path)
        data = []

        for file in scraped_files:
            file_path = os.path.join(self.text_file_path, file)

            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='UTF-8') as infile:
                    file_content = infile.read()

                    data.append({'filename': file, 'content': file_content})

            else:
                print(f"Warning: {file} not found!")

        # Convert the list of file data into a Pandas DataFrame
        dataset = pd.DataFrame(data)
        return dataset
    
    def preprocess_analyze(self, dataset):
        """
            Preprocesses the text data in the dataset by removing stopwords, special characters, digits, and 
            performing sentence tokenization. Returns the dataset with an additional column of cleaned content.

            Args:
                dataset (pd.DataFrame): The dataset containing the text data to preprocess.
            
            Returns:
                pd.DataFrame: The updated dataset with a new 'content_cleaned' column containing cleaned sentences.
                Returns None if the dataset is empty or an error occurs.
        """
        if dataset is None:
            print("Dataset is empty or not provided!")
            return None

        def remove_stopwords_special_chars(content):
            """
                Cleans the input content by removing stopwords, special characters, digits, and extra spaces. 
                Tokenizes sentences and words, then processes each word individually.
            """
            stop_words = []

            with open(self.stop_words_out, 'r') as st_file:
                stop_words = st_file.read().lower()

            cleaned_stop_words = stop_words.replace(' | ', '\n').split('\n')
            self.stop_words_lst = [word.strip() for word in cleaned_stop_words]

            # Tokenize the content into sentences
            sentences = sent_tokenize(content)
            cleaned_sentences = []

            # Process each sentence
            for sentence in sentences:
                # Split the sentence into words
                words = word_tokenize(sentence)

                # Convert all words to lowercase
                words = [word.lower() for word in words]

                # Replace hyphens with spaces (for words like 'AI-based')
                words = [word.replace('-', ' ') for word in words]

                # Remove punctuation using regex
                words = [re.sub(r'[^\w\s]', '', word) for word in words]

                # Remove any digits
                words = [re.sub(r'\d+', '', word) for word in words]

                # Filter out stopwords and non-alphanumeric words
                filtered_words = [word for word in words if word not in self.stop_words_lst]

                # Join the filtered words to form the cleaned sentence
                cleaned_sentence = ' '.join(filtered_words)

                # remove triple spaces
                cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)

                # Join the filtered words to form the cleaned sentence
                cleaned_sentences.append(cleaned_sentence.strip())

            return cleaned_sentences

        try:
            # Apply the cleaning function to the content column in the dataset
            dataset['content_cleaned'] = list(map(lambda x: remove_stopwords_special_chars(x), dataset['content']))

        except Exception as e:
            print(f"Error processing dataset: {e}")
            return None
        
        # Return the processed dataset
        return dataset
    
    def get_masterdic_files(self, zip_file_masterDir):
        self.zip_file_masterDir = zip_file_masterDir

        """
            Extracts positive and negative words from a zip file containing a master dictionary.

            Args:
                zip_file_masterDir (str): Path to the zip file containing the master dictionary.

            Attributes:
                self.positive_words (list): List of positive words, excluding stopwords.
                self.negative_words (list): List of negative words, excluding stopwords.
        """

        try:
            with zipfile.ZipFile(self.zip_file_masterDir, 'r') as zip_file:
                all_files = zip_file.namelist()

                # Filter the file names to find the positive and negative word files
                positive_dic = [file for file in all_files if 'masterdictionary' in file.lower() and 'positive' in file.lower()]
                negative_dic = [file for file in all_files if 'masterdictionary' in file.lower() and  'negative' in file.lower()]

                # Check if both files are present
                if not positive_dic or not negative_dic:
                    print("Positive or Negative word file is not found in the zip!")
                    # return None 

                # Extract and read the positive and negative words file
                with zip_file.open(positive_dic[0]) as pos_file:
                    positive_words = set(
                        line.strip().lower()
                        for line in pos_file.read().decode('ISO-8859-1', errors='ignore').splitlines() if line.strip()
                    )

                with zip_file.open(negative_dic[0]) as neg_file:
                    # Decode content and process it
                    negative_words = set(
                        line.strip().lower()
                        for line in neg_file.read().decode('ISO-8859-1', errors='ignore').splitlines() if line.strip()
                    )

                # Filter out stopwords from the positive and negative word lists
                self.positive_words = [word for word in positive_words if word not in self.stop_words_lst]
                self.negative_words = [word for word in negative_words if word not in self.stop_words_lst]

        except Exception as e:
            print(f"An error occurred: {e}")

    def compute_variables(self, dataset):
        positive_words = self.positive_words
        negative_words = self.negative_words

        # Tokenize content_cleaned into words
        dataset['content_tokenized'] = dataset['content_cleaned'].apply(
            lambda x: [word for sentence in x for word in word_tokenize(sentence)]
        )

        # Calculate POSITIVE and NEGATIVE SCORE 
        dataset['POSITIVE SCORE'] = dataset['content_tokenized'].apply(
            lambda tokens: sum(1 for word in tokens if word in positive_words)
        )
        
        dataset['NEGATIVE SCORE'] = dataset['content_tokenized'].apply(
            lambda tokens: sum(1 for word in tokens if word in negative_words)
        )

        # Polarity and Subjectivity Scores
        dataset['POLARITY SCORE'] = (
            (dataset['POSITIVE SCORE'] - dataset['NEGATIVE SCORE']) / 
            ((dataset['POSITIVE SCORE'] + dataset['NEGATIVE SCORE']) + 0.000001)
        ).round(2)
        
        dataset['SUBJECTIVITY SCORE'] = (
            (dataset['POSITIVE SCORE'] + dataset['NEGATIVE SCORE']) /
            ((dataset['content_tokenized'].apply(len)) + 0.000001)
        ).round(2)

        # Analysis of Readability
        # Average Sentence Length (ASL)
    
        dataset['AVG SENTENCE LENGTH'] = (dataset['content_tokenized'].apply(len) / dataset['content_cleaned'].apply(len)).round(1)

        # Complex Words
        dataset['COMPLEX WORDS'] = dataset['content_tokenized'].apply(
            lambda row: [word for word in row if syllapy.count(word) > 2]
        )
        # Percentage of Complex words 
        dataset['PERCENTAGE OF COMPLEX WORDS'] = ((dataset['COMPLEX WORDS'].apply(len) / dataset['content_tokenized'].apply(len)) * 100.0).round(2)
        
        # Fog Index 
        dataset['FOG INDEX'] = (0.4 * (dataset['AVG SENTENCE LENGTH'] + dataset['PERCENTAGE OF COMPLEX WORDS'])).round(2)
        
        # Average Number of Words Per Sentence
        dataset['AVG NUMBER OF WORDS PER SENTENCE'] = dataset['AVG SENTENCE LENGTH']

        # Calculating Complex Word Count
        dataset['COMPLEX WORD COUNT'] = dataset['COMPLEX WORDS'].apply(len)

        # Word Count
        StopWords = set(stopwords.words('english'))

        dataset['WORD COUNT'] = dataset['content_tokenized'].apply( 
            lambda row: len([word for word in row if word not in StopWords])
        )

        # Syllable Count Per Word
        def count_syllables(word):
            word = word.lower()  # Convert word to lowercase for consistency
            vowels = "aeiou"
            syllable_count = 0
            
            # Iterate over each character in the word
            for char in word:
                if char in vowels:
                    syllable_count += 1
            
            # Handle exceptions like words ending with "es" or "ed"
            if word.endswith("es") or word.endswith("ed"):
                syllable_count -= 1
            
            return syllable_count  # Ensure at least 1 syllable for non-empty words

        # Average | syllables Count for each word
        dataset['SYLLABLE PER WORD'] = dataset['content_tokenized'].apply(
            lambda row: round(sum(count_syllables(word) for word in row) /
            len(row), 2)
        )

        # Personal Pronouns
        personal_pronouns = (
            "I", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
            "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs"
        )

        pattern = r'\b(' + '|'.join(personal_pronouns) + r')\b'

        dataset['PERSONAL PRONOUNS'] = dataset['content'].apply(lambda x: re.findall(pattern, x.lower())).apply(len)

        # Average Word Length
        dataset['AVG WORD LENGTH'] = dataset['content_tokenized'].apply(lambda x: round(sum(len(word) for word in x) / len(x), 1) if len(x) > 0 else 0.0)

        # URL and Output Formatting
        dataset.rename(columns={'filename': 'URL_ID'}, inplace=True)
        dataset.loc[:, 'URL_ID'] = dataset['URL_ID'].str.replace('.txt', '')

        dataset2 = pd.read_excel('data/Output Data Structure.xlsx')
        dataset.loc[:, 'URL'] = dataset2['URL']

        dataset = dataset[['URL_ID', 'URL', 'POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
                    'PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT',
                    'SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']]

        return dataset.to_excel('data/Output_file.xlsx', index=False)



