# Import necessary modules from the 'article_scraper_nlp' package (.py file)
from scripts.article_scraper_nlp import ArticleScraper, TextPreprocessing
import os


def main():
    # Input file path for scraping articles and log file location
    input_file_path = "-------"
    log_file = "-------"

    # Create the 'logs' directory if it doesn't exist
    log_dir = os.path.dirname(log_file) 
    if not os.path.exists(log_dir):  
        os.makedirs(log_dir)  # Create the directory if it doesn't exist
    if not os.path.exists('data/StopWords'):
        os.makedirs('data/StopWords')

    # Initialize the ArticleScraper with input file and log file and call the method to scrape all articles
    scrapper = ArticleScraper(input_file=input_file_path, log_file=log_file)
    scrapper.scrape_all_articles()

    # Paths for stop words file, stop words folder, and folder to save scraped text files
    stop_words_out = '---/combined_stop_words.txt'
    stp_file_path = '----/StopWords'
    txt_file_path = 'scraped_txt_files'
    zip_path_masterDir = '----/MasterDictionary-001.zip' 

    # Initialize the TextPreprocessing class for text data processing
    preprocessor = TextPreprocessing()    

    # Get the stop words and cleaned data by calling the 'get_stopwords_get_data' method, This function loads the stop words and cleans the text data
    cleaned_data = preprocessor.get_stopwords_get_data(text_file_path=txt_file_path, stop_words_out=stop_words_out, stp_file_path=stp_file_path)

    # Preprocess and analyze the cleaned data, which includes tokenization and other analyses
    dataset = preprocessor.preprocess_analyze(cleaned_data)

    # Extract master dictionary files (positive/negative word lists) from the ZIP file
    preprocessor.get_masterdic_files(zip_file_masterDir=zip_path_masterDir)

    # Compute various text analysis variables, such as positive/negative score, readability, etc.
    preprocessor.compute_variables(dataset)


# Entry point of the script
if __name__ == "__main__":
    main()


