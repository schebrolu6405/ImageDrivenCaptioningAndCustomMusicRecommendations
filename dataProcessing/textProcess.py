from datetime import datetime

import pandas as pd
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from common import CommonModule
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")

# Configuration variables
CSV_FILE_PATH = 'C:/Users/sushe/Documents/BDA_Fall_2023/Sem1/BDA_696_Python/Project/sample_text.csv'
BASE_URL1 = "https://www.lifewire.com"
BASE_URL2 = "https://www.oberlo.com"
QUOTES_CLASS = "comp mntl-sc-block mntl-sc-block-html"
URL1 = f"{BASE_URL1}/best-instagram-captions-4171697"
URL2 = f"{BASE_URL2}/blog/instagram-captions"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.8",
}
quotes = []
quotes1 = []


class TextModule:
    @staticmethod
    def scrape_quotes_from_website1(url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            ul_elements = soup.find_all('ul', class_=QUOTES_CLASS)
            for u in ul_elements:
                li_tags = u.find_all('li')
                quotes.extend(li.text for li in li_tags)
        return quotes

    @staticmethod
    def scrape_quotes_from_website1(url):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            ul_elements = soup.find_all('ul', class_=QUOTES_CLASS)
            for u in ul_elements:
                li_tags = u.find_all('li')
                quotes.extend(li.text for li in li_tags)
        return quotes

    @staticmethod
    def scrape_quotes_from_website2(url):
        response = requests.get(url)
        response.status_code
        content = BeautifulSoup(response.text, 'html.parser')
        for i in content.find_all("li", style="font-weight: 400;"):
            quotes.append(i.get_text())
        return quotes

    @staticmethod
    def get_next_page_url(soup):
        next_page_url = soup.find('a', class_='next_page')
        if next_page_url:
            return next_page_url['href']
        else:
            return None

    @staticmethod
    def scrape_quotes_from_whole_website():
        url = "https://www.goodreads.com/quotes/tag/free?page="
        while True:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            quote_container = soup.find_all('div', class_='quote')
            for quote in quote_container:
                text = quote.find('div', class_='quoteText').text.strip()
                quotes1.append(text)
            next_page_url = TextModule.get_next_page_url(soup)
            if not next_page_url:
                break
            else:
                url = "https://www.goodreads.com" + next_page_url
        return quotes1

    @staticmethod
    def process_and_print_quotes(quotes, text_data):
        matching_quotes = []
        if isinstance(text_data, str):
            words = [word for word, tag in pos_tag(word_tokenize(text_data)) if tag.startswith(('VB', 'NN', 'JJ'))]
            keyword_list = list(set(words))
            synonyms = set()
            for word in keyword_list:
                for syn in wordnet.synsets(word):
                    for lm in syn.lemmas():
                        synonyms.add(lm.name().lower())
            matching_quotes = [quote for quote in quotes if any(keyword in quote for keyword in synonyms)]
            if len(matching_quotes) == 0:
                matching_quotes = quotes

        val = []
        for quote in matching_quotes:
            cosine_sim = CommonModule.similarity_score(text_data.strip(), quote.strip())
            val.append((cosine_sim, quote.strip(), text_data.strip()))
        sorted_val_desc = sorted(val, key=lambda x: x[0], reverse=True)
        return (([i[1].replace("\n", '').replace("  ", '') for i in sorted_val_desc[:4]], synonyms))

    @staticmethod
    def hashtag_generator(caption):
        inputs = tokenizer([caption], max_length=1024, truncation=True, return_tensors="pt")
        all_tags = []
        for i in range(1, 6):
            output = model.generate(**inputs, num_beams=i, do_sample=True, min_length=4, max_length=100)
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            tags = list(set(decoded_output.strip().split(", ")))
            all_tags.extend(tags)  # Append tags from each iteration to the list
        return set(all_tags)


def text_process(caption):
    caption = caption.replace('startseq','')

    '''quotes = scrape_quotes_from_website1(URL1)#scrape from lifewire website
    quotes = scrape_quotes_from_website2(URL2)#scrape from oberlo website
    quotes1 = scrape_quotes_from_whole_website()'''

    '''df_quotes = pd.DataFrame({'Quotes': quotes})
    df_quotes1 = pd.DataFrame({'Quotes': quotes1})

    output_file_path = "C:/Users/sushe/Documents/BDA_Fall_2023/quotes_output1.csv"
    df_quotes.to_csv(output_file_path, sep='|', header=True, index=False)

    output_file_path1 = "C:/Users/sushe/Documents/BDA_Fall_2023/quotes_output2.csv"
    df_quotes1.to_csv(output_file_path1, sep='|', header=True, index=False)'''

    fquotes = pd.read_csv("/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/datasets/quotes_output1.csv", sep='|', header='infer',skipinitialspace=True)
    fquotes_dt = fquotes['Quotes']
    fquotes1 = pd.read_csv("/Users/Manisha/Documents/MS/SDSU/course/BDA-696/final_project/project/ImageDrivenCaptioningAndCustomMusicRecommendations/resources/datasets/quotes_output2.csv", sep='|', header='infer',skipinitialspace=True)
    fquotes_dt1 = fquotes1['Quotes']

    captions1 = TextModule.process_and_print_quotes(fquotes_dt, caption)
    captions2 = TextModule.process_and_print_quotes(fquotes_dt1, caption)
    img_captions = captions1[0] + captions2[0]
    img_synonyms = captions1[1].union(captions2[1])
    img_hashtag = TextModule.hashtag_generator(caption)
    hashtags = img_synonyms.union(img_hashtag)
    return ((img_captions[:5], list(hashtags)))

# print(datetime.now())
# print(text_process("fishing fishing is through a boat on the water on the boat on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on the water on"))
# print(datetime.now())
