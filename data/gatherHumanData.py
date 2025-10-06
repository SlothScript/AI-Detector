import requests
from bs4 import BeautifulSoup
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def get_sentences_from_html(url, max_sentences=8000):
    response = requests.get(url)
    response.encoding = 'utf-8'  # Ensure proper text decoding
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Try different containers for the main text
    text_blocks = []

    # 1️⃣ Try <p> tags first
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    if len(paragraphs) > 10:
        text_blocks = paragraphs
    else:
        # 2️⃣ Try <div> blocks if <p> tags were too few
        divs = [div.get_text() for div in soup.find_all('div')]
        if len(divs) > 10:
            text_blocks = divs
        else:
            # 3️⃣ Fallback: use the whole text content
            text_blocks = [soup.get_text()]

    text = ' '.join(text_blocks)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Project Gutenberg.*?End of the Project Gutenberg', '', text, flags=re.DOTALL)  # remove license text

    sentences = sent_tokenize(text)
    return sentences[:max_sentences]

if __name__ == "__main__":
    urls = [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.html",     # Pride and Prejudice
        "https://www.gutenberg.org/cache/epub/11/pg11-images.html",  # Alice's Adventures in Wonderland
        "https://www.gutenberg.org/cache/epub/84/pg84-images.html",  # Frankenstein
        "https://www.gutenberg.org/cache/epub/1661/pg1661.html",     # Sherlock Holmes
    ]

    all_sentences = []
    for url in urls:
        sentences = get_sentences_from_html(url, 6000)
        print(f"Extracted {len(sentences)} from {url}")
        all_sentences.extend(sentences)

    print(f"✅ Total sentences collected: {len(all_sentences)}")

    with open("humanData.txt", "a", encoding="utf-8") as f:
        for sentence in all_sentences[:9000]:
            f.write(sentence.strip() + "\n")

    print("✅ Data extraction complete.")
