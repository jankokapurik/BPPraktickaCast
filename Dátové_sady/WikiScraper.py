import re
import wikipediaapi
from tqdm import tqdm  # Pre vizuálny pokrok

# Načítanie slov zo súboru
with open("./Dátové_sady/Whitelist.txt", "r", encoding="utf-8") as file:
    words = [line.strip() for line in file.readlines() if line.strip()]

# Zadajte User-Agent
user_agent = "Wikiscraper/1.0(jankokapurik@gmail.com)"
wiki_wiki = wikipediaapi.Wikipedia(language='sk', user_agent=user_agent)

def get_sentences_with_word(word):
    page = wiki_wiki.page(word)
    if page.exists():
        # Použijeme regulárne výrazy na rozdelenie textu na vety
        sentences = re.split(r'(?<=[.!?]) +', page.text)  # Delíme podľa bodky, výkričníka alebo otázniku
        filtered = [s.strip() for s in sentences if word.lower() in s.lower()]
        return filtered
    return []

output_file = "wikipedia_sentences.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for word in tqdm(words, desc="Spracovávanie slov", unit="slovo"):  # Pokrok pre každé slovo
        sentences = get_sentences_with_word(word)
        if sentences:
            f.write(f"\n=== {word.upper()} ===\n")
            for sentence in sentences:
                f.write(sentence + "\n")

print(f"Uložené vety do {output_file}")
