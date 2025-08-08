import spacy



lyrics="""
Girl, crowded town. Silent bed, pick a place to rest your head. Give me a minute to hold my girl. Give me a minute to hold my girl. I've been dreaming about us, working hard and saving it up. We'll go and see the man on the moon. My girl, we've got nothing to lose. Cold nights and the Sunday mornings. On your way, out of the grey. I've got time, I've got love. Got confidence to rise above. Give me a minute to hold my girl. Give me a minute to hold my girl. Crowded town
"""

# Initialize a dictionary to hold loaded spaCy models
# This ensures models are loaded only once
nlp_models = {}

def load_spacy_model(lang_code):
    if lang_code not in nlp_models:
        try:
            # Use a larger model if 'sm' (small) isn't sufficient for complex cases
            # For most sentence segmentation, 'sm' models are usually fine.
            # Consider 'md' or 'lg' if you notice poor segmentation for certain languages.
            model_name = {
                # Germanic languages
                'eng': 'en_core_web_sm',
                'deu': 'de_core_news_sm',
                'nld': 'nl_core_news_sm',  # Dutch - will need to download
                
                # Romance languages
                'fra': 'fr_core_news_sm',
                'spa': 'es_core_news_sm',
                'por': 'pt_core_news_sm',  # Portuguese - will need to download
                'ita': 'it_core_news_sm',  # Italian - will need to download
                'ron': 'ro_core_news_sm',  # Romanian - will need to download
                
                # Slavic languages
                'pol': 'pl_core_news_sm',  # Polish - will need to download
                'rus': 'ru_core_news_sm',  # Russian - will need to download
                'ces': 'cs_core_news_sm',  # Czech - will need to download
                'slk': 'sk_core_news_sm',  # Slovak - will need to download
                'hrv': 'hr_core_news_sm',  # Croatian - will need to download
                'slv': 'sl_core_news_sm',  # Slovenian - will need to download
                'bul': 'bg_core_news_sm',  # Bulgarian - will need to download
                'srp': 'sr_core_news_sm',  # Serbian - will need to download
                'bos': 'bs_core_news_sm',  # Bosnian - will need to download
                
                # Asian languages
                'jpn': 'ja_core_news_sm',
                'zho': 'zh_core_web_sm',  # Chinese - will need to download
                'kor': 'ko_core_news_sm',  # Korean - will need to download
                'hin': 'hi_core_news_sm',  # Hindi - will need to download
                'tur': 'tr_core_news_sm',  # Turkish - will need to download
                'vie': 'vi_core_news_sm',  # Vietnamese - will need to download
                'ind': 'id_core_news_sm',  # Indonesian - will need to download
                'msa': 'ms_core_news_sm',  # Malay - will need to download
                'tha': 'th_core_news_sm',  # Thai - will need to download
                
                # Other languages
                'ara': 'ar_core_news_sm',  # Arabic - will need to download
                'ell': 'el_core_news_sm',  # Greek - will need to download
                'fin': 'fi_core_news_sm',  # Finnish - will need to download
                'hun': 'hu_core_news_sm',  # Hungarian - will need to download
                'swe': 'sv_core_news_sm',  # Swedish - will need to download
                'dan': 'da_core_news_sm',  # Danish - will need to download
                'nor': 'nb_core_news_sm',  # Norwegian - will need to download
                'isl': 'is_core_news_sm',  # Icelandic - will need to download
                'lit': 'lt_core_news_sm',  # Lithuanian - will need to download
                'lav': 'lv_core_news_sm',  # Latvian - will need to download
                'est': 'et_core_news_sm',  # Estonian - will need to download
                'sqi': 'sq_core_news_sm',  # Albanian - will need to download
                'eus': 'eu_core_news_sm',  # Basque - will need to download
                'cat': 'ca_core_news_sm',  # Catalan - will need to download
                'glg': 'gl_core_news_sm',  # Galician - will need to download
                'epo': 'eo_core_news_sm',  # Esperanto - will need to download
                'sah': 'sah_core_news_sm',  # Yakut - will need to download
                'kaz': 'kk_core_news_sm',  # Kazakh - will need to download
                'ben': 'bn_core_news_sm',  # Bengali - will need to download
                'sh': 'sh_core_news_sm',   # Serbo-Croatian - will need to download
                
                # Fallback for English
                'en': 'en_core_web_sm'
            }.get(lang_code, None)

            if model_name:
                nlp_models[lang_code] = spacy.load(model_name)
            else:
                print(f"Warning: No specific spaCy model defined for language code '{lang_code}'.")
                # You might define a fallback behavior here, e.g.,
                # splitting by common punctuation, though this is less reliable.
                nlp_models[lang_code] = None # Mark as not found to avoid repeated warnings
        except OSError:
            print(f"Error: spaCy model for '{lang_code}' not found. Please download it: python -m spacy download {model_name}")
            nlp_models[lang_code] = None
    return nlp_models.get(lang_code)

def get_sentences(text, language_code):
    """
    Splits text into sentences using the appropriate spaCy model.
    Returns a list of strings, each representing a distinct sentence.
    """
    nlp = load_spacy_model(language_code)
    if nlp is None:
        # Fallback if model not available or not defined
        # This is a very basic fallback and might not be accurate for all languages
        if language_code in ['ja', 'zh', 'ko']: # CJK languages
            return [s.strip() for s in text.split('。') if s.strip()]
        else: # European languages
            return [s.strip() for s in text.split('.') if s.strip()] # Simple period split

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


if __name__ == "__main__":
    print(get_sentences(lyrics, 'en'))

# Example usage for your database processing loop:
# Assuming your database records are like {'lyrics': '...', 'language': 'ja'}
# for row in database_records: # Replace with your actual database query loop
#     snippet_lyrics = row['lyrics']
#     snippet_language = row['language']
#     distinct_lines = get_sentences(snippet_lyrics, snippet_language)
#     # Now you have distinct_lines for the current snippet
#     # Store these lines and their original snippet_id/song_id along with the language
#     # for later embedding.