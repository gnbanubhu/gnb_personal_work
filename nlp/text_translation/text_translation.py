from transformers import pipeline


def get_translator(src_lang: str, tgt_lang: str):
    model_map = {
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
        ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
        ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    }
    key = (src_lang, tgt_lang)
    if key not in model_map:
        raise ValueError(f"Translation from '{src_lang}' to '{tgt_lang}' is not supported.")
    return pipeline("translation", model=model_map[key])


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    translator = get_translator(src_lang, tgt_lang)
    result = translator(text)
    return result[0]["translation_text"]


def print_translation(text: str, translated: str, src_lang: str, tgt_lang: str) -> None:
    print(f"Source ({src_lang.upper()})     : {text}")
    print(f"Translated ({tgt_lang.upper()}) : {translated}")
    print("-" * 65)


if __name__ == "__main__":

    sample_texts = [
        "Machine learning is transforming the world of technology.",
        "Artificial intelligence helps automate complex tasks efficiently.",
        "Data engineering is the backbone of every AI application.",
    ]

    translations = [
        ("en", "fr", "French"),
        ("en", "de", "German"),
        ("en", "es", "Spanish"),
        ("en", "it", "Italian"),
        ("en", "zh", "Chinese"),
    ]

    print("=" * 65)
    print("            TEXT TRANSLATION RESULTS")
    print("=" * 65)
    print()

    for src_lang, tgt_lang, lang_name in translations:
        print(f"Translating English --> {lang_name}")
        print("=" * 65)
        for text in sample_texts:
            translated = translate(text, src_lang, tgt_lang)
            print_translation(text, translated, src_lang, tgt_lang)
        print()
