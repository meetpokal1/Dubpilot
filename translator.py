"""Translates transcript segments into English.

Uses IndicTrans2 for Indian-language sources when it's installed (better
quality on Hindi/Marathi/Tamil/etc than a general translator), and falls
back to deep-translator (a free wrapper around Google Translate, no API
key needed) for everything else. This keeps the pipeline usable out of
the box while still giving the quality bump the assignment hints at for
Indian languages.

Note: the IndicTrans2 checkpoints on Hugging Face are gated — you need
to accept the model's terms on huggingface.co and run `huggingface-cli
login` once before this path will work. Without that, the code below
just falls back to deep-translator automatically.
"""

from deep_translator import GoogleTranslator

from utils import log

# Whisper gives ISO 639-1 codes (hi, mr, ta...); IndicTrans2 wants FLORES-200
# style codes (hin_Deva, mar_Deva...). This maps the languages we support.
WHISPER_TO_FLORES = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}

_indictrans_model = None


def _load_indictrans():
    """Lazily load IndicTrans2, only if it's installed and actually needed.
    Returns (tokenizer, model, processor) or False if unavailable.
    """
    global _indictrans_model
    if _indictrans_model is not None:
        return _indictrans_model

    try:
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "ai4bharat/indictrans2-indic-en-1B"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        processor = IndicProcessor(inference=True)

        _indictrans_model = (tokenizer, model, processor, device)
    except Exception as exc:
        log("translate", f"IndicTrans2 unavailable ({exc}), will use fallback translator")
        _indictrans_model = False

    return _indictrans_model


def _translate_with_indictrans(segments, flores_lang, loaded, batch_size=8):
    import torch

    tokenizer, model, processor, device = loaded

    # Long videos can have hundreds of segments — batching keeps memory
    # bounded instead of feeding the whole transcript through at once.
    for i in range(0, len(segments), batch_size):
        chunk = segments[i:i + batch_size]
        texts = [seg["text"] for seg in chunk]

        batch = processor.preprocess_batch(texts, src_lang=flores_lang, tgt_lang="eng_Latn")
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5)

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = processor.postprocess_batch(decoded, lang="eng_Latn")

        for seg, translation in zip(chunk, translations):
            seg["english"] = translation


def _translate_with_google(segments, source_lang):
    google_source = source_lang if source_lang != "unknown" else "auto"
    translator = GoogleTranslator(source=google_source, target="en")

    for seg in segments:
        seg["english"] = translator.translate(seg["text"]) if seg["text"] else ""


def translate_segments(segments, source_lang):
    """Translate every segment's text to English, adding an 'english' key
    to each. Returns the same list for convenience.
    """
    if source_lang == "en":
        log("translate", "source is already English, skipping translation")
        for seg in segments:
            seg["english"] = seg["text"]
        return segments

    flores_lang = WHISPER_TO_FLORES.get(source_lang)
    loaded = _load_indictrans() if flores_lang else False

    if loaded:
        log("translate", f"using IndicTrans2 for '{source_lang}'")
        _translate_with_indictrans(segments, flores_lang, loaded)
    else:
        log("translate", f"using GoogleTranslator for '{source_lang}'")
        _translate_with_google(segments, source_lang)

    log("translate", f"translated {len(segments)} segments")
    return segments
