def clean_text(text):
    if text is None:
        return ""
    return text.lower().strip()
