def load_corpus(filepath):
    """Load a plain text corpus file.
    Each non-empty line is treated as one document.
    Returns a list of document strings.
    """
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
