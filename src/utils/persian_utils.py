from src.core.constants import PERSIAN_DIGITS, ARABIC_DIGITS

def normalize_persian_text(text: str) -> str:
    text = text.replace('ي', 'ی').replace('ك', 'ک')
    trans = str.maketrans(PERSIAN_DIGITS, ARABIC_DIGITS)
    return text.translate(trans)

def persian_to_english_digits(text: str) -> str:
    trans = str.maketrans(PERSIAN_DIGITS, ARABIC_DIGITS)
    return text.translate(trans)

def english_to_persian_digits(text: str) -> str:
    trans = str.maketrans(ARABIC_DIGITS, PERSIAN_DIGITS)
    return text.translate(trans)