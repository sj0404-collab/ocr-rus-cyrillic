from ocr_rus_cyrillic.corrector import normalize_russian_text


def test_confusable_latin_is_converted():
    assert normalize_russian_text("TeKcTa", allow_dictionary=True) == "Текста"
    assert normalize_russian_text("Pyсckого", allow_dictionary=True) == "Русского"


def test_one_edit_correction():
    assert normalize_russian_text("роверка", allow_dictionary=True) == "проверка"
    assert normalize_russian_text("ешё", allow_dictionary=True) == "ещё"
