"""Тесты приёма задачи: признаки, классификация, поиск похожих условий.

Точность измеряется на настоящих условиях из `evals/tasks/index.jsonl`,
причём **год выпуска разделяет выборку**: веса признаков подбирались по
задачам 2021–2024, а 2025 остаётся контрольным. Порог в тестах поставлен
ниже измеренного, чтобы ловить регресс, а не дрожание.

Оговорка, без которой цифра вводит в заблуждение: условия РК2 2025 я
читал при разработке, так что «контрольный» год не является полностью
незнакомым. Честная оценка обобщения — на задачах 2026, когда они выйдут.

Мерить точность можно только там, где разметка содержательная
(`label == "проверено"`). Позиционная разметка экзамена содержательной
не является: третьим вопросом билета идёт и теорема о замкнутости,
и «построить синтаксический моноид» — измерять на ней значит измерять шум.
"""

from __future__ import annotations

import pytest

from tfl.intake import (
    Analysis,
    Candidate,
    analyse,
    classify,
    find_asks,
    find_features,
    load_index,
)

ATTRIBUTE = """Язык атрибутной грамматики:
S →S′ ; S′.a > S′.b
S′ →TS′ ; S′0.a := T.a + S′1.a, S′0.b := max(T.b, S′1.b)
T →ε ; T.a := 0, T.b := 0"""

SRS_BASIS = "Язык SRS с правилами aa →aba, ba →abb, базис (ab)∗."

DESCRIPTION = "Язык {w1w2w3 | wi ∈{a, b}+ & w1 = w3 ∨ |w2|a = |w3|a}."


@pytest.fixture(scope="module")
def index():
    return load_index()


@pytest.fixture(scope="module")
def content(index):
    """Задачи с проверенной, содержательной разметкой."""
    return [r for r in index.records if r.get("label") == "проверено" and r["code"]]


def names(evidence) -> set[str]:
    return {e.name for e in evidence}


# --------------------------------------------------------------------------
# Признаки
# --------------------------------------------------------------------------


def test_attribute_grammar_is_recognized():
    assert "атрибутная грамматика" in names(find_features(ATTRIBUTE))


def test_srs_and_basis_are_recognized():
    found = names(find_features(SRS_BASIS))
    assert "правила SRS" in found and "базис" in found


def test_grammar_rules_are_not_mistaken_for_srs():
    """`S → aSb` — правило грамматики, а не переписывания."""
    found = names(find_features("S →aSb | ε"))
    assert "КС-грамматика" in found and "правила SRS" not in found


def test_description_recognized_without_braces():
    """Регрессия: у описаний языка в извлечённом тексте пропадают скобки.

    В LaTeX внешние `{}` набраны крупными разделителями и приходят
    управляющими символами. Из-за этого 46 задач РК2 не опознавались
    вовсе — описание пришлось опознавать ещё и по начинке.
    """
    mangled = "Язык\n\x1a\nw1anb∗cn+kw2\n\x0c w1, w2 ∈{a, b}+ & |w1| = |w2|\n\x1b\n."
    found = names(find_features(mangled))
    assert "принадлежность алфавиту" in found
    assert classify(mangled, "РК2")[0].code == "RK2-B"


def test_asks_are_separate_from_features():
    assert "регулярность" in names(find_asks("Проверить язык на регулярность"))
    assert "регулярность" not in names(find_features("Проверить язык на регулярность"))


def test_attribute_grammar_marker_is_precise(content):
    """`:=` не встречается ни в одном другом жанре — проверяем на всём индексе."""
    wrong = [
        r["id"]
        for r in content
        if ("атрибутная грамматика" in names(find_features(r["text"])))
        != (r["code"] == "RK2-C")
    ]
    assert len(wrong) <= 2, f"признак сработал не там: {wrong[:5]}"


# --------------------------------------------------------------------------
# Классификация
# --------------------------------------------------------------------------


def test_attribute_grammar_task_classified():
    assert classify(ATTRIBUTE, "РК2")[0].code == "RK2-C"


def test_srs_basis_task_classified():
    assert classify(SRS_BASIS, "РК2")[0].code == "RK2-A"


def test_description_task_classified():
    assert classify(DESCRIPTION, "РК2")[0].code == "RK2-B"


def test_hint_restricts_the_pool():
    """Одно и то же описание языка в РК1 и в РК2 — разные классы, и это норма."""
    assert classify(DESCRIPTION, "РК1")[0].code.startswith("RK1")
    assert classify(DESCRIPTION, "РК2")[0].code.startswith("RK2")


def test_lab3_wording_classified():
    text = ("Проанализировать язык на детерминизм. Построить PDA. "
            "Проанализировать язык на беспрефиксность.")
    assert classify(text)[0].code == "LAB-3"


def test_no_features_gives_no_candidates():
    assert classify("Здравствуйте, помогите пожалуйста с домашкой") == []


def test_accuracy_on_held_out_year(content):
    """Контрольный год: веса подбирались на 2021–2024."""
    rows = [r for r in content if r["year"] == 2025]
    hits = sum(
        bool(c := classify(r["text"], r["form"], 1)) and c[0].code == r["code"]
        for r in rows
    )
    assert hits / len(rows) >= 0.95, f"{hits}/{len(rows)}"


def test_accuracy_on_tuning_years(content):
    rows = [r for r in content if r["year"] <= 2024]
    hits = sum(
        bool(c := classify(r["text"], r["form"], 1)) and c[0].code == r["code"]
        for r in rows
    )
    assert hits / len(rows) >= 0.90, f"{hits}/{len(rows)}"


def test_accuracy_without_a_hint_is_lower_but_usable(content):
    """Без пометки формы часть классов неразличима — и так и должно быть.

    `RK1-B` и `RK2-B` — оба «описание языка множеством»; отличает их
    только то, на какой работе задача выдана. Угадывать здесь нечего.
    """
    hits = sum(
        bool(c := classify(r["text"], None, 1)) and c[0].code == r["code"]
        for r in content
    )
    assert hits / len(content) >= 0.80, f"{hits}/{len(content)}"


# --------------------------------------------------------------------------
# Поиск похожих
# --------------------------------------------------------------------------


def test_index_is_loaded(index):
    assert len(index) > 1000


def test_verified_labels_always_have_a_code(index):
    assert all(r["code"] for r in index.records if r.get("label") == "проверено")


def test_exam_labels_are_marked_positional(index):
    exams = [r for r in index.records if r["form"] == "экзамен"]
    assert exams and all(r["label"] == "позиционно" for r in exams)


def test_similar_finds_the_same_task_in_other_years(index):
    """Задача РК2 2025 должна найти свой аналог 2023–2024 года.

    Это и есть смысл поиска: у прошлогоднего аналога может быть авторский
    разбор, а формулировка задачи из года в год почти не меняется.
    """
    task = next(r for r in index.records if r["id"] == "rk2-2025-v01-t3")
    others = [
        hit for _, hit in index.similar(task["text"], limit=40, exclude=task["id"])
        if hit["source"] != task["source"]
    ]
    assert others[0]["code"] == "RK2-C"
    assert others[0]["year"] != 2025


def test_retrieval_keeps_the_class(index, content):
    rows = [r for r in content if r["year"] == 2025 and r["form"] == "РК2"]
    hits = 0
    for r in rows:
        others = [
            hit for _, hit in index.similar(r["text"], limit=40, exclude=r["id"])
            if hit["source"] != r["source"]
        ]
        hits += bool(others) and others[0]["code"] == r["code"]
    assert hits / len(rows) >= 0.85, f"{hits}/{len(rows)}"


def test_similar_can_be_restricted_to_a_form(index):
    hits = index.similar(DESCRIPTION, limit=5, form="Аптека")
    assert hits and all(hit["form"] == "Аптека" for _, hit in hits)


# --------------------------------------------------------------------------
# Сводный разбор
# --------------------------------------------------------------------------


def test_analysis_reports_evidence(index):
    result = analyse(ATTRIBUTE, "РК2", index)
    text = result.report()
    assert "RK2-C" in text and "атрибутная грамматика" in text
    assert "Похожие условия" in text


def test_confidence_requires_a_gap():
    clear = Analysis("", None, (), (), (Candidate("RK2-C", 10), Candidate("RK2-A", 2)))
    tie = Analysis("", None, (), (), (Candidate("RK1-B", 5), Candidate("RK2-B", 4)))
    assert clear.confident and not tie.confident


def test_report_warns_when_there_is_no_gap():
    tie = Analysis("", None, (), (), (Candidate("RK1-B", 5), Candidate("RK2-B", 4)))
    assert "Отрыва у лидера нет" in tie.report()


def test_attribute_grammar_written_with_plain_equals():
    """В работах присваивание пишут через `=`, а не `:=` (photo_120).

    Ссылка на атрибут `S2.attr` — сама по себе достаточный признак,
    иначе условие, переписанное с фотографии, не опознаётся.
    """
    text = (
        "Язык, определяемый следующей атрибутной грамматикой:\n"
        "S -> S S ; S2.attr < S1.attr, S0.attr = S1.attr - S2.attr\n"
        "S -> b A ; S.attr = A.attr"
    )
    assert "атрибутная грамматика" in {e.name for e in find_features(text)}
    assert classify(text, limit=1)[0].code == "RK2-C"


def test_mu_expression_is_not_an_attribute_grammar():
    """`µY.bX` из билета — точка там значит связывание, а не атрибут."""
    text = "Описать язык µ-выражения: µX.(a(µY.bX|Y a|(µZ.ZZ|cc))bX|ε)."
    assert "атрибутная грамматика" not in {e.name for e in find_features(text)}


def test_rk2_asks_to_describe_the_language():
    """Вопрос в РК2 не задан глаголом — условие это именная группа."""
    for text in (
        "Язык SRS ba2 -> ba, ab -> ba над базисом a^n b^n a^n.",
        "Язык {w1 (ab)* b+ w2 | w1, w2 из (abb|ba)+}.",
        "Язык, определяемый следующей атрибутной грамматикой:",
    ):
        assert "описать язык" in {e.name for e in find_asks(text)}, text
