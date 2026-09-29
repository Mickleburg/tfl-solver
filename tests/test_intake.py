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
    load_seminar_index,
    suggest_micro_methods,
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


def test_micro_methods_describe_small_steps_before_theorem():
    evidence = find_features(DESCRIPTION) + find_asks(DESCRIPTION)
    methods = suggest_micro_methods(evidence)
    assert "расставить скобки в логическом условии" in methods
    assert "разделить конъюнкции и дизъюнкции на случаи" in methods


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


def test_current_rk_hint_accepts_the_whole_known_bank():
    assert classify(ATTRIBUTE, "РК")[0].code == "RK2-C"
    assert classify(DESCRIPTION, "РК")[0].code.startswith("RK")


def test_current_lab_number_does_not_force_an_old_recipe():
    text = "Построить Generic GLR-разбор с графовидным стеком и лесом разбора."
    assert classify(text, "ЛР1")[0].code == "LAB-5"


def test_exam_error_is_a_separate_route():
    candidates = classify("Найти первую ошибку в готовом решении задачи", "экзамен")
    assert candidates[0].code == "EXAM-ERROR"


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


def test_training_analysis_prioritizes_a_verified_seminar(index):
    seminar = load_seminar_index()
    result = analyse(
        "Построить инъективный морфизм с неограниченной задержкой раскодирования",
        "семинар",
        index,
        seminar_index=seminar,
    )
    assert result.mode == "training"
    assert result.seminar_similar
    assert result.seminar_similar[0][1]["id"] == "sem-2026-09-05-unbounded-delay"
    report = result.report()
    assert "Сначала проверить аналоги" in report
    assert "Путь открытия" in report
    assert "префиксность не годится" in report


def test_short_concrete_statement_still_finds_the_delay_seminar():
    result = analyse(
        "Дан морфизм h: {x,y,z}* -> {0,1}*: h(x)=0, h(y)=01, h(z)=11. "
        "Докажите, что он инъективен, и выясните, ограничена ли задержка "
        "раскодирования.",
        "семинар",
        seminar_index=load_seminar_index(),
    )
    assert result.seminar_similar
    assert result.seminar_similar[0][1]["id"] == "sem-2026-09-05-unbounded-delay"
    briefing = result.solver_briefing()
    assert "Малые действия до основного метода" not in briefing
    assert "алгоритмом Сардинаса–Паттерсона" in briefing
    assert "xz^k" in briefing and "yz^(k-1)" in briefing
    assert "общий префикс" in briefing
    assert "python3 -c" in briefing


def test_route_filter_does_not_force_a_seminar_analog():
    result = analyse(
        "Является ли регулярным язык L={a^n b^m c^k | n,m,k>=0 ∧ "
        "(m!=n ∨ k mod 2 != n mod 2 ∨ k mod 2 = m mod 2)}.",
        "РК1",
        seminar_index=load_seminar_index(),
    )
    assert not result.seminar_similar
    briefing = result.solver_briefing()
    assert "Проверенного семинарного аналога не найдено" in briefing
    assert "Малые действия до основного метода" in briefing


def test_delay_card_does_not_capture_a_plain_injectivity_question():
    result = analyse(
        "Коды символов заданы так: h(S)=12, h(b)=1, h(c)=2. Образы разных "
        "символов попарно различны. Следует ли отсюда инъективность морфизма "
        "на всех строках? Перечислите все естественные декодирования строки 12.",
        "семинар",
        mode="full",
        seminar_index=load_seminar_index(),
    )
    assert result.seminar_similar
    assert result.seminar_similar[0][1]["id"] == "sem-2026-09-05-decode-candidate"


def test_verified_seminars_store_reproducible_discovery_paths():
    records = load_seminar_index().records
    assert records
    assert all(record["routes"] for record in records)
    assert all(
        isinstance(record.get("required_patterns", []), list) for record in records
    )
    assert all(len(record["discovery_path"]) >= 2 for record in records)
    assert all("→" in step for record in records for step in record["discovery_path"])


def test_pattern_cards_contain_executable_checks_and_correct_proof_directions():
    records = {record["id"]: record for record in load_seminar_index().records}
    normal = records["sem-2026-09-05-pattern-normal-form"]
    brackets = records["sem-2026-09-05-bracket-deletion"]
    assert "python3 -c" in normal["verification"]
    assert "сумма позиций b" in normal["method"]
    assert "единственную нормальную форму" in normal["method"]
    assert "ab -> (3,[aaa])" in normal["verification"]
    assert "ровно на |X|+1" in normal["verification"]
    assert "длины 1–3 (не 1–4)" in normal["verification"]
    assert any("-2+2=0" in step for step in normal["discovery_path"])
    assert "python3 -c" in brackets["verification"]
    assert "оборачивает существующее подслово X" in brackets["method"]
    assert any(
        "произвольное удаление может нарушить правильность" in step
        for step in brackets["discovery_path"]
    )


def test_encoding_cards_contain_executable_checks_and_general_arguments():
    records = {record["id"]: record for record in load_seminar_index().records}
    rotation = records["sem-2026-09-05-rotation"]
    decoding = records["sem-2026-09-05-decoding-srs"]

    assert "python3 -c" in rotation["verification"]
    assert "D(vx)=xv" in rotation["method"]
    assert "обратная функция, а не морфизм" in rotation["method"]
    assert "E(ab)=ba" in rotation["method"]
    assert "ab^n" in rotation["method"]
    assert "общий вывод об инъективности даёт явная обратная функция" in rotation["verification"]
    assert "E(wx)=xw означала бы другой" in rotation["verification"]
    assert "лемма об общем префиксе не нужна" in rotation["verification"]
    assert len(rotation["required_patterns"]) == 2

    assert "python3 -c" in decoding["verification"]
    assert "ada" in decoding["method"]
    assert "a) -> ()" in decoding["method"]
    assert "d( -> )(" in decoding["method"]
    assert "a$ -> ($" in decoding["method"]
    assert "d$ -> )$" in decoding["method"]
    assert "ada_nfs" in decoding["verification"]
    assert "ad_end_nfs" in decoding["verification"]
    assert "end_local" in decoding["verification"]
    assert "chr(36)+'()abcd'" in decoding["verification"]
    assert "'$()abcd'" not in decoding["verification"]
    assert "критических пар" in decoding["verification"]
    assert "все 10" in decoding["verification"]
    assert "не перебор входных слов" in decoding["verification"]
    assert len(decoding["required_patterns"]) == 2


def test_full_mode_is_reported(index):
    result = analyse("Проверить язык на регулярность", index=index, mode="full")
    assert "Режим решения: full" in result.report()


def test_unknown_solver_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        analyse("задача", mode="unknown")


def test_confidence_requires_a_gap():
    clear = Analysis("", None, (), (), (Candidate("RK2-C", 10), Candidate("RK2-A", 2)))
    tie = Analysis("", None, (), (), (Candidate("RK1-B", 5), Candidate("RK2-B", 4)))
    assert clear.confident and not tie.confident


def test_report_warns_when_there_is_no_gap():
    tie = Analysis("", None, (), (), (Candidate("RK1-B", 5), Candidate("RK2-B", 4)))
    assert "Отрыва у лидера нет" in tie.report()


def test_attribute_grammar_written_with_plain_equals():
    """В условии присваивание может быть записано через `=`, а не `:=`.

    Ссылка на атрибут `S2.attr` — сама по себе достаточный признак,
    Иначе условие после OCR не опознаётся.
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
