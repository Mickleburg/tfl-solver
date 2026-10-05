"""Педагогический holdout: режим, аналог и воспроизводимый путь решения."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from tfl.cli import main
from tfl.intake import load_seminar_index
from tfl.pedagogy_eval import (
    CASES_PATH,
    MANIFEST_PATH,
    build_prompt,
    load_cases,
    score_report,
    score_response,
)


def good_response(case):
    assert case.id == "training-unbounded-delay"
    return {
        "case_id": case.id,
        "mode": "training",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-unbounded-delay",
        "discovery_path": [
            {
                "clue": "Морфизм полностью задан словами 0, 01 и 11.",
                "action": "Сравнить кодовые слова на префиксность.",
                "result": "Слово 0 является префиксом 01, поэтому мгновенного декодирования нет.",
                "next_step": "Проверить более слабое свойство однозначной декодируемости.",
            },
            {
                "clue": "Нужно различить прообразы, начинающиеся с x и y.",
                "action": "Продолжить их образами z и поискать растущий общий префикс.",
                "result": "Пары xz^k и yz^(k-1) дают префиксы из всё более длинных серий единиц.",
                "next_step": "Записать длину общего префикса формулой для произвольного k.",
            },
        ],
        "micro_methods": [
            "Сначала отличить префиксность от однозначной декодируемости.",
            "Заменить конечные примеры параметрическим семейством для произвольного k.",
        ],
        "prerequisites_used": ["морфизм слов", "однозначно декодируемый код"],
        "chosen_method": (
            "Проверить код алгоритмом Сардинаса–Паттерсона, затем взять xz^k и "
            "yz^(k-1), образы которых имеют растущий общий префикс."
        ),
        "oracle_calls": [
            {
                "module": "tfl.code",
                "operation": "uniquely_decodable",
                "input": "0, 01, 11",
                "result": "код однозначно декодируем",
            }
        ],
        "final_answer": (
            "Морфизм инъективен, поскольку код {0,01,11} однозначно "
            "декодируем по алгоритму Сардинаса–Паттерсона. Задержка "
            "неограничена: для произвольного k образы xz^k и yz^(k-1) "
            "имеют общий префикс, длина которого растёт вместе с k."
        ),
        "limitations": ["Конечная кривая задержки используется только как иллюстрация."],
    }


def good_pattern_response(case):
    assert case.id == "training-pattern-normal-form"
    return {
        "case_id": case.id,
        "mode": "training",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-pattern-normal-form",
        "discovery_path": [
            {
                "clue": "X — произвольная строковая переменная, а правила переставляют буквы и заменяют b на aa.",
                "action": "Посчитать число b и сумму позиций b.",
                "result": "Лексикографическая пара строго убывает.",
                "next_step": "Найти инвариант и вид неприводимых слов.",
            },
            {
                "clue": "Любое b является редексом Xb при X=epsilon.",
                "action": "Вычислить инвариант |w|_a+2|w|_b.",
                "result": "Нормальная форма единственна и равна a^mu(w).",
                "next_step": "Заключить конфлюэнтность завершимой системы.",
            },
        ],
        "micro_methods": [
            "посчитать изменение числа b",
            "посчитать сумму позиций b",
            "описать неприменимость правил",
        ],
        "prerequisites_used": ["фундированный порядок", "инвариант"],
        "chosen_method": (
            "Проверить оракулом лексикографическую меру, затем использовать "
            "инвариант для единственной нормальной формы."
        ),
        "oracle_calls": [
            {
                "module": "pattern",
                "operation": "check_measure + check_invariant",
                "input": "aXb -> bXa; Xb -> aaX",
                "result": "мера убывает, инвариант сохраняется на проверенном срезе",
            }
        ],
        "final_answer": (
            "Система завершима по лексикографической мере из числа b и суммы "
            "их позиций. Инвариант определяет единственную нормальную форму "
            "a^mu(w), поэтому завершимая система конфлюэнтна."
        ),
        "limitations": ["Конечный запуск проверяет формулы, общее доказательство дано отдельно."],
    }


def good_bracket_response(case):
    assert case.id == "training-bracket-deletion"
    return {
        "case_id": case.id,
        "mode": "training",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-bracket-deletion",
        "discovery_path": [
            {
                "clue": "Нужно доказать две импликации тогда и только тогда.",
                "action": "Успешное удаление открывающей и более поздней закрывающей скобки прочитать назад.",
                "result": "Обратная вставка сохраняет префиксный баланс.",
                "next_step": "Получить необходимость от epsilon к исходному слову.",
            },
            {
                "clue": "В непустой правильной скобочной последовательности есть соседняя пара ().",
                "action": "Удалить её и применить индукцию по длине.",
                "result": "Правильное слово сводится к epsilon.",
                "next_step": "Соединить обе импликации.",
            },
        ],
        "micro_methods": [
            "разделить доказательство на две импликации",
            "прочитать успешную редукцию назад",
            "выбрать параметр индукции — длину",
        ],
        "prerequisites_used": ["префиксный баланс", "индукция"],
        "chosen_method": (
            "Использовать префиксный баланс для обратных вставок и удалять "
            "соседнюю пару () по индукции."
        ),
        "oracle_calls": [
            {
                "module": "pattern",
                "operation": "reachable",
                "input": "(X) -> X",
                "result": "на конечном срезе контрпримеров нет",
            }
        ],
        "final_answer": (
            "Получено равносильное условие: w сводится к epsilon тогда и только "
            "тогда, когда w является правильной скобочной последовательностью. "
            "Необходимость следует из обратных вставок, достаточность — по индукции."
        ),
        "limitations": ["Конечный срез не заменяет две общие импликации."],
    }


def test_pedagogy_holdout_is_frozen_and_has_both_modes():
    cases = load_cases()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(CASES_PATH.read_text(encoding="utf-8").encode()).hexdigest()
    assert manifest["version"] == 2
    assert manifest["rubric_revision"] == 35
    assert manifest["case_count"] == len(cases) == 8
    assert manifest["cases_sha256"] == digest
    assert {case.mode for case in cases} == {"training", "full"}


def test_all_named_seminar_analogs_exist():
    known = {record["id"] for record in load_seminar_index().records}
    assert all(
        not case.seminar_analog or case.seminar_analog in known
        for case in load_cases()
    )


def test_prompt_exposes_precomputed_intake_but_hides_rubric_files():
    for case in load_cases():
        prompt = build_prompt(case)
        assert case.statement in prompt
        assert f"Режим: {case.mode}" in prompt
        assert "Стартовый пакет решателя" in prompt
        assert "не вызывай skill, intake" in prompt
        assert "1. Сверка без инструментов" in prompt
        assert "2. Одна исполняемая проверка" in prompt
        assert "3. Общее доказательство" in prompt
        assert "4. Сериализация" in prompt
        assert "`final_answer` и `chosen_method` — строки, не массивы" in prompt
        assert "только эти точные" in prompt
        assert "идентификаторы без пояснений" in prompt
        assert "Не используй bash" in prompt
        assert "буквально перенеси все применимые" in prompt
        assert "Не реализуй формальную семантику самодельным скриптом" in prompt
        assert "импорт `tfl.pattern` означает точное значение `pattern`" in prompt
        assert "`from tfl.srs import parse_srs` обе означают `srs`" in prompt
        assert "Если карточка\nпроверяет другие правила или другой объект" in prompt
        assert "print(parse_srs('RULES').terminates())" in prompt
        assert "`ba` означает `b < a`, не `b > a`" in prompt
        assert "не выдавай запуск на старой задаче за проверку новой" in prompt
        assert "Различие конкретных правил само по себе не отменяет аналог" in prompt
        assert "(длина слова, число пар x...y)" in prompt
        assert "не называй обычный" in prompt
        assert "цепочка `x > yx > yyx > ...` бесконечна" in prompt
        assert "Фактически выполни ровно один разрешённый вызов" in prompt
        assert "разные формальные объекты" in prompt
        assert "не добавляй в `micro_methods` вид неприводимых слов" in prompt
        assert "пробел между правилами меняет формальный объект" in prompt
        assert "сохрани id\nаналога" in prompt
        assert "не буквальные роли правил" in prompt
        assert "Проверь также `prerequisites_used`" in prompt
        assert "не\n   называй её компоненты неубывающими" in prompt
        assert "Все поясняющие поля JSON пиши по-русски" in prompt
        assert "если команда не импортирует `tfl`, укажи `python`" in prompt
        assert "не привязывайся к фразе «заметим, что»" in prompt
        if case.seminar_analog:
            assert case.seminar_analog in prompt
        else:
            assert "Проверенного семинарного аналога не найдено" in prompt
        assert "Не читай evals/pedagogy_holdout" in prompt

    full_case = next(case for case in load_cases() if case.id == "full-injectivity-levels")
    full_prompt = build_prompt(full_case)
    assert "h=parse_morphism('S -> 12; b -> 1; c -> 2')" in full_prompt
    assert "не существует общего критерия" in full_prompt


def test_precomputed_fields_do_not_leak_hidden_expected_values():
    case = load_cases()[0]
    changed_rubric = replace(
        case,
        task_class="RK1-A",
        seminar_analog="hidden-wrong-analog",
    )
    assert build_prompt(case) == build_prompt(changed_rubric)


def test_complete_pedagogical_response_scores_thirteen():
    case = load_cases()[0]
    score = score_response(case, good_response(case), ["py -3 -c import tfl.code"])
    assert score.total == 13
    assert score.passed
    assert not score.issues


def test_pattern_normal_form_reference_response_scores_thirteen():
    case = load_cases()[1]
    score = score_response(case, good_pattern_response(case), ["python3 -c import tfl.pattern"])
    assert score.total == 13
    assert score.passed


def test_pattern_method_accepts_a_common_lexicographic_typo():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["chosen_method"] = response["chosen_method"].replace(
        "лексикографическую", "лексиографическую"
    )
    assert "лексиографическую" in response["chosen_method"]
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert score.components["метод"] == 1
    assert score.passed


def test_bracket_reference_response_scores_thirteen():
    case = load_cases()[2]
    score = score_response(case, good_bracket_response(case), ["python3 -c import tfl.pattern"])
    assert score.total == 13
    assert score.passed


def test_rotation_rejects_confusing_source_letters_with_image_letters():
    case = next(case for case in load_cases() if case.id == "training-cyclic-shift")
    response = {
        "case_id": case.id,
        "mode": "training",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-rotation",
        "discovery_path": [],
        "micro_methods": [],
        "prerequisites_used": [],
        "chosen_method": "циклический сдвиг",
        "oracle_calls": [],
        "final_answer": "E(w1) и E(w2) начинаются с различных букв a и b.",
        "limitations": ["Конечная проверка не заменяет доказательство."],
    }
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("недопустимое утверждение" in issue for issue in score.issues)

    response["final_answer"] = "Коды имеют общий префикс, а исходные слова начинаются с разных букв."
    response["chosen_method"] = "Доказать инъективность через обратный морфизм."
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("обратн" in issue and "морфизм" in issue for issue in score.issues)

    response["chosen_method"] = "Использовать левый циклический сдвиг."
    response["final_answer"] = "Определим E(wx)=xw и назовём это левым сдвигом."
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("E\\(wx" in issue for issue in score.issues)

    response["final_answer"] = "Формулы следуют из леммы об общем префиксе."
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("префикс" in issue for issue in score.issues)


def test_decoding_srs_accepts_boundary_marker_after_boundary_wording():
    case = next(case for case in load_cases() if case.id == "training-decoding-srs")
    response = {
        "case_id": case.id,
        "mode": "training",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-decoding-srs",
        "discovery_path": [],
        "micro_methods": [
            "проверить перекрытия и критические пары; отдельно проверить границу слова маркером $"
        ],
        "prerequisites_used": [],
        "chosen_method": "пополнение по критическим парам",
        "oracle_calls": [],
        "final_answer": "Наивная система не конфлюэнтна на ada; добавить маркер конца $ и правила a) -> (), d( -> )(." ,
        "limitations": ["Конечный список критических пар проверяется полностью."],
    }
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["малые приёмы"] == 2

    response["limitations"] = [
        "Оракул проверяет критические пары с ограничением max_len, это лишь иллюстрация."
    ]
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("критическ" in issue and "огранич" in issue for issue in score.issues)


def test_termination_only_case_rejects_false_transferred_normal_forms():
    case = next(case for case in load_cases() if case.id == "training-simple-srs-measure")
    response = {
        "case_id": case.id,
        "mode": "training",
        "task_class": "LAB-1",
        "seminar_analog": "sem-2026-09-05-pattern-normal-form",
        "discovery_path": [],
        "micro_methods": [],
        "prerequisites_used": [],
        "chosen_method": "лексикографическая мера (длина, inv(w))",
        "oracle_calls": [],
        "final_answer": "Мера убывает, поэтому система завершаема.",
        "limitations": ["Нормальные формы имеют вид b^m a^n."],
    }
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("нормальн" in issue for issue in score.issues)

    response["limitations"] = ["Оракул проверяет только данную систему."]
    response["final_answer"] = "Возьмём строку приоритета ba, то есть b > a."
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("b\\s*>" in issue for issue in score.issues)

    response["final_answer"] = "Мера убывает, поэтому система завершаема."
    response["limitations"] = ["Проверка ограничена словами длины 1–3."]
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("огранич" in issue and "длин" in issue for issue in score.issues)

    response["limitations"] = ["Оракул проверяет только данную систему."]
    response["final_answer"] = (
        "Мера (|w|, ord_lex(w)) убывает, поскольку лексикографический порядок "
        "на конечных словах хорошо основан."
    )
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("ord_lex" in issue for issue in score.issues)

    response["final_answer"] = (
        "Мера (|w|, число пар a...b) убывает в лексикографическом порядке "
        "на N x N, который хорошо основан."
    )
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 1

    response["limitations"] = ["Неприводимые слова имеют вид b*a*. "]
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("b[*]a[*]" in issue for issue in score.issues)

    response["limitations"] = ["Проверяется только завершимость."]
    response["oracle_calls"] = [
        {
            "module": "srs",
            "operation": "terminates",
            "input": "ab->ba aa->a",
            "result": "не выяснено",
        }
    ]
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("ab->ba aa->a" in issue for issue in score.issues)

    response["oracle_calls"] = []
    response["prerequisites_used"] = [
        "Лексикографический порядок на всех конечных словах хорошо основан."
    ]
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("лексикографическ" in issue for issue in score.issues)

    response["prerequisites_used"] = [
        "Shortlex: сначала длина, затем лексикографика при равной длине."
    ]
    response["chosen_method"] = (
        "Общий порядок: короткая длина, затем лексикографика с b < a."
    )
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["метод"] == 1
    assert score.components["границы"] == 1

    response["discovery_path"] = [
        {
            "clue": "Правила уменьшают меру.",
            "action": "Сравнить компоненты.",
            "result": "Обе компоненты неубывают, и одна строго падает.",
            "next_step": "Сделать вывод о завершимости.",
        }
    ]
    score = score_response(case, response, ["python3 -c pass"])
    assert score.components["границы"] == 0
    assert any("неубыва" in issue for issue in score.issues)


def test_full_injectivity_case_rejects_a_fake_length_two_theorem():
    case = next(case for case in load_cases() if case.id == "full-injectivity-levels")
    response = {
        "case_id": case.id,
        "mode": "full",
        "task_class": "CODE",
        "seminar_analog": "sem-2026-09-05-decode-candidate",
        "discovery_path": [
            {
                "clue": "Образы букв попарно различны, но спрашивают слова.",
                "action": "Разделить инъективность на буквах и на словах.",
                "result": "Нужно проверить конкатенации кодовых слов.",
                "next_step": "Сравнить образы S и bc.",
            },
            {
                "clue": "Строка 12 допускает разные разбиения.",
                "action": "Сравнить S и bc.",
                "result": "h(S)=12=h(bc), поэтому найдена коллизия.",
                "next_step": "Перечислить все разбиения 12.",
            },
        ],
        "micro_methods": [
            "разделить инъективность букв и слов",
            "сравнить S и bc и исчерпать разбиения строки 12",
        ],
        "prerequisites_used": ["морфизм слов", "конкатенация образов"],
        "chosen_method": "Проверить оракулом и предъявить коллизию S и bc.",
        "oracle_calls": [
            {
                "module": "code",
                "operation": "is_injective",
                "input": "S -> 12; b -> 1; c -> 2",
                "result": "не инъективен: 12 = 1·2",
            }
        ],
        "final_answer": (
            "Морфизм не инъективен: h(S)=12=h(bc), хотя S и bc — разные "
            "слова. Все декодирования строки 12: одно кодовое слово S либо "
            "два кодовых слова b,c, то есть слово bc."
        ),
        "limitations": ["Полнота доказана перебором разбиений заданной строки."],
    }
    command = "python3 -c \"from tfl.code import parse_morphism\""
    score = score_response(case, response, [command])
    assert score.total == 13
    assert score.passed

    response["limitations"] = [
        "Морфизм инъективен тогда и только тогда, когда это верно на словах длины не более 2."
    ]
    score = score_response(case, response, [command])
    assert score.components["границы"] == 0
    assert any("длин" in issue for issue in score.issues)


def test_wrong_analog_is_a_hard_failure():
    case = load_cases()[0]
    response = good_response(case)
    response["seminar_analog"] = "sem-2026-09-05-rotation"
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["аналог"] == 0
    assert not score.passed


def test_unobserved_oracle_is_a_hard_failure():
    case = load_cases()[0]
    score = score_response(case, good_response(case))
    assert score.components["запуск оракула"] == 0
    assert not score.passed


def test_forbidden_heavy_shortcut_is_rejected():
    case = load_cases()[0]
    response = good_response(case)
    response["chosen_method"] += " Затем применить матричную интерпретацию."
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["границы"] == 0
    assert not score.passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_missing_verification_boundary_loses_component():
    case = load_cases()[0]
    response = good_response(case)
    response["limitations"] = []
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["границы"] == 0
    assert not score.passed
    assert any("граница машинной проверки" in issue for issue in score.issues)


def test_missing_micro_methods_is_a_hard_failure():
    case = load_cases()[0]
    response = good_response(case)
    response["micro_methods"] = []
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["малые приёмы"] == 0
    assert not score.passed


def test_list_of_ready_observations_does_not_replace_discovery_path():
    case = load_cases()[0]
    response = good_response(case)
    response["discovery_path"] = [
        {
            "clue": "Морфизм задан образами букв.",
            "action": "Сразу назвать теорему.",
            "result": "Ответ уже известен.",
            "next_step": "Применить готовый метод.",
        }
    ]
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["путь к решению"] == 0
    assert not score.passed
    assert any("хотя бы два" in issue for issue in score.issues)


def test_false_clue_invalidates_an_otherwise_correct_path():
    case = load_cases()[0]
    response = good_response(case)
    response["discovery_path"][0]["clue"] = "h(z)=11 является суффиксом h(y)=01."
    score = score_response(case, response, ["py -3 check.py"])
    assert not score.content_passed
    assert not score.passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_sardinas_residuals_cannot_replace_the_common_prefix_family():
    case = load_cases()[0]
    response = good_response(case)
    response["discovery_path"][1]["result"] = (
        "Длины остатков S_k алгоритма Сардинаса–Паттерсона равны 2k-1."
    )
    score = score_response(case, response, ["py -3 check.py"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_common_prefix_is_a_lower_bound_not_an_exact_delay_value():
    case = load_cases()[0]
    response = good_response(case)
    response["discovery_path"][1]["result"] = (
        "Для этой пары задержка раскодирования равна 2k."
    )
    score = score_response(case, response, ["py -3 check.py"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_nonempty_sardinas_cycle_cannot_be_called_empty():
    case = load_cases()[0]
    response = good_response(case)
    response["discovery_path"][0]["result"] = (
        "Все множества Сардинаса–Паттерсона пусты."
    )
    score = score_response(case, response, ["py -3 check.py"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_finite_delay_curve_cannot_prove_unboundedness():
    case = load_cases()[0]
    response = good_response(case)
    response["oracle_calls"][0]["result"] = (
        "delay_growth=[1,3,5,7,9,11] подтверждает неограниченность."
    )
    score = score_response(case, response, ["py -3 check.py"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_pattern_system_rejects_the_observed_false_nonconfluence():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["discovery_path"][1]["result"] = (
        "Система не конфлюэнтна, потому что ba неприводимо."
    )
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_pattern_invariant_values_must_match_the_words():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["discovery_path"][1]["result"] = (
        "Получены ab -> aaa при μ=4, ba -> aaa при μ=4 и abb -> aaaaa при μ=6."
    )
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_pattern_position_drop_is_the_length_of_x_plus_one():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["limitations"] = [
        "Сдвиг k равен числу букв a справа от перемещаемой b."
    ]
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_pattern_invariant_weight_arithmetic_is_exact():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["discovery_path"][1]["result"] = (
        "Правило Xb -> aaX меняет вес на -1+2=0."
    )
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_pattern_response_cannot_invent_a_theorem_name_or_test_bound():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["limitations"] = [
        "Проверены слова длины 1–4; конфлюэнтность следует по теореме Ньютона-Бёрджесса."
    ]
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_unique_normal_form_argument_does_not_imply_strong_confluence():
    case = load_cases()[1]
    response = good_pattern_response(case)
    response["discovery_path"][1]["next_step"] = (
        "Завершимость и слабая конфлюэнтность дают сильную конфлюэнтность."
    )
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_arbitrary_bracket_deletion_does_not_preserve_prefix_balance():
    case = load_cases()[2]
    response = good_bracket_response(case)
    response["chosen_method"] += " Каждое удаление сохраняет префиксный баланс."
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_reverse_bracket_step_must_keep_the_variable_substring():
    case = load_cases()[2]
    response = good_bracket_response(case)
    response["discovery_path"][0]["result"] = (
        "Обратный шаг — это вставка () между частями v()t."
    )
    score = score_response(case, response, ["python3 -c import tfl.pattern"])
    assert not score.content_passed
    assert any("недопустимое утверждение" in issue for issue in score.issues)


def test_answer_accepts_grammatical_space_in_not_bounded():
    case = load_cases()[0]
    response = good_response(case)
    response["final_answer"] = response["final_answer"].replace(
        "неограничена", "не ограничена"
    )
    score = score_response(case, response, ["py -3 check.py"])
    assert score.components["ответ"] == 2
    assert score.passed


def test_budget_is_reported_separately_from_content_quality():
    case = load_cases()[0]
    commands = ["py -3 check.py"] * 9
    score = score_response(case, good_response(case), commands, command_budget=8)
    assert score.total == 13
    assert score.content_passed
    assert not score.within_budget
    assert not score.passed


def test_schema_violation_is_not_a_content_pass():
    case = load_cases()[0]
    response = good_response(case)
    response["final_answer"] = [response["final_answer"]]
    score = score_response(case, response, ["py -3 check.py"])
    assert not score.valid_response
    assert not score.content_passed
    assert score.components["ответ"] == 0
    assert any("final_answer" in issue for issue in score.issues)


def test_provider_outage_is_reported_separately_from_model_failure():
    case = load_cases()[0]
    score = score_response(
        case,
        None,
        runner="opencode",
        runner_error="503 Service Unavailable: ERR_CONNECT_FAIL 110",
    )
    assert score.infrastructure_error
    assert not score.passed
    report = score_report([score])
    assert "INFRA ERROR, 0/13" in report
    assert "инфраструктурных ошибок 1" in report


def test_malformed_model_output_is_not_an_infrastructure_error():
    case = load_cases()[0]
    score = score_response(
        case,
        None,
        runner="opencode",
        runner_error="в потоке нет структурированного финального ответа",
    )
    assert not score.infrastructure_error
    assert "FAIL, 0/13" in score_report([score])


def test_cli_lists_and_prints_pedagogy_prompt(capsys):
    assert main(["pedagogy", "list"]) == 0
    assert "training-unbounded-delay" in capsys.readouterr().out
    assert main(
        ["pedagogy", "prompt", "--case", "training-logic-collapse"]
    ) == 0
    output = capsys.readouterr().out
    assert "режиме training" in output
    assert "n,m,k" in output
