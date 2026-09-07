"""Набор для оценки прогоняется целиком и сверяется с записанными исходами.

Смысл этого теста — не в зелёных галочках. Утверждения вида «класс закрыт»
жили в прозе `docs/OPEN-GAPS.md`; здесь каждое из них исполняется, и любое
расхождение с записанным исходом — либо регрессия, либо повод обновить
запись. Доля `вручную` показывает, где агент по-прежнему только помощник.
"""

from __future__ import annotations

import pytest

from evals.suite import CASES, MANUAL, PARTIAL, SOLVED
from tools.eval_suite import markdown, summary


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_case_lands_where_recorded(case):
    status, detail = case.evaluate()
    assert status == case.expected, f"{case.id}: {status} вместо {case.expected} — {detail}"


def test_every_class_of_the_taxonomy_is_represented():
    """Четырнадцать классов рецептов — и хотя бы по задаче на каждый.

    `RK1-A/B/C` и `RK2-A/B/C` считаются отдельно, `EXAM-2` набором пока
    не покрыт: второй вопрос билета это детерминизм грамматики, и он
    разбирается тем же кодом, что `LAB-3`.
    """
    covered = {case.task_class for case in CASES}
    expected = {
        "LAB-1", "LAB-2", "LAB-3", "LAB-4",
        "RK1-A", "RK1-B", "RK1-C",
        "RK2-A", "RK2-B", "RK2-C",
        "EXAM-1", "EXAM-3", "PHARMA",
    }
    assert expected <= covered


def test_ids_are_unique():
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))


def test_statuses_are_from_the_three_allowed():
    assert {case.expected for case in CASES} <= {SOLVED, PARTIAL, MANUAL}


def test_the_report_is_generated_from_the_run():
    results = [(case, case.expected, "—", 0.0) for case in CASES]
    text = markdown(results)
    assert "| класс | решено | частично | вручную | ошибок |" in text
    assert all(case.id in text for case in CASES)

    table = summary(results)
    assert sum(row[SOLVED] for row in table.values()) >= 20


def test_manual_cases_say_why():
    """«Вручную» без объяснения — это не запись, а отговорка."""
    for case in CASES:
        if case.expected != MANUAL:
            continue
        _status, detail = case.evaluate()
        assert len(detail) > 40, case.id
