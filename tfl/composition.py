"""Portable regular-language compositions and exact comparison (stdlib only).

The file can also be copied as a standalone module. Budgets produce an
exception, never an equivalence verdict. Formula references are automata only;
every operation occurrence in the tree is counted, including repeated ones.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json


class BudgetExceeded(RuntimeError):
    pass


def load_json(text: str, *, allow_trailing_commas: bool = False):
    """Return (document, removed-comma offsets); recovery must be explicit."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    if not allow_trailing_commas:
        return json.loads(text, object_pairs_hook=unique_object), []
    quoted = escaped = False
    output, removed = [], []
    for i, c in enumerate(text):
        if quoted:
            output.append(c)
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
            output.append(c)
        elif c == ",":
            following = i + 1
            while following < len(text) and text[following].isspace():
                following += 1
            if following < len(text) and text[following] in "}]":
                removed.append(i)
            else:
                output.append(c)
        else:
            output.append(c)
    return json.loads("".join(output), object_pairs_hook=unique_object), removed


@dataclass
class Machine:
    alphabet: tuple[str, ...]
    size: int
    start: int
    finals: frozenset[int]
    edges: dict[tuple[int, str | None], frozenset[int]]

    def closure(self, states):
        seen = set(states)
        todo = list(seen)
        while todo:
            for q in self.edges.get((todo.pop(), None), ()):
                if q not in seen:
                    seen.add(q)
                    todo.append(q)
        return frozenset(seen)

    def step(self, states, symbol):
        return self.closure(r for q in states for r in self.edges.get((q, symbol), ()))

    def accepts(self, word):
        states = self.closure({self.start})
        for symbol in word:
            states = self.step(states, symbol)
        return bool(states & self.finals)

    def as_json(self):
        return {"kind": "nfa", "alphabet": list(self.alphabet),
                "states": list(range(self.size)), "start": self.start,
                "finals": sorted(self.finals), "transitions": [
                    {"from": q, "symbol": c, "to": r}
                    for (q, c), targets in sorted(self.edges.items(), key=lambda x: (x[0][0], x[0][1] or ""))
                    for r in sorted(targets)]}


def parse_machine(raw: dict, *, alphabet=None) -> Machine:
    if not isinstance(raw, dict):
        raise ValueError("Expected an automaton object")
    if "dfa" in raw:
        if not isinstance(raw["dfa"], dict):
            raise ValueError("Expected a DFA object")
        raw = dict(raw["dfa"], kind="dfa")
    if not isinstance(raw, dict) or raw.get("kind", "dfa") not in {"dfa", "nfa"}:
        raise ValueError("Expected a DFA or NFA object")
    symbols = raw.get("alphabet", alphabet)
    if not isinstance(symbols, list) or any(not isinstance(c, str) or len(c) != 1 for c in symbols):
        raise ValueError("Alphabet must list single characters")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Duplicate alphabet symbols")
    if alphabet is not None and set(symbols) != set(alphabet):
        raise ValueError("Argument alphabet differs from formula alphabet")
    transitions = raw.get("transitions")
    if not isinstance(transitions, list):
        raise ValueError("Missing transition list")
    if not isinstance(raw.get("finals"), list) or len(set(raw["finals"])) != len(raw["finals"]):
        raise ValueError("Final states must be a list without duplicates")
    mentioned = [raw.get("start"), *raw.get("finals", [])]
    for edge in transitions:
        if not isinstance(edge, dict) or set(edge) != {"from", "symbol", "to"}:
            raise ValueError("Transition needs from/symbol/to")
        mentioned.extend((edge["from"], edge["to"]))
    if any(type(q) not in (str, int) for q in mentioned):
        raise ValueError("State names must be strings or integers")
    states = raw.get("states", list(dict.fromkeys(mentioned)))
    if not isinstance(states, list) or not states or any(type(q) not in (str, int) for q in states):
        raise ValueError("Invalid state list")
    if len(set(states)) != len(states) or not set(mentioned) <= set(states):
        raise ValueError("Duplicate or undeclared state")
    index = {q: i for i, q in enumerate(states)}
    edges = {}
    for edge in transitions:
        symbol = edge["symbol"]
        if symbol is not None and symbol not in symbols:
            raise ValueError("Transition outside alphabet")
        if raw.get("kind", "dfa") == "dfa" and symbol is None:
            raise ValueError("DFA cannot have epsilon transitions")
        key = (index[edge["from"]], symbol)
        if raw.get("kind", "dfa") == "dfa" and key in edges:
            raise ValueError("Duplicate deterministic transition")
        edges.setdefault(key, set()).add(index[edge["to"]])
    if raw.get("trap_state") is not None:
        trap = index.get(raw["trap_state"])
        if trap is None or raw["trap_state"] in raw["finals"] or any(
                edges.get((trap, c)) != {trap} for c in symbols):
            raise ValueError("Invalid trap_state")
    return Machine(tuple(symbols), len(states), index[raw["start"]],
                   frozenset(index[q] for q in raw["finals"]),
                   {k: frozenset(v) for k, v in edges.items()})


ARITY = {"union": 2, "concat": 2, "star": 1, "shuffle": 2, "sqrt": 1}


def operation_count(node, arguments) -> int:
    if isinstance(node, dict) and set(node) == {"ref"}:
        if not isinstance(node["ref"], str) or node["ref"] not in arguments:
            raise ValueError("Unknown argument reference")
        return 0
    if not isinstance(node, dict) or set(node) != {"op", "args"}:
        raise ValueError("Expected a reference or operation tree")
    op, children = node["op"], node["args"]
    if op not in ARITY or not isinstance(children, list) or len(children) != ARITY[op]:
        raise ValueError("Unknown operation or wrong arity; cascade needs an explicit separate semantics")
    return 1 + sum(operation_count(child, arguments) for child in children)


def combine(op: str, parts: list[Machine], *, max_states=100_000) -> Machine:
    alphabet = parts[0].alphabet
    if any(set(p.alphabet) != set(alphabet) for p in parts):
        raise ValueError("Composition alphabet mismatch")
    if op == "sqrt":
        base = determinize(parts[0], max_states=max_states)
        identity = tuple(range(base.size))
        seen, todo, edges, finals = {identity: 0}, deque([identity]), {}, set()
        while todo:
            transform = todo.popleft()
            q = seen[transform]
            if transform[transform[base.start]] in base.finals:
                finals.add(q)
            for c in alphabet:
                target = tuple(next(iter(base.edges[p, c])) for p in transform)
                if target not in seen:
                    if len(seen) >= max_states:
                        raise BudgetExceeded("Square-root transformation budget")
                    seen[target] = len(seen)
                    todo.append(target)
                edges[q, c] = frozenset({seen[target]})
        return Machine(alphabet, len(seen), 0, frozenset(finals), edges)
    if op == "shuffle":
        a, b = parts
        size = a.size * b.size
        if size > max_states:
            raise BudgetExceeded("Shuffle state budget")
        edges = {}
        for q in range(a.size):
            for r in range(b.size):
                for c in (*alphabet, None):
                    targets = {p*b.size+r for p in a.edges.get((q, c), ())}
                    targets.update(q*b.size+p for p in b.edges.get((r, c), ()))
                    if targets:
                        edges[q*b.size+r, c] = frozenset(targets)
        return Machine(alphabet, size, a.start*b.size+b.start,
                       frozenset(q*b.size+r for q in a.finals for r in b.finals), edges)
    offsets, size = [], 0
    for part in parts:
        offsets.append(size)
        size += part.size
    extra = op in {"union", "star"}
    if size + extra > max_states:
        raise BudgetExceeded("Composition state budget")
    edges = {(q+off, c): set(r+off for r in targets)
             for part, off in zip(parts, offsets) for (q, c), targets in part.edges.items()}
    if op == "union":
        start = size
        edges[start, None] = {p.start+off for p, off in zip(parts, offsets)}
        finals = {q+off for p, off in zip(parts, offsets) for q in p.finals}
    elif op == "concat":
        a, b = parts
        start, finals = a.start, {q+offsets[1] for q in b.finals}
        for q in a.finals:
            edges.setdefault((q, None), set()).add(b.start+offsets[1])
    elif op == "star":
        a = parts[0]
        start, finals = size, {size, *a.finals}
        edges[start, None] = {a.start}
        for q in a.finals:
            edges.setdefault((q, None), set()).add(a.start)
    else:
        raise ValueError("Unknown operation")
    return Machine(alphabet, size+extra, start, frozenset(finals),
                   {k: frozenset(v) for k, v in edges.items()})


def determinize(machine: Machine, *, max_states=100_000) -> Machine:
    start = machine.closure({machine.start})
    seen, todo, edges, finals = {start: 0}, deque([start]), {}, set()
    while todo:
        current = todo.popleft()
        q = seen[current]
        if current & machine.finals:
            finals.add(q)
        for c in machine.alphabet:
            target = machine.step(current, c)
            if target not in seen:
                if len(seen) >= max_states:
                    raise BudgetExceeded("Determinization state budget")
                seen[target] = len(seen)
                todo.append(target)
            edges[q, c] = frozenset({seen[target]})
    return Machine(machine.alphabet, len(seen), 0, frozenset(finals), edges)


def compile_document(document, *, max_operations=5, max_states=100_000):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    definitions = document["arguments"]
    if not isinstance(definitions, dict) or not definitions or any(not name for name in definitions):
        raise ValueError("Arguments must be a nonempty named map")
    count = operation_count(document["formula"], definitions)
    if count > max_operations:
        raise ValueError(f"Operation limit exceeded: {count} > {max_operations}")
    arguments = {name: parse_machine(raw, alphabet=document["alphabet"])
                 for name, raw in definitions.items()}
    if any(m.size > max_states for m in arguments.values()):
        raise BudgetExceeded("Argument state budget")
    def build(node):
        if "ref" in node:
            return arguments[node["ref"]]
        return combine(node["op"], [build(p) for p in node["args"]], max_states=max_states)
    return build(document["formula"]), count


def compare(left: Machine, right: Machine, *, max_pairs=1_000_000):
    """Exact BFS in the product of subset automata; shortest counterexample."""
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    alphabet = sorted(set(left.alphabet) | set(right.alphabet))
    start = (left.closure({left.start}), right.closure({right.start}))
    todo, parent = deque([start]), {start: None}
    while todo:
        pair = todo.popleft()
        a, b = pair
        if bool(a & left.finals) != bool(b & right.finals):
            word = []
            while parent[pair] is not None:
                pair, c = parent[pair]
                word.append(c)
            return {"equivalent": False, "counterexample": "".join(reversed(word)), "pairs": len(parent)}
        for c in alphabet:
            target = (left.step(a, c), right.step(b, c))
            if target not in parent:
                if len(parent) >= max_pairs:
                    raise BudgetExceeded("Exact-comparison pair budget")
                parent[target] = (pair, c)
                todo.append(target)
    return {"equivalent": True, "counterexample": None, "pairs": len(parent)}
