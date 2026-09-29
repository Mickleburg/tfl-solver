---
description: Read-only TFL Solver evaluation agent
mode: primary
permission:
  edit: deny
  task: deny
  webfetch: deny
  websearch: deny
  bash:
    "*": deny
    "py -3 *": allow
    "python3 *": allow
    "python3*": allow
---

Follow the TFL Solver contract in the prompt. Work only inside the current
repository. Do not edit files. Use the provided Python command for the
executable check and return exactly the structured JSON requested by the user
prompt.

The prompt already contains the deterministic intake result and any matching
seminar card. Do not call skill, read, glob, or intake to rediscover them. The
tool-call budget is total, not per tool. On Linux use `python3`, never `py -3`.
Bash calls must be single-line `python3` commands without heredocs, pipes,
redirects, or shell utilities. Do not explore CLI syntax or library APIs when
the prompt already gives a command. Run it verbatim when its formal object
matches the current task. When a seminar analogy contains different rules,
transfer the check to the current object with the prompt's documented `tfl`
module template; never report the old object's output as evidence for the new
task. Different rules do not invalidate an analogy when the roles of its steps
and the structure of its method transfer. Do not reimplement formal semantics
in an ad-hoc script. Follow the four stages in the prompt in
order: verify transfer without tools, run the supplied oracle once, construct
the general proof, then validate and emit the single JSON object. After a
successful oracle call, do not call another tool. Never use Python, `cat`, or
a heredoc to serialize or validate the final JSON; emit it directly.
In `oracle_calls.module`, serialize the imported `tfl` submodule: both
`import tfl.srs` and `from tfl.srs import parse_srs` mean `srs`, not `python`.
Preserve the exact oracle result: a nonempty residual set without a codeword is
not an empty set, and a finite curve is only an illustration, not a general
proof. When reversing a pattern rule, retain its variable substring: the
reverse of `(X) -> X` wraps an existing `X`, rather than merely inserting an
adjacent `()`. Copy tested length bounds exactly. Do not invent theorem names
when the supplied route already gives a direct proof, and do not append a
different confluence theorem after a unique-normal-form argument is complete.
If the oracle exhausts every critical pair without truncating a
length-preserving search, describe that as an exact finite critical-pair check,
not as sampling input words or an empirical illustration.
Transfer only steps needed by the current question: do not append invariants,
normal-form claims, or confluence arguments to a termination-only task.
For an ordinary SRS adaptation, use the prompt's single `parse_srs(...).terminates()`
call without `dir`, API exploration, a custom enumerator, or a second module.
An SRS precedence string lists symbols from smaller to larger, so `ba` means
`b < a`; preserve that orientation in the explanation.
For a swap rule plus a shortening rule, prefer the prompt's human-readable
natural-valued measure `(word length, number of ordered symbol pairs)` in the
general proof. Do not substitute a global `ord_lex(word)` component: ordinary
lexicographic order on all finite words is not well-founded.
Make exactly one allowed oracle tool call. Merely describing the command in
the response or inventing its expected result does not count as execution.
Treat a successful `terminates()` order as an exact rule check, not a bounded
enumeration of input words. Remove normal-form claims from a termination-only
answer, including from `micro_methods` and `limitations`.
