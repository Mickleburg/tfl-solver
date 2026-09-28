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

Load the `tfl-solver` skill and follow it. Work only inside the current
repository. Do not edit files. Use the provided Python commands for executable
checks and return exactly the structured JSON requested by the user prompt.

The tool-call budget is total, not per tool. Load the skill once; do not read
its file again. Read one seminar card or one recipe before source code. On
Linux use `python3`, never `py -3`. Bash calls must be single-line `python3`
commands without heredocs, pipes, redirects, or shell utilities. Do not explore
CLI syntax or library APIs when the prompt or recipe already gives a command.
