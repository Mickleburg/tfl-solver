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
---

Load the `tfl-solver` skill and follow it. Work only inside the current
repository. Do not edit files. Use the provided Python commands for executable
checks and return exactly the structured JSON requested by the user prompt.
