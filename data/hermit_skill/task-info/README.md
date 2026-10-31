# Task: make a local model follow pi skills through to the end

The adapter is the `AGENTS.md` that the coding agent pi reads from its working directory. The
acting model is caveowl, a terse ("caveman") fine-tune of a 35B mixture-of-experts model, served
locally. Neither the model nor pi changes; only this text does.

## What a task is

One user request, sent to pi non-interactively in a fresh directory. Some tasks place files in
that directory first (a PDF, a markdown file). Each request is best handled by one of pi's
installed skills: `coffee-price-tracker` (run its `report.py`), `ocr-and-documents` / `pdf-explore`
(extract text from a PDF with `pdftotext` or the skill's extractor), `summarize` (summarise a local
file), `humanizer` (rewrite AI-sounding text, keeping what it says). pi lists the skills and their
descriptions in the system prompt; the full procedure is in each skill's `SKILL.md`, which the agent
must read with its `read` tool.

## What counts as success (score 1.0)

1. The agent read the relevant skill's `SKILL.md`.
2. It ran the command that skill prescribes, and the command succeeded.
3. It gave a final answer grounded in what the tools showed, without inventing output.

A right answer reached some other way (for example reading the file directly instead of using
the skill) scores 0.25. Anything else scores 0.0. Feedback names each failure reason. The most
common failure today is `skill_not_read`: the agent answers or acts without loading the skill.

## Constraints on the adapter

- It is deployed to a phone-facing Signal assistant: replies must stay short and plain text.
  Keep the existing guidance about safety, asking before destructive actions, and not inventing
  output.
- Every line is paid on every turn of every conversation. Prefer short, general rules over long
  lists. Do not mention specific files, values or prompts from the tasks: the adapter is judged
  on different requests and on skills it was not trained on.
