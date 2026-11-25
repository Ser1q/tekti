# Architecture: Tekti

Intent: [requirements-review-board.prd.md](prd.md). Status: decided 26 November 2026. Deadline: 27 November 2026, 23:59.

## Problem and goals

A spec author needs to know, before hand-off, what a developer and a tester could not decide from the document, and whom to ask. Every decision below is judged against three things: findings that can be trusted (each tied to real text, little noise), questions addressed to whoever holds the answer, and a run that another person can repeat from the README. The targeted nomination is best architecture, so the design must also be explainable and measurable.

## Approaches considered

| Approach | Trade-off | Verdict |
| :--- | :--- | :--- |
| One strong model, one structured call | Fast to build; misses cross-document conflicts; nothing to show as architecture | Kept as the `fast` mode and as the baseline the panel must beat |
| Fixed review board: staged pipeline with a panel of reviewers, voting and a challenge pass | More calls and more merging logic; predictable, parallel, measurable | **Chosen** |
| Free-form agent that plans its own investigation with tools | Flexible on unknown documents; not repeatable, hard to measure | Rejected for the batch report |
| Board hosted on a managed agent runtime, CLI as a client | Strong cloud story; ties the repeat run to an AWS account | Hosting kept as a stretch, not the core |

## Recommended approach

Tekti is a fixed pipeline whose reviewing steps are agents. A document passes through seven stages; only the order is fixed, and inside the panel stage many reviewers work in parallel.

1. **Ingest.** Markdown or text PDF becomes a list of blocks, each with a stable anchor to its place in the source. Embedded instructions aimed at a reviewer are flagged here.
2. **Segment.** The text is split into atomic requirements. Document identifiers are kept; unnumbered requirements get new ones. Each carries its exact quote.
3. **Rules.** Code, not a model, checks what code can check: empty sections, weak wording, units, and a table of facts (ranges, types, status codes) compared across the parts of the document.
4. **Panel.** Five review angles (functional, API contract, data and integration, tester, developer) are each run by three model families: fifteen reviewers in parallel. The tester tries to derive test cases and the developer tries to derive the contract; whatever they are forced to assume becomes a finding.
5. **Adjudicate.** Findings are grouped by the requirement and the quality criterion they concern, so merging is mechanical and each group has a vote count. Every quote is checked against the source in code. Groups with weak support are challenged by a model from a different family than the one that raised them.
6. **Questions.** Surviving findings become clarifying questions, deduplicated, ranked by what they block, each addressed to a role.
7. **Report.** A template renders the Russian report from the run data. The verdict and all counts are computed in code.

Three modes use the same pipeline: `fast` (one family), `standard`, `board` (full matrix).

## Key decisions

**Stack and libraries**

- **Python with uv.** Already the builder's stack. Alternative considered: TypeScript, rejected for weaker document tooling.
- **Strands Agents for the reviewers.** Each reviewer, the challenger and the question writer is a Strands agent returning structured output; the stage order stays in plain code. Chosen by the builder over plain Python and LangGraph. It runs locally, so the repeat run needs no cloud account.
- **Strands' model connectors as the model layer.** Bedrock is the default; an OpenAI-compatible connector covers Gemini's free tier and local Ollama. This replaces the hand-written two-adapter layer decided earlier, since Strands already provides it.
- **Distribution.** One Python package, run as `uvx tekti <input> <report>`, plus a Docker image.
- **No web demo.** Decided by the builder: CLI plus a recorded run.

**Data model**

One run produces one data file, and the report is only a rendering of it:

- Document → Blocks (anchored) → Requirements (identifier, source identifier, quote, anchor).
- Facts: values extracted for the rule checks.
- Findings: requirement(s), criterion from a closed list drawn from ISO/IEC/IEEE 29148 and IEEE 830, severity, quotes, explanation, which reviewers raised it, challenge verdict.
- Questions: the findings each resolves, the addressed role, what stays blocked.
- Run passport: document, date, models, agent version, mode, and a trace of every model call.

The closed criterion list is what makes merging, voting and scoring against the gold set mechanical.

**Boundaries and contracts**

- **CLI contract:** input path and report path, as the organisers require. Fixed from the first commit.
- **Untrusted input:** the document is data. It never reaches a model as instructions, no model output is trusted without the quote check, and the verdict is computed in code, so text inside a document cannot set the score.
- **Secrets:** read from the environment only; `.env.example` in the root; nothing committed.
- **Data leaving the machine:** document text goes to the chosen model provider. The local profile exists for cases where it must not.
- **Spend:** a cap on calls and output size per run.

**Other**

- **Evaluation is part of the product.** The gold set, a scorer and a comparison table (single family, panel, panel plus rules; recall, noise, cost, time) ship in the repository. This is the test of the PRD's hypothesis and the main architecture evidence.
- **Failure of one reviewer does not fail the run.** A family that errors or returns invalid output is dropped and recorded in the passport.

## Missing pieces

- The gold set as a data file, with "critical" defined per item.
- The closed criterion list with one-line definitions in Russian, paraphrased from the standards.
- The fact table design for the rule checks.
- Prompts for five angles, the challenger and the question writer.
- The Russian report template.
- A text PDF of the organisers' document to test ingestion.

## Spikes and experiments

```
Question:      Which model families on Bedrock return valid structured output in Russian through Strands?
Spike:         One angle, one requirement, every candidate family; 30 minutes.
Decision rule: A family joins the panel if every call returns valid output; otherwise it is dropped.
               If fewer than three pass, the panel shrinks and the passport says so.
```

```
Question:      Does Strands give structured output and parallel runs the way this design assumes?
Spike:         Read the current Strands documentation and run two agents in parallel; 30 minutes.
               My knowledge of its current interface is unverified.
Decision rule: Use Strands for reviewers if both work; if not, reviewers become plain model calls
               behind the same stage contracts and Strands is dropped.
```

```
Question:      How good is one strong model alone?
Spike:         Run the fast mode against the gold set as soon as stages 1 to 3 and one angle exist.
Decision rule: This number is the bar. If the panel later fails to beat it by 10 points of recall,
               the panel becomes an option, as the PRD states.
```

```
Question:      Do tables survive text PDF extraction?
Spike:         Convert the organisers' document to PDF and compare extracted requirements with the Markdown run.
Decision rule: Same requirement count and quotes, keep the extractor; otherwise try one alternative, then
               document the limit.
```

## Stretch, in order

1. Deploy the same package to AgentCore Runtime behind an authorizer, as proof it runs managed.
2. An executable model of the functional rules, with the acceptance table run against it.

## Open questions

- Which three families make the panel: settled by the first spike.
- Who challenges whom, when a finding was raised by all three families: default is no challenge needed.
- Whether the jury repeats the run with paid models: unknown; the README documents both the Bedrock and the free profile.
- Is "tekti" free as a package name: check before the first publish.
- The gold set is unreviewed and written by one model; results on it should be presented with that caveat.
