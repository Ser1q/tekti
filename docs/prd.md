# PRD: Tekti, a requirements review board

Status: draft, 26 November 2026. Owner: solo builder.
Update, 27 November 2026: the web demo in section 6 was dropped (CLI only, see architecture.md); the jury can repeat the run with their own model keys (see README.md).
First proof: BCC-HUB x MNU FinTech Hackathon submission, due 27 November 2026, 23:59.

## 1. Problem statement

A specification is written by someone who knows the customer and read by people who know the system. Neither side holds the other's context: product cannot see the codebase, development cannot see the customer's need. Each side now consults an AI assistant, and that assistant inherits the same blind spot.

The result is that specifications bounce between product and development. Contradictions, gaps and untestable wording are found late, by a developer or tester who has to stop and ask. Teams cope by pasting the specification into a chatbot again and again and by building prototypes to discover what was meant.

The cost of not solving it: rework, repeated demos, and features built on a guess.

## 2. Evidence

| Claim | Status |
| :--- | :--- |
| A developer receiving vague briefs builds repeated demos and prototypes to find the real need | Observed first-hand: the builder's own projects show dozens of plan files and repeated demo seeds and prototypes |
| A chatbot review "misses the real issues while serving unrelated stuff" | Observed first-hand by the builder; not measured |
| The organisers' sample specification contains serious defects a manual checklist did not catch | Observed: about 45 defects found in the sample, including direct contradictions between a requirement and its acceptance criterion; its 14-row review checklist is unfilled |
| A developer writing a PRD or epic still needs product-side input to finish it | Observed first-hand by the builder; this PRD itself has open questions only product or the jury can answer |
| Bank analysts feel this pain and would change how they work | Assumption. Validate via jury feedback at the final and a conversation with a BCC analyst |
| The same approach helps a developer review a customer's brief | Assumption. Validate by running it on one real brief |

## 3. Thesis (why build it)

**Why this.** The missing piece in a weak specification is almost always something only the other side knows. A clarifying question is what carries context across that gap. A review that ends in a list of defects still leaves the author guessing whom to ask; a review that ends in addressed questions tells them what to do next.

**Why now.** Both sides already use AI, so specifications are produced faster than they are checked. The checking step has not kept up, and a general chatbot does not fill it.

**Why it beats the current cope.** Against a chatbot review, three things must be true for someone to switch:

1. Every finding points at the exact text, so it can be trusted or dismissed in seconds.
2. Findings are about real blockers, with little noise.
3. Every question names who can answer it (product, development, security, the customer) and what stays blocked until they do.

**Vision.** No spec reaches development until the questions only the other side can answer have been asked.

## 4. Hypothesis

We believe that showing a spec author each defect against the exact text, plus questions addressed to whoever holds the answer, will let them fix the spec before hand-off, so it stops bouncing between product and development.

**We'll know we're right if**, on the organisers' document:

- at least 80% of the critical seeded defects are found;
- every finding quotes text that really exists in the document;
- every critical defect carries an addressed question;
- several independent reviewers together beat the best single reviewer on recall by 10 points or more;
- outcome signal: a place among the 12 finalists on 1 December 2026.

**We'll know we're wrong if**:

- more than one finding in five is noise; or
- the group of reviewers does not beat a single reviewer, in which case the group is ceremony and becomes an option, not the core.

## 5. Target user and job to be done

**Primary user.** The analyst who writes the specification, at the moment before hand-off to development.

**Job to be done.** When I've finished a spec and am about to hand it to development, I want to know what a developer and a tester could not decide from it, so it doesn't bounce back.

**Secondary user.** A developer on the other side of the gap, in two situations with the same job:

- receiving a brief from a customer who cannot yet say what they need, and wanting the questions to send back;
- drafting a PRD or an epic themselves and finding it still needs input only the product side can give. The review should say which parts are blocked on product and what to ask.

**Wider audience.** The team around the spec (reviewer, tester, delivery lead) reads the same report to decide "ready or not".

**Non-users.** People who want the specification written for them; code reviewers; compliance or legal sign-off.

## 6. MVP

The thinnest line that proves or kills the hypothesis:

- One command takes a requirements document and writes a report file in Russian.
- The report has the six sections the organisers require: run passport, how requirements were split, check of each requirement, check of the whole set, summary, and a verdict on whether development and testing can start and what to fix first.
- On top of those: a ranked list of clarifying questions, each addressed to a role and tied to the defect it resolves.
- Input: Markdown and text PDF.
- Several independent reviewers examine the document, and a finding must survive a challenge before it is reported.
- A short comparison of one reviewer against the group, measured on the gold set, to test the hypothesis.

After that works end to end, in this order: a deployed web demo where a visitor uploads a document, sees the findings against the text and answers the questions.

**Cut order if time runs short:** scanned PDFs first, then the web demo. The group of reviewers is cut last.

**Doors.** One-way: the repository contents and report at the deadline, and anything public that spends cloud credits. Everything else is reversible, so no spikes are needed.

## 7. Success metrics

| Metric | Target | How measured |
| :--- | :--- | :--- |
| Recall on critical defects | 80% or more | Against the gold set for the organisers' document |
| Noise | 20% of findings or fewer | Findings not matching a gold item, reviewed by hand |
| Grounding | 100% of findings quote existing text | Automatic check of each quote against the document |
| Question coverage | Every critical defect has an addressed question | Count in the report |
| Group against single reviewer | 10 points or more of recall | Same gold set, same document |
| Reproducibility | A second person produces a report from the README alone | One clean-machine run before submission |
| Selection | Among the 12 finalists | Organisers' announcement, 28 November to 1 December |
| Personal use | At least 3 questions the builder would really send a customer | TBD, needs validation on one real brief |

## 8. Non-goals

- Writing or rewriting the specification for the author.
- Jira, Slack or codebase integration. These are the direction of the product and appear only as such in the presentation.
- Compliance or security sign-off.
- Scanned or image-only PDFs.
- Accounts, teams and history in the web demo.

## 9. Constraints

- Solo builder with little hands-on time; most of the build is delegated to AI agents, with the builder deciding architecture and focus.
- Deadline: 27 November 2026, 23:59. Final on 4 December 2026.
- Fixed by the organisers: report in Russian; one-command run taking an input path and an output path; the report must be produced by the agent, not by hand; no secrets in the repository, with an example config in the root; a README that lets another person repeat the run.
- Judging: clarification, architecture, value, and presentation at 25% each. Primary nomination targeted: best architecture.

## 10. Open questions

- [ ] The gold set is unreviewed and was written by one AI model; it may favour reviewers that think alike. Who checks it, and when?
- [ ] Will the jury repeat the run themselves, and with what access to paid models?
- [ ] Do bank analysts want questions addressed by role, or by named person or team?
- [ ] Which real brief serves as the personal-use test?
- [ ] How is "critical" defined for a defect, so the 80% target is not adjustable after the fact?
- [ ] Is the name "tekti" free on the package registry the CLI will be published to?

Engineering decisions (models, providers, hosting, structure of the reviewers, error handling) are deliberately left to the architecture stage.
