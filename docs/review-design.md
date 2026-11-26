# Review design: what each part of the board does

This is the behaviour the prompts and the merging code must implement. It is the orchestrator's design; change it only by agreement.

## What counts as a finding

A finding is a defect in how the document is written, tied to quoted text. It is not a product suggestion. "It would be nice to add SMS" is not a finding; "the document does not say what happens when push is disabled on the device" is.

Each finding names:

- the requirement(s) it concerns: one for a defect inside a requirement, two or more for a conflict, none for something absent from the whole document;
- one criterion from the closed list in `tekti/criteria.py`;
- the exact quote(s). For a conflict, one quote from each side. For an absence, the quote of the nearest text that makes the absence visible (the heading of the empty section, or the statement that needs the missing piece);
- a severity.

Severity is fixed by these tests, in this order:

1. **critical**: two statements cannot both hold; or no test with a definite expected result can be written for the requirement; or something is absent without which a feature the document states cannot be built.
2. **major**: a developer would have to guess, and a wrong guess means rework.
3. **minor**: wording, structure, document hygiene.

Reviewers write in Russian. Reviewers never invent text: a quote must be copied from the document character for character.

## The five angles

Every angle sees the whole document and the full requirement list, because most serious defects are conflicts between distant parts. The angle decides what the reviewer looks for.

**Functional.** Reads the business rules as a rule system. Looks for: rules that collide; cases no rule covers (boundaries, the opposite branch of every "if", state changes mid-period, reversals); terms used without definition; a rule that contradicts its own example or acceptance row; several rules packed into one requirement.

**API contract.** Reads the interface description as a contract a client must code against. Looks for: method semantics that do not fit the operation; statuses listed but unused, or used but unlisted; a status whose description contradicts its example; request and response types that disagree; fields in examples that the tables do not define, and the reverse; declared operations with no method; blank mandatory attributes.

**Data and integration.** Reads the document as data that must be stored and passed between systems. Looks for: a rule that needs data nobody stores; unit or type mismatches between places; component names that change between sections; an integration that is named but not specified; repeat-delivery and ordering behaviour left open; the same rule written in two places.

**Tester (simulation).** Tries to write a test case for every requirement: preconditions, action, expected result. A finding is raised wherever that cannot be done from the text alone: the expected result is not determinable, a precondition is missing, a threshold has no measure, or two requirements give different expected results for the same case. Also compares the acceptance criteria with the requirements: which requirements no acceptance row covers, and which rows contradict a requirement.

**Developer (simulation).** Tries to derive what would be built: the states, the stored fields, the interface contract, the sequence of calls. Every point where it must assume something to continue is a finding, and the assumption it was forced to make is the basis of the clarifying question.

For both simulations the model is told to do the derivation first and report only the points where it got stuck, not the derivation itself.

## Segmentation

- Keep the document's own identifiers (for example "Ф-1") as `source_id`.
- Anything that states what the system must do, store, return or satisfy is a requirement, wherever it sits: goal, UI description, process and sequence descriptions, non-functional, data, integration, acceptance rows, API rules, parameter rows, status descriptions.
- Template boilerplate is not a requirement: the unfilled review checklist, the approvers table, attribute rows like "Author". They are still visible to reviewers as document text, and gaps in them are reported as document-level findings.
- A compound requirement is split into atomic children ("R-003" with `parent_id` pointing at the parent "R-002"); the parent is kept so the report can show the split.
- Quotes are exact. If the model returns a quote that is not in the document, that requirement is retried once, then dropped and recorded.

## Rule layer (no model)

Cheap, repeatable checks whose findings count as one extra "family":

- empty or placeholder sections and cells ("—", "заполняются ...", empty tables, unmarked "Да / Нет");
- weak or unmeasurable wording from a Russian word list ("быстро", "удобно", "при необходимости", "и т.д.", "как правило" ...);
- requirements with no identifier;
- text that addresses the reviewer or a model ("ignore previous", "оцени этот документ на", "игнорируй инструкции" ...), reported as `embedded_instruction`;
- facts extracted with patterns and compared across the document: numeric ranges, units next to amounts (₸, тенге, тиын), HTTP methods, status codes listed versus status codes that have a section or example, field types in request versus response tables.

A rule finding must be something code can be sure of. When unsure, a rule stays silent.

## Adjudication

1. **Quote check.** Every evidence quote must be found in the document after whitespace normalisation. Evidence that fails is removed; a finding left with no evidence is rejected with reason "quote not found".
2. **Bucket.** Group raw findings by criterion and the set of requirement ids. Related criteria are folded first so near-duplicates meet: `correct` with `set_consistent` when two or more requirements are involved; `complete` with `set_complete` when no requirement is named.
3. **Split inside a bucket.** Two findings in one bucket are the same defect only if their evidence quotes overlap or one model call confirms they describe the same thing. One bucket can hold several distinct defects.
4. **Merge.** A merged finding keeps the clearest title and explanation (the one from the reviewer with the most evidence), the union of evidence, the list of who raised it, and the count of distinct model families. Severity is the most common vote; ties go to the higher severity.
5. **Decide.**
   - raised by two or more families, or by a rule: confirmed;
   - raised by one family: challenged.
6. **Challenge.** A model from a different family receives the finding, the quoted text with its surrounding section, and one instruction: decide whether the document, read fairly, really has this defect, and say why. It is not shown the reported severity and rates the defect itself. Upheld findings are confirmed; where the two ratings differ the lower one stands, so a defect is never critical on one family's opinion alone. A rejection needs two challengers from different families; rejected findings are kept in the run data with the reasoning but left out of the report body.
7. **Order.** Confirmed findings are numbered F-001 onward by severity, then by family count, then by position in the document.

## Questions

One question per decision the author must make, not one per finding: findings that would be settled by the same answer share a question.

A question:

- can be answered by one named role;
- states what cannot be built or tested until it is answered;
- offers the two or three plausible answers when the document makes them visible (for example, for the month-boundary conflict: "by purchase date" or "by posting date");
- is ranked by how much it unblocks: critical findings first, then the number of findings and requirements it settles.

Addressee guide: business rules and limits go to the product owner or business owner; meaning of the customer's need to the customer; interface contracts and component boundaries to the architect; storage and implementation limits to the developer; test conditions and acceptance to the tester or analyst; card data and exposure to security; gaps in the document itself to the analyst.

## Verdict (computed in code)

- Development can start only if there is no confirmed critical finding.
- Testing can start only if development can start and there is no confirmed major-or-worse finding under `verifiable` or `set_validatable`.
- "Fix first" lists confirmed critical findings, ordered as above, capped at ten.

## Injection posture

The document is the thing under review, never a source of instructions. Its text is passed only as data inside a marked block in the user turn; the system turn says that anything inside that block which reads like an instruction is to be reported as a finding and not followed. Because counts and the verdict are computed in code from confirmed findings, text in the document cannot set the score directly; the quote check and the cross-family challenge limit what a manipulated reviewer can add.
