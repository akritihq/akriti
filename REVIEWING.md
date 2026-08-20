# Reviewing

Findings from RFC-0001's review passes that generalise beyond it. RFC-0001's
own deliberation record is PR #10 and one commit per pass; this file carries
only what transfers to the next document.

## Checking

- **A count check cannot tell a use from a mention.** Keyword totals move when
  a passage that *quotes* a requirement is relocated. Classify every delta per
  token before believing it.
- **The test is whether a reference resolves, not whether it was true when
  written.** Cite by section name and git object, never by line number.
- **A cut passage is a referent as well as a referrer.** Before removing text,
  check what points *at* it. The check is bidirectional.
- **A pointer that summarises what it points at is a second copy with a
  shorter half-life.** Point, or state it -- not both.
- **A script being committed and a figure being reproducible are different
  claims.** Only the second is worth asserting, and only after running it.
- **A figure a decision is reopened against must not move** when someone runs
  it on a different laptop.
- **A wrong figure inside a review finding is the one least likely to be
  recomputed before it is written down.** Re-measure the findings, not just the
  document.

## Writing

- **A changelog asserting work never done** is the clean-plausible-wrong signal
  RFC-0001 §9 exists to catch, pointed at ourselves.
- **A decision already merged is overturned explicitly**, never superseded in
  passing by the branch that depends on it.
- **Fix the class a finding belongs to, not the instance it named.**
- **A trap a reader cannot see becomes a standing regression test, not prose.**
- **Tables are for genuinely short, structured data.**
- **D-numbers are stable identifiers, not a dense sequence.** Do not renumber
  to close a gap.

## The keyword sweep

Cheap, and it does not converge on the first pass: "audited before the change"
is necessary rather than sufficient. One reader auditing their own change
caught one of six.

Sweep for **lowercase obligations**, not just for miscased keywords. The two
that carried the most weight were both found late:

- §9.2's clean-room note -- an AGPLv3 prohibition on reading giotto source,
  carrying legal rather than editorial weight, sitting in an italicised
  lowercase aside.
- §11's "omitting it must be a `TypeError`", checked once and cleared on the
  reasoning that §5.1's "Omitting it MUST raise" already covered it. It does
  not: §5.1 fixes *that* omission raises and §11 fixes *what* it raises and
  where, so under a caps-only reading an implementer could give
  `reduced_homology` a default and still satisfy the letter of the RFC.

**Three sites were audited and deliberately left lowercase.** Do not reopen
them without new grounds: §3.1's I8 row, where the preamble MUST carries
enforcement; §4.2's "`xp` is required for, and only for, an empty `diagrams`",
the ordinary Python sense, with the same paragraph's "the namespace MUST come
from the caller" carrying the obligation; and §6.3's "`core.py` may not convert
either one", explanatory, with §3.3's stdlib-only MUST behind it.