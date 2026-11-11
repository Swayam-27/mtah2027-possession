# More than a Hack 2027 — Project Brief for Claude Code

## What this is
We are a team competing in **More than a Hack 2027**, a football analytics hackathon run by
Mobile World Capital Barcelona, Barça Innovation Hub and Metrica Sports, using real FC Barcelona
match data.

Key links:
- Official challenge repo (cloned here): https://github.com/metrica-sports/more-than-a-hack-2027
- Hackathon website (registration, overview, prizes): https://landing.mobileworldcapital.com/more-than-a-hack-2027
- Organisers' announcement with the full timeline: https://www.metrica-sports.com/blog/more-than-a-hack-2027-build-something-real-with-fc-barcelona-match-data
- Talent Arena (where the final happens): https://talentarena.tech
- Questions about enrolment or terms: hackathon.talentarena@mobileworldcapital.com
- Questions about data, metrics or docs: open an issue on the official repo

## The three rounds
1. **Round 1 – Initial Challenge (online, deadline 17 Nov 2026)** ← WE ARE HERE
   Ball-possession estimation on one match. Scored automatically. Top 20 teams advance
   (results before 26 Nov).
2. **Round 2 – Innovation Challenge (online, 26 Nov 2026 – 14 Jan 2027, top 20 only)**
   10 professional matches with tracking + event data, open-ended design challenges
   (details announced later). Top 5 advance.
3. **Final pitch (2 Mar 2027, Talent Arena, Barcelona, top 5 only)**
   Present the Round 2 project to a professional jury.

Write Round 1 code cleanly and modularly (data loading, possession model, smoothing,
evaluation) so it can be reused for Round 2.

## Round 1 task
Match: Barcelona (home, `H`) vs Athletic Club (away, `A`), id `BAR-ATH`.
For **every frame of the tactical video (199,319 frames, 0…199318)** output one label:
- `H` – Barcelona in possession, ball alive
- `A` – Athletic in possession, ball alive
- `D` – dead ball (not in play)
- `X` – abstain / not part of play (pre-match, halftime, post-match)

Output: CSV with header exactly `frame,label`, 199,319 rows, frames in order, no blanks.

Graded frames: halves 11,185–90,651 and 110,808–190,197. Frames 110,808–113,741 are graded
and are all `D` (second half kicks off at 113,742). Everything else is ungraded.

## Inputs provided (see DATA.md for formats — read the docs, not the data)
- Automatic tracking data (ATD): players + ball, frame by frame. Uncorrected: gaps, ID switches,
  **ball missing on ~41% of graded frames** (often at the most important moments).
- Smart Tagging: automatic tactical phases with team tags and set pieces/restarts. Useful
  but noisy; phases ≠ possessions.
- Camera-to-pitch homography.
- Tactical video + broadcast video. Broadcast is on its own timeline (no mapping provided);
  submission is indexed to the TACTICAL video.
- Public ground truth for the first 5 minutes only. It is a sanity check, NOT a test set —
  do not overfit to it (the smoothing window that wins there loses on the full match).

## Scoring
- Ranked on **spellF1**: how many real possessions are recovered as whole possessions.
  Flickering labels are heavily punished. Treat possession as segments, not independent frames.
- Tie-break: **macroF1** (per-frame).
- On graded frames, `X` is filled with our last label in that half for spellF1 — abstaining
  through an occlusion inside a possession is free; abstaining across a hand-over is costly.
- Baseline to beat: spellF1 **0.2433** (nearest player to ball + majority-vote window).

## Deliverables (submitted via Google Form, link comes by email)
1. `<team_name>.csv` – the possession file
2. Public GitHub repo that reproduces EXACTLY the same CSV from the data bundle
   (pinned deps, install + run instructions, one command to regenerate). NO match data in the repo.
3. `<team_name>.pdf` or `.md` technical doc: approach, assets used/dropped, possession rules
   (loose balls, challenges, restarts), limitations, reproduction steps.

## Hard rules
- Labels must be produced by an algorithm. No hand-labelling.
- No use of the private ground truth beyond the public 5 minutes.
- **Terms & Conditions forbid uploading the match videos, datasets or any extracts to a
  third-party cloud-hosted LLM — that includes you (Claude).** Therefore:
  - NEVER read, cat, head, open, or display anything inside `data/` or any file derived from it
    (including our submission CSVs and intermediate outputs).
  - Do not run commands whose output prints data rows, frame-level values or video content.
  - You MAY read the official docs (README, CHALLENGE, DATA, EVALUATION, SUBMISSION, RULES, FAQ)
    and the starter/evaluation code.
  - Write code; the team runs it locally. When you need results, have scripts print only
    aggregate metrics (spellF1, macroF1, possession count, mean duration) — and ask a team
    member before running anything against the data.
- Always run `python evaluation/validate_submission.py <file>.csv` before any submission.

## How we work on the plan
**Approach chosen on 10 Nov 2026** — see `ROUND1_STORY.md` (plain-language plan + progress log;
keep its log updated as work lands). In short: rule-based possession engine as the safe
submission, a learned model trained on OTHER matches (DFL open data, Metrica sample data) on top,
every change judged on a benchmark built from those external matches degraded to look like our
ATD. The official repo is cloned in `official/` (data bundle goes in `official/data/`).
C: drive is nearly full — keep venvs, caches and downloads on D:.

The original process, kept for reference:
1. Read the official docs (README, CHALLENGE, DATA, EVALUATION, SUBMISSION, RULES, FAQ,
   OVERVIEW) and the `starter/` and `evaluation/` code.
2. Propose 2–4 candidate approaches for Round 1, each with: the idea, why it should improve
   spellF1 over the baseline, effort estimate, risks, and what it needs from the data.
3. Discuss with the team and wait for us to pick one. Do not start implementing until we agree.
4. Keep the deadline in mind (17 Nov 2026): always keep a valid, validated submission on hand,
   and prefer approaches we can finish, explain in the technical doc and reproduce exactly.

## Team
Team name (must match all members' registration exactly): Chandal Chowkdi
Members: <FILL IN>
