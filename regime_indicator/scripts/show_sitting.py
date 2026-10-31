"""Print one panel sitting to the terminal, as the Transcripts tab shows it.

Reads results/transcripts.json, the cached record of all 582 sittings, and
renders a single day's exchange: the detector's proposal, what each of the four
specialists replied, the critic's verdict, and the label that was finally
emitted. Intended for taking a screenshot of real model output.

Usage:
    python scripts/show_sitting.py 2018-12-03
    python scripts/show_sitting.py 2018-12-03 --no-color
    python scripts/show_sitting.py --list 2018-12
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

# Windows consoles default to cp1252, which cannot print the transcript text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "transcripts.json"
WIDTH = 96

ROLE_TITLES = {
    "volatility": "VOLATILITY DESK",
    "market_sentiment": "MARKET SENTIMENT DESK",
    "news_sentiment": "NEWS SENTIMENT DESK",
    "credit_conditions": "CREDIT CONDITIONS DESK",
}
ADJ_WORDS = {"raise": "detector should be MORE certain",
             "lower": "detector should be LESS certain",
             "hold": "detector's certainty is about right",
             "keep": "detector's certainty is about right"}


class Ink:
    def __init__(self, on):
        self.on = on

    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s):  return self._w("1", s)
    def dim(self, s):   return self._w("2", s)
    def cyan(self, s):  return self._w("36", s)
    def amber(self, s): return self._w("33", s)
    def red(self, s):   return self._w("31", s)
    def green(self, s): return self._w("32", s)


def wrap(text, indent=4):
    pad = " " * indent
    return "\n".join(textwrap.wrap(str(text), WIDTH - indent,
                                   initial_indent=pad, subsequent_indent=pad))


def rule(ink, char="─"):
    return ink.dim(char * WIDTH)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("date", nargs="?", help="sitting date, e.g. 2018-12-03")
    ap.add_argument("--list", metavar="PREFIX", dest="list_prefix",
                    help="list available sitting dates starting with PREFIX")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"not found: {SRC}")
    data = json.loads(SRC.read_text(encoding="utf-8"))
    sittings = {v["date"]: v for v in data["calls"].values()}

    if a.list_prefix:
        hits = sorted(d for d in sittings if d.startswith(a.list_prefix))
        print(f"{len(hits)} sitting(s) matching {a.list_prefix!r}:")
        for d in hits:
            print("  ", d)
        return 0

    if not a.date:
        ap.error("give a date, or --list a prefix")
    if a.date not in sittings:
        near = sorted(d for d in sittings if d[:7] == a.date[:7])
        sys.exit(f"no sitting on {a.date}. That month has: {', '.join(near) or 'none'}")

    s = sittings[a.date]
    ink = Ink(not a.no_color and sys.stdout.isatty())
    prop = s["proposal"]

    print()
    print(rule(ink, "═"))
    print(ink.bold(f"  AGENTIC PANEL SITTING — {a.date}")
          + ink.dim("    (claude-sonnet-5, 5 calls)"))
    print(rule(ink, "═"))

    print("\n" + ink.cyan(ink.bold("  DETECTOR'S PROPOSAL")) + ink.dim("   (what the panel is asked to review)"))
    probs = "  ".join(f"{k} {v:.3f}" for k, v in prop["probabilities"].items())
    print(f"    label       {ink.amber(prop['label'])}")
    print(f"    confidence  {prop['confidence']:.4f}")
    print(ink.dim(f"    {probs}"))

    print("\n" + rule(ink))
    print(ink.cyan(ink.bold("  WHAT EACH SPECIALIST REPLIED"))
          + ink.dim("   (four concurrent calls, disjoint evidence)"))
    print(rule(ink))

    for role in data["roles"]:
        r = s.get(role)
        if not r:
            continue
        head = ROLE_TITLES.get(role, role.upper())
        flag = "" if r.get("inf", True) else ink.dim("  [not informative]")
        print("\n  " + ink.bold(head) + flag)
        print(f"    state       {ink.amber(r['state'])}"
              + ink.dim(f"    confidence {r['conf']}"))
        print(ink.dim(f"    adjustment  {ADJ_WORDS.get(r.get('adj'), r.get('adj'))}"))
        print(ink.dim("    reasoning"))
        print(wrap(r["why"], 6))

    c = s["critic"]
    print("\n" + rule(ink))
    print(ink.red(ink.bold("  CRITIC")) + ink.dim("   (sees all four opinions plus the proposal)"))
    print(rule(ink))
    print(f"\n    verdict     {ink.amber(c['verdict'])}"
          + ink.dim(f"    confidence multiplier {c['mult']}"))
    print(f"    final call  {ink.amber(c['final'])}"
          + ink.dim(f"    recency weight {c.get('recency', '-')}"))
    print(ink.dim("    objection"))
    print(wrap(c["obj"], 6))
    print(ink.dim("    what would falsify it"))
    print(wrap(c["fals"], 6))

    print("\n" + rule(ink, "═"))
    changed = s["emitted"] != prop["label"]
    verdict = (ink.red("LABEL CHANGED by the panel") if changed
               else ink.green("label kept, confidence adjusted only"))
    print(ink.bold(f"  EMITTED: {s['emitted']}") + "     " + verdict)
    print(rule(ink, "═"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
