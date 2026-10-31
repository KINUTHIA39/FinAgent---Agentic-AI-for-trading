"""Rebuild deck/deck.html from the template plus the current results.

Kept separate so a rerun of the contest does not mean hand-editing a 90 KB file
with a JSON blob in the middle of it. Edit deck/deck.template.html; this fills
in the payload.
"""
from __future__ import annotations

import json
import re
import pathlib

from . import config, figures


def _render(name: str, data: dict) -> None:
    root = config.ROOT
    tpl = (root / "deck" / f"{name}.template.html").read_text(encoding="utf-8")
    config.paths()["results"].joinpath(f"{name}_data.json").write_text(
        json.dumps(data, separators=(",", ":"), default=str), encoding="utf-8")

    # A literal </script> inside the JSON would close the block early. The
    # replacement is a backslash followed by a slash, which JSON readers accept
    # and the HTML parser does not treat as a closing tag.
    blob = json.dumps(data, separators=(",", ":"), default=str).replace("</", r"<\/")
    out = root / "deck" / f"{name}.html"
    out.write_text(tpl.replace("__DATA__", blob), encoding="utf-8")
    print(f"written: {out}  ({out.stat().st_size / 1024:.0f} KB)")


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{desc}">
<meta name="color-scheme" content="light">
</head>
<body>
{body}
</body>
</html>
"""

# The Artifact runtime injects a doctype, head and a CSS reset around whatever
# it is given. Cloudflare Pages serves the file verbatim, so the standalone
# build has to supply that skeleton itself -- and the reset too, or the pages
# inherit the browser's default margins and the sticky header sits wrong.
RESET = """<style>
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0}
img,svg,video,canvas{display:block;max-width:100%}
button,input,select,textarea{font:inherit;color:inherit}
</style>
"""

PAGES = {
    "index": ("dashboard",
              "Watch an LLM agent panel and Hamilton's 1989 Markov switching "
              "model call S&P 500 regimes day by day, 2005-2023."),
    "deck": ("deck",
             "A 1989 two-state Markov switching model beats a three-state HMM "
             "and an LLM agent panel at detecting S&P 500 regimes."),
}


def _standalone() -> None:
    """Wrap the built pages for Cloudflare Pages into dist/."""
    root = config.ROOT
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    for out_name, (src, desc) in PAGES.items():
        body = (root / "deck" / f"{src}.html").read_text(encoding="utf-8")

        # The artifact templates open with <title> and the font <link>s, which
        # belong in the head. Left in the body they are a parse error, and the
        # font requests only start once the parser reaches them -- so the page
        # renders a frame in the fallback stack first. Hoist them.
        head_bits = []
        for pat in (r"<title>.*?</title>", r'<link rel="[^"]*"[^>]*>'):
            for m in re.findall(pat, body, re.S):
                head_bits.append(m)
                body = body.replace(m, "", 1)

        # Slot the reset in ahead of the page's own styles so the page wins.
        page = SHELL.format(desc=desc, body=RESET + body.lstrip())
        page = page.replace("</head>", "\n".join(head_bits) + "\n</head>")
        p = dist / f"{out_name}.html"
        p.write_text(page, encoding="utf-8")
        print(f"  dist/{out_name}.html  ({p.stat().st_size / 1024:.0f} KB)")

    # Long cache on nothing: these are single documents that change on rebuild.
    (dist / "_headers").write_text(
        "/*\n  Cache-Control: public, max-age=0, must-revalidate\n"
        "  X-Content-Type-Options: nosniff\n", encoding="utf-8")


def main() -> None:
    cfg = config.load()
    _render("deck", figures.build(cfg))
    _render("dashboard", figures.build_dashboard(cfg))
    _standalone()


if __name__ == "__main__":
    main()
