"""Render docs/TECHNICAL.md to "outputs/Chandal Chowkdi.pdf" for the submission form.

Needs the `markdown` package (documentation only, not part of the pipeline) and Microsoft Edge
or Google Chrome, which print the page headlessly.

    python scripts/build_doc_pdf.py
"""

import os
import subprocess
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "docs", "TECHNICAL.md")
OUT = os.path.join(ROOT, "outputs", "Chandal Chowkdi.pdf")
BROWSERS = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe")
STYLE = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt; } h2 { font-size: 12.5pt; margin: 14pt 0 4pt;
border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eee; } code, pre { font-family: Consolas, monospace; font-size: 8.5pt; }
pre { background: #f4f4f4; padding: 6pt; white-space: pre-wrap; } p, li { margin: 3pt 0; }
"""

if __name__ == '__main__':
    with open(SOURCE, encoding="utf-8") as handle:
        body = markdown.markdown(handle.read(), extensions=["tables", "fenced_code"])
    html_path = os.path.join(ROOT, "outputs", "technical.html")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html><html><head><meta charset='utf-8'><style>{}</style></head>"
                     "<body>{}</body></html>".format(STYLE, body))
    browser = next((b for b in BROWSERS if os.path.isfile(b)), None)
    if browser is None:
        sys.exit("no Edge or Chrome found; open {} and print it to PDF".format(html_path))
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf={}".format(OUT), "file:///" + html_path.replace(os.sep, "/")],
                   check=True, timeout=120)
    os.remove(html_path)
    print("wrote {} ({:.0f} KB)".format(OUT, os.path.getsize(OUT) / 1024))
