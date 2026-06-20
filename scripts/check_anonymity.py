"""Fail if the anonymized PDF leaks any identifying string."""
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
anon_pdf = repo_root / "paper" / "main-anon.pdf"

if not anon_pdf.exists():
    print(f"BLOCKED -- {anon_pdf} not found; build it first with latexmk")
    sys.exit(2)

result = subprocess.run(
    ["pdftotext", str(anon_pdf), "-"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("BLOCKED -- pdftotext failed:", result.stderr.strip())
    sys.exit(2)

txt = result.stdout.lower()
forbidden = [
    "ampixa",
    "barnamala",
    "thapa",
    "ashish",
    "voidash",
    "github.com/ampixa",
]
hits = [w for w in forbidden if w in txt]
if hits:
    print("ANON LEAK -- found:", hits)
    sys.exit(1)

print("ANON CLEAN -- no identifying strings in main-anon.pdf")
