"""
Back-translate paper prose EN -> Japanese -> EN.
Uses [PNP] placeholders which survive Japanese tokenization intact.

Usage: python backtranslate.py sections/*.tex
"""
import re, sys, time
sys.path.insert(0, '/tmp/transenv/lib/python3.13/site-packages')
from deep_translator import GoogleTranslator

SKIP_ENVS = ('table','figure','tabular','equation','align',
             'itemize','enumerate','abstract')

TECH_TERMS = sorted([
    "knowledge distillation", "Knowledge distillation",
    "test-time augmentation", "teacher ensemble", "teacher ensembles",
    "intrinsic floor", "error floor", "significance frontier",
    "McNemar test", "McNemar", "Wilson confidence interval",
    "Wilson score interval", "Wilson CI",
    "Expected Calibration Error", "mean corruption accuracy",
    "squeeze-and-excitation", "pre-activation",
    "soft labels", "soft targets", "top-1 accuracy", "top-1",
    "benchmark saturation",
    "Devanagari", "CMATERdb", "NHCD", "DHCD",
    "PreAct", "SE-ResBlock", "SE-ResNet",
    "mCA", "TTA", "ECE",
], key=len, reverse=True)

LATEX_RE = re.compile(
    r'\$\$.*?\$\$'
    r'|\$[^$\n]+?\$'
    r'|~?\\cite\{[^}]+\}'
    r'|\\(?:ref|label|eqref)\{[^}]+\}'
    r'|\\(?:textbf|emph|textit|modelname|repourl|noindent)\{[^}]*\}'
    r'|\\[a-zA-Z]+\[[^\]]*\]\{[^}]*\}'
    r'|\\[a-zA-Z]+\{[^}]*\}'
    r'|\\[a-zA-Z]+\b'
    r'|~',
    re.DOTALL
)

def protect(text):
    slots = {}
    n = [0]
    def slot(val):
        k = f'[P{n[0]}P]'
        slots[k] = val
        n[0] += 1
        return k

    # Tech terms first (longest first)
    for term in TECH_TERMS:
        if term in text:
            text = text.replace(term, slot(term))

    # LaTeX commands
    text = LATEX_RE.sub(lambda m: slot(m.group(0)), text)
    return text, slots

def restore(text, slots):
    for k in sorted(slots, key=len, reverse=True):
        text = text.replace(k, slots[k])
    text = re.sub(r'  +', ' ', text)
    return text

def translate(text, src, tgt):
    if not text.strip(): return text
    words, chunks, cur = text.split(), [], []
    for w in words:
        if sum(len(x)+1 for x in cur)+len(w) > 4600:
            chunks.append(' '.join(cur)); cur=[]
        cur.append(w)
    if cur: chunks.append(' '.join(cur))
    out = []
    for c in chunks:
        try:
            out.append(GoogleTranslator(source=src, target=tgt).translate(c))
            time.sleep(0.4)
        except Exception as e:
            print(f'  [warn] {e}', file=sys.stderr)
            out.append(c)
    return ' '.join(out)

def backtranslate(para):
    plain = LATEX_RE.sub('', para)
    if len(plain.strip()) < 40: return para
    protected, slots = protect(para)
    ja  = translate(protected, 'en', 'ja')
    en  = translate(ja, 'ja', 'en')
    result = restore(en, slots)
    # Sanity: if any placeholder leaked through, fall back
    if re.search(r'\[P\d+P\]', result):
        leaked = re.findall(r'\[P\d+P\]', result)
        print(f'  [fallback] {len(leaked)} placeholder(s) leaked', file=sys.stderr)
        return para
    return result

_ENV_OPEN = re.compile(
    r'(\\begin\{(' + '|'.join(SKIP_ENVS) + r')[^}]*\})', re.IGNORECASE)

def iter_blocks(text):
    i = 0
    while i < len(text):
        m = _ENV_OPEN.match(text[i:])
        if m:
            env = m.group(2)
            close = f'\\end{{{env}}}'
            end = text.find(close, i+len(m.group(1)))
            if end != -1:
                end += len(close)
                yield text[i:end], False
                i = end; continue
        nxt = _ENV_OPEN.search(text, i)
        end = nxt.start() if nxt else len(text)
        for part in re.split(r'(\n\n+)', text[i:end]):
            prose = bool(part.strip()) and len(LATEX_RE.sub('', part).strip()) > 40
            yield part, prose
        i = end

def process(path):
    with open(path) as f: text = f.read()
    out, changed = [], 0
    for block, prose in iter_blocks(text):
        if prose:
            r = backtranslate(block)
            if r != block: changed += 1
            out.append(r)
        else:
            out.append(block)
    with open(path, 'w') as f: f.write(''.join(out))
    print(f'{path}: {changed} paragraphs changed')

if __name__ == '__main__':
    for p in sys.argv[1:]: process(p)
