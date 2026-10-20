# Fills translated routine bodies into a C++ skeleton: each line
#   //@@BODY name.cc|old=>new|~word=>new|%regex=>new|VOID@@
# is replaced by the body of <tr dir>/name.cc (from f2cc.py), with the given
# replacements and the common clean-ups applied. See doc/fortran_port.md.
# Usage: python assemble.py skeleton.cc out.cc [tr dir, default tr]
import os
import re
import sys

NL = chr(10)
TR = sys.argv[3] if len(sys.argv) > 3 else 'tr'

COMMON = [
    (r'\(R\((\d+)\.0\),\s*R\(0\.0\)\)', r'T(\1)'),
    (r'\bSign\(', 'std::copysign('),
    (r'\bstd::real\((\w+)\)', r'std::real(\1)'),
    (r'\bR\((\d+)\.0\)', r'R(\1)'),
    (r'\bR\((\d+\.\d+)e0\)', r'R(\1)'),
    (r'\bR\(0\.(\d+)\)', r'R(0.\1)'),
    (r'^\s*// End of \w+\s*$', ''),
    (r'^\s*// Parameters\.$', ''),
    (r'&\*(\w+)', r'\1'),
]


def body(spec):
    parts = spec.split('|')
    name = parts[0].strip()
    text = open(os.path.join(TR, name)).read()
    for rep in parts[1:]:
        if rep == 'VOID':
            text = text.replace('return info;', 'return;')
            text = re.sub(r'\s*// End of \w+\s*$', '', text)
            if text.endswith('return;'):
                text = text[:-len('return;')].rstrip()
            continue
        old, new = rep.split('=>')
        if old.startswith('~'):
            # Whole-word replacement.
            text = re.sub(r'\b' + old[1:] + r'\b', new, text)
        elif old.startswith('%'):
            # Regular expression.
            text = re.sub(old[1:], new.replace('\\n', NL), text)
        else:
            text = text.replace(old.replace('\\n', NL),
                                new.replace('\\n', NL))
    if not re.search(r'\bblasint info\b', text):
        text = re.sub(r'^(\s*)info = 0;', r'\1blasint info = 0;', text,
                      count=1, flags=re.M)
    # A label right before a closing brace needs a statement.
    text = re.sub(r'^(\s*L\d+):\n(\s*\})', r'\1:;\n\2', text, flags=re.M)
    out = []
    for line in text.split(NL):
        for pat, rep in COMMON:
            line = re.sub(pat, rep, line)
        out.append(line)
    return NL.join(out)


def main():
    src = open(sys.argv[1]).read()
    res = []
    for line in src.split(NL):
        m = re.match(r'^\s*//@@BODY (.*)@@\s*$', line)
        if m:
            res.append(body(m.group(1)))
        else:
            res.append(line)
    open(sys.argv[2], 'w', newline=NL).write(NL.join(res))


if __name__ == '__main__':
    main()
