# Rough translator of LAPACK fixed-form Fortran routines into the C++ style
# of hwyBLAS's LAPACK port (1-based views, lower-case scalars). The output
# is a starting point that is reviewed and fixed by hand; constructs it does
# not understand are marked with /*CHECK ...*/. See doc/fortran_port.md.
#
# Usage: python f2cc.py file.f > out.cc
import re
import sys

NL = chr(10)

# Arguments kinds of the called routines: v = value, p = scalar output
# (pointer), a = array, i = INFO (dropped; returned). Names without the
# precision letter; real names map to the complex template name.
SIGS = {
    'GEMM': ('Gemm', 'vvvvvvavavvav'),
    'LACPY': ('Lacpy', 'vvvavav'),
    'LASCL': ('Lascl', 'vvvvvvvavi'),
    'LASET': ('Laset', 'vvvvvav'),
    'GEQRF': ('Geqrf', 'vvavaavi'),
    'GELQF': ('Gelqf', 'vvavaavi'),
    'UNGQR': ('Ungqr', 'vvvavaavi'),
    'UNGLQ': ('Unglq', 'vvvavaavi'),
    'GEBRD': ('Gebrd', 'vvavaaaaavi'),
    'UNGBR': ('Ungbr', 'vvvvavaavi'),
    'UNMBR': ('Unmbr', 'vvvvvvavaavavi'),
    'BDSQR': ('Bdsqr', 'vvvvvaaavavavai'),
    'LARTG': ('Lartg', 'vvppp'),
    'LAS2': ('Las2', 'vvvpp'),
    'LASV2': ('Lasv2', 'vvvpppppp'),
    'LASQ1': ('Lasq1', 'vaaai'),
    'LASQ2': ('Lasq2', 'vai'),
    'LASR': ('Lasr', 'vvvvvaaav'),
    'ROT': ('Rot', 'vavavvv'),
    'SCAL': ('Scal', 'vvav'),
    'SWAP': ('Swap', 'vavav'),
    'COPY': ('Copy', 'vavav'),
    'LASRT': ('Lasrt', 'vvai'),
    'XERBLA': ('Xerbla', 'vv'),
    'RSCAL': ('Rscal', 'vvav'),
    'GEMV': ('Gemv', 'vvvvavavvav'),
    'LACGV': ('Lacgv', 'vav'),
    'LARFG': ('Larfg', 'vpavp'),
    'LARF1F': ('Larf1f', 'vvvavvavva'),
    'LABRD': ('Labrd', 'vvvavaaaaavav'),
    'GEBD2': ('Gebd2', 'vvavaaaaai'),
    'LACRM': ('Lacrm', 'vvavavavaa'),
    'LARCM': ('Larcm', 'vvavavavaa'),
    'LACP2': ('Lacp2', 'vvvavav'),
    'BDSDC': ('Bdsdc', 'vvvaaavavaaaai'),
    'LASDQ': ('Lasdq', 'vvvvvvaaavavavai'),
    'LASD0': ('Lasd0', 'vvaaavavvaai'),
    'LASD1': ('Lasd1', 'vvvappavavaaai'),
    'LASD2': ('Lasd2', 'vvvpaavvavavaavavaaaaai'),
    'LASD3': ('Lasd3', 'vvvvaavaavavavavaaai'),
    'LASD4': ('Lasd4', 'vvaaavpai'),
    'LASD5': ('Lasd5', 'vaaavpa'),
    'LASD6': ('Lasd6', 'vvvvaaappaapavavaaaapppaai'),
    'LASD7': ('Lasd7', 'vvvvpaaaaaaavvaaaaapavavppi'),
    'LASD8': ('Lasd8', 'vvaaaaaavaai'),
    'LASDA': ('Lasda', 'vvvvaaavaaaaaaaavaaaaaai'),
    'LASDT': ('Lasdt', 'vppaaav'),
    'LAMRG': ('Lamrg', 'vvavva'),
    'LAED6': ('Laed6', 'vvvaavpi'),
    'DOT': ('Dot', 'vavav'),
    'NRM2': ('Nrm2', 'vav'),
    'LANST': ('Lanst', 'vvaa'),
    'LAPY2': ('Lapy2', 'vv'),
    'LAMC3': ('Lamc3', 'vv'),
}
REAL2CPLX = {'ORGQR': 'UNGQR', 'ORGLQ': 'UNGLQ', 'ORGBR': 'UNGBR',
             'ORMBR': 'UNMBR', 'ORMQR': 'UNMQR', 'ORMLQ': 'UNMLQ',
             'DROT': 'ROT', 'SROT': 'ROT', 'DSCAL': 'RSCAL'}
FUNCS = {
    'MAX': 'std::max', 'MIN': 'std::min', 'ABS': 'std::abs',
    'SQRT': 'std::sqrt', 'SIGN': 'Sign', 'DBLE': 'std::real',
    'REAL': 'std::real', 'INT': 'static_cast<blasint>', 'LSAME': 'Lsame',
    'MOD': 'Mod', 'DCONJG': 'Conj', 'CONJG': 'Conj', 'DIMAG': 'std::imag',
    'AIMAG': 'std::imag', 'NINT': 'Nint', 'DISNAN': 'std::isnan',
    'SISNAN': 'std::isnan', 'LOG': 'std::log', 'DCMPLX': 'T',
    'CMPLX': 'T',
}
OPS = [('.EQ.', ' == '), ('.NE.', ' != '), ('.LT.', ' < '), ('.LE.', ' <= '),
       ('.GT.', ' > '), ('.GE.', ' >= '), ('.AND.', ' && '), ('.OR.', ' || '),
       ('.NOT.', '!'), ('.TRUE.', 'true'), ('.FALSE.', 'false')]

TOKEN = re.compile(r"""
  (?P<num>(\d+\.(?![A-Z]+\.)\d*|\.\d+|\d+)([DE][-+]?\d+)?)
 |(?P<op>\.(EQ|NE|LT|LE|GT|GE|AND|OR|NOT|TRUE|FALSE)\.)
 |(?P<id>[A-Z][A-Z0-9_]*)
 |(?P<str>'[^']*')
 |(?P<pow>\*\*)
 |(?P<cat>//)
 |(?P<ch>[-+*/(),=:])
 |(?P<ws>\s+)
""", re.X)


def tokenize(s):
    out, pos = [], 0
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise ValueError('bad token at ' + s[pos:])
        pos = m.end()
        if m.lastgroup != 'ws':
            out.append((m.lastgroup, m.group()))
    return out


class Routine:
    def __init__(self):
        self.ints, self.reals, self.cplx, self.logs, self.chars = (
            set(), set(), set(), set(), set())
        self.arrays = {}  # name -> rank
        self.params = {}
        self.args = []
        self.name = ''
        self.externals = set()
        self.funcs_int = set()


def split_args(toks):
    """Splits a token list at top-level commas."""
    args, depth, cur = [], 0, []
    for t in toks:
        if t[1] == '(':
            depth += 1
        elif t[1] == ')':
            depth -= 1
        if t[1] == ',' and depth == 0:
            args.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        args.append(cur)
    return args


def matching(toks, i):
    depth = 0
    for j in range(i, len(toks)):
        if toks[j][1] == '(':
            depth += 1
        elif toks[j][1] == ')':
            depth -= 1
            if depth == 0:
                return j
    raise ValueError('unbalanced')


def num(s, r):
    if '.' in s or 'D' in s or 'E' in s:
        v = s.replace('D', 'e').replace('E', 'e')
        f = float(v)
        if f == int(f) and abs(f) < 1e6:
            v = repr(float(int(f)))
        return 'R(' + v + ')'
    return s


def is_int_expr(toks, r):
    for kind, val in toks:
        if kind == 'num' and ('.' in val or 'D' in val or 'E' in val):
            return False
        if kind == 'id' and val not in r.ints and val not in (
                'MAX', 'MIN', 'MOD', 'INT', 'NINT') and val not in r.funcs_int:
            return False
    return True


def expr(toks, r):
    """Translates an expression token list to C++."""
    out, i = [], 0
    while i < len(toks):
        kind, val = toks[i]
        if kind == 'num':
            out.append(num(val, r))
        elif kind == 'op':
            out.append(dict(OPS)[val])
        elif kind == 'str':
            inner = val[1:-1]
            out.append("'" + inner[0] + "'" if len(inner) >= 1 else "' '")
        elif kind == 'pow':
            base = out.pop()
            # Exponent: a number, an identifier, or a parenthesized group.
            if toks[i + 1][1] == '(':
                j = matching(toks, i + 1)
                ex = '(' + expr(toks[i + 2:j], r) + ')'
                i = j + 1
            else:
                ex = toks[i + 1][1]
                i += 2
                if ex.isdigit():
                    pass
                else:
                    ex = ex.lower()
            if ex == '2':
                out.append(base + ' * ' + base)
            else:
                out.append('std::pow(' + base + ', ' + ex + ')')
            continue
        elif kind == 'cat':
            out.append('/*CHECK CAT*/ + ')
        elif kind == 'ch' and val == '(':
            j = matching(toks, i)
            out.append('(' + expr(toks[i + 1:j], r) + ')')
            i = j + 1
            continue
        elif kind == 'id':
            nxt = toks[i + 1][1] if i + 1 < len(toks) else ''
            if nxt == '(':
                j = matching(toks, i + 1)
                inner = split_args(toks[i + 2:j])
                sub = ', '.join(expr(a, r) for a in inner)
                if val in r.arrays:
                    out.append(val + '(' + sub + ')')
                elif val in ('DBLE', 'REAL') and is_int_expr(toks[i + 2:j], r):
                    out.append('R(' + sub + ')')
                elif val in FUNCS:
                    f = FUNCS[val]
                    if val in ('MAX', 'MIN'):
                        if is_int_expr(toks[i + 2:j], r):
                            f += '<blasint>'
                        if len(inner) > 2:
                            out.append(f + '({' + sub + '})')
                        else:
                            out.append(f + '(' + sub + ')')
                    else:
                        out.append(f + '(' + sub + ')')
                else:
                    out.append(call_expr(val, inner, r, as_func=True))
                i = j + 1
                continue
            out.append(val.lower())
        else:
            out.append(val if val not in '+-*/=' else ' ' + val + ' ')
        i += 1
    s = ''.join(out)
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('( ', '(').replace(' )', ')')
    s = s.strip()
    # Unary minus.
    s = re.sub(r'(^|[(,=] |[(]|return |[<>] |== |!= )- ', lambda m: m.group(1) + '-', s)
    return s


def base_name(name):
    if name in ('XERBLA', 'LSAME', 'ILAENV'):
        return '', name
    p, rest = name[0], name[1:]
    if name in REAL2CPLX:
        return p, REAL2CPLX[name]
    if rest in REAL2CPLX:
        rest = REAL2CPLX[rest]
    return p, rest


def call_expr(name, args, r, as_func=False):
    """Translates CALL name(args) or a function reference."""
    p, base = base_name(name)
    if name == 'XERBLA':
        nm = args[0][0][1][2:-1]
        return 'Xerbla<T>("%s", %s)' % (nm, expr(args[1], r))
    if name == 'ILAENV':
        return 'Ilaenv<T>(' + ', '.join(expr(a, r) for a in args) + \
            ') /*CHECK ILAENV*/'
    if base == 'LAMCH':
        return 'Lamch<R>(' + expr(args[0], r) + ')'
    if base == 'LANGE':
        cname = 'Lange'
        kinds = 'vvvavv'
    elif base in SIGS:
        cname, kinds = SIGS[base]
    else:
        cname = base.capitalize()
        kinds = None
    # Real routine called from a complex one: real template argument.
    rout_cplx = r.name[0] in 'CZ'
    if rout_cplx and p in 'DS' and base not in ('LAMCH',):
        cname += '<R>'
    outs, ret = [], None
    for k, a in enumerate(args):
        kind = kinds[k] if kinds and k < len(kinds) else '?'
        if kind == 'i':
            ret = expr(a, r)
            continue
        s = arg(a, kind, r)
        outs.append(s)
    if kinds is None:
        outs.append('/*CHECK SIG*/')
    call = cname + '(' + ', '.join(outs) + ')'
    if ret is not None and not as_func:
        return ret + ' = ' + call
    return call


def arg(a, kind, r):
    """Translates one actual argument."""
    if len(a) >= 1 and a[0][0] == 'id' and a[0][1] in r.arrays:
        name = a[0][1]
        if len(a) == 1:
            return name.lower() if kind != 'v' else name.lower()
        if a[1][1] == '(' and matching(a, 1) == len(a) - 1:
            inner = split_args(a[2:-1])
            sub = ', '.join(expr(x, r) for x in inner)
            if kind == 'v':
                return name + '(' + sub + ')'
            return name + '.ptr(' + sub + ')'
    if kind == 'p' and len(a) == 1 and a[0][0] == 'id':
        return '&' + a[0][1].lower()
    if kind == 'a' and len(a) == 1 and a[0][0] == 'id':
        return '&' + a[0][1].lower() + '/*CHECK*/'
    return expr(a, r)


def lines_of(path):
    """Fortran statements: (label, text, comment) with continuations
    joined; comments are kept as separate entries."""
    raw = open(path, errors='ignore').read().split(NL)
    stmts = []
    for line in raw:
        line = line.rstrip()
        if not line:
            continue
        if line[0] in '*cC!':
            if line.startswith('*>') or line.startswith('*  =') or \
                    line.startswith('*  -') or line.startswith('*  --'):
                continue
            txt = line[1:].strip()
            stmts.append(('C', txt))
            continue
        if len(line) > 5 and line[5] not in ' 0' and stmts:
            # Continuation.
            for k in range(len(stmts) - 1, -1, -1):
                if stmts[k][0] == 'S':
                    stmts[k] = ('S', stmts[k][1], stmts[k][2] + ' ' +
                                line[6:].strip())
                    break
            continue
        if line[6:].strip().upper() == 'IMPLICIT NONE':
            continue
        label = line[:5].strip()
        stmts.append(('S', label, line[6:].strip()))
    return stmts


DECL = re.compile(r'^(INTEGER|DOUBLE PRECISION|REAL|COMPLEX\*16|COMPLEX|'
                  r'LOGICAL|CHARACTER(\*\d+)?)\s+(.*)$')


def parse_decls(stmts, r):
    for st in stmts:
        if st[0] != 'S':
            continue
        t = st[2]
        m = re.match(r'^SUBROUTINE\s+(\w+)\s*\((.*)\)', t)
        if m:
            r.name = m.group(1)
            r.args = [x.strip() for x in m.group(2).split(',')]
            continue
        m = re.match(r'^(INTEGER|LOGICAL|DOUBLE PRECISION|REAL|COMPLEX\*16)'
                     r'\s+FUNCTION\s+(\w+)\s*\((.*)\)', t)
        if m:
            r.name = m.group(2)
            r.args = [x.strip() for x in m.group(3).split(',')]
            continue
        m = DECL.match(t)
        if m:
            typ = m.group(1)
            toks = tokenize(m.group(3))
            for a in split_args(toks):
                nm = a[0][1]
                if len(a) > 1:
                    r.arrays[nm] = len(split_args(a[2:-1]))
                {'INTEGER': r.ints, 'DOUBLE PRECISION': r.reals,
                 'REAL': r.reals, 'COMPLEX*16': r.cplx, 'COMPLEX': r.cplx,
                 'LOGICAL': r.logs}.get(typ, r.chars).add(nm)
            continue
        m = re.match(r'^EXTERNAL\s+(.*)$', t)
        if m:
            for x in m.group(1).split(','):
                r.externals.add(x.strip())
            continue
        m = re.match(r'^PARAMETER\s*\((.*)\)$', t)
        if m:
            toks = tokenize(m.group(1))
            for a in split_args(toks):
                r.params[a[0][1]] = a[2:]


def ctype(nm, r):
    if nm in r.ints:
        return 'blasint'
    if nm in r.logs:
        return 'bool'
    if nm in r.chars:
        return 'char'
    if nm in r.cplx:
        return 'T'
    return 'R'


def translate(path):
    stmts = lines_of(path)
    r = Routine()
    parse_decls(stmts, r)
    out = []
    ind = [1]

    def emit(s):
        out.append('  ' * ind[0] + s)

    # Labels targeted by GO TO.
    gotos = set()
    for st in stmts:
        if st[0] == 'S':
            for m in re.finditer(r'GO\s*TO\s+(\d+)', st[2]):
                gotos.add(m.group(1))
    do_stack = []  # (label or None)
    started = False
    pending_comments = []
    for st in stmts:
        if st[0] == 'C':
            if started:
                txt = st[1]
                if txt.startswith('..') or txt.startswith('.. '):
                    continue
                if txt:
                    emit('// ' + txt)
                else:
                    pass
            continue
        _, label, t = st
        if re.match(r'^(SUBROUTINE|INTEGER|DOUBLE|REAL|COMPLEX|LOGICAL|'
                    r'CHARACTER|PARAMETER|EXTERNAL|INTRINSIC|IMPLICIT)\b', t):
            if not t.startswith(('REAL(', 'INTEGER(')):
                continue
        if t == 'END':
            continue
        if not started:
            started = True
            # Declarations.
            emit('// Parameters.')
            for k, v in r.params.items():
                emit('const %s %s = %s;' % (ctype(k, r), k.lower(),
                                            expr(v, r)))
            for nm in sorted(r.arrays):
                if nm in r.args:
                    rank = r.arrays[nm]
                    view = 'Matrix' if rank == 2 else 'Vector'
                    ld = ', ld' + nm.lower() if rank == 2 else ''
                    emit('const %s<%s> %s(%s%s);' % (view, ctype(nm, r), nm,
                                                     nm.lower(), ld))
                else:
                    emit('%s %s[/*CHECK*/];' % (ctype(nm, r), nm.lower()))
            for group, typ in ((r.ints, 'blasint'), (r.logs, 'bool'),
                               (r.reals, 'R'), (r.cplx, 'T'),
                               (r.chars, 'char')):
                names = sorted(n.lower() for n in group
                               if n not in r.arrays and n not in r.args and
                               n not in r.params and n != r.name and n not in r.externals)
                if names:
                    emit('%s %s;' % (typ, ', '.join(names)))
        stmt(label, t, r, emit, ind, do_stack, gotos)
    return r, NL.join(out)


def close_do(label, emit, ind, do_stack, gotos):
    while do_stack and do_stack[-1] == label:
        do_stack.pop()
        ind[0] -= 1
        emit('}')


def stmt(label, t, r, emit, ind, do_stack, gotos):
    # Labeled statement.
    is_do_end = label and label in do_stack
    if label and label in gotos and not is_do_end:
        ind[0] -= 1
        emit('L%s:' % label)
        ind[0] += 1
    if t == 'CONTINUE':
        if is_do_end:
            if label in gotos:
                emit('L%s:;  /*CHECK continue*/' % label)
            close_do(label, emit, ind, do_stack, gotos)
        return
    body(t, r, emit, ind, do_stack)
    if is_do_end:
        close_do(label, emit, ind, do_stack, gotos)


def body(t, r, emit, ind, do_stack):
    m = re.match(r'^IF\s*\((.*)\)\s*THEN$', t)
    if m:
        emit('if (%s) {' % expr(tokenize(m.group(1)), r))
        ind[0] += 1
        return
    m = re.match(r'^ELSE\s*IF\s*\((.*)\)\s*THEN$', t)
    if m:
        ind[0] -= 1
        emit('} else if (%s) {' % expr(tokenize(m.group(1)), r))
        ind[0] += 1
        return
    if t == 'ELSE':
        ind[0] -= 1
        emit('} else {')
        ind[0] += 1
        return
    if re.match(r'^END\s*(IF|DO)$', t):
        if t.replace(' ', '') == 'ENDDO':
            do_stack.pop()
        ind[0] -= 1
        emit('}')
        return
    m = re.match(r'^DO\s+(\d+)?\s*,?\s*(\w+)\s*=\s*(.*)$', t)
    if m and not t.startswith('DO WHILE'):
        lab, var, rng = m.group(1), m.group(2), m.group(3)
        parts = split_args(tokenize(rng))
        lo, hi = expr(parts[0], r), expr(parts[1], r)
        v = var.lower()
        if len(parts) == 3:
            st = expr(parts[2], r)
            if st.startswith('-'):
                st = st.replace(' ', '')
                inc = '--' + v if st == '-1' else v + ' -= ' + st[1:]
                emit('for (%s = %s; %s >= %s; %s) {' % (v, lo, v, hi, inc))
            else:
                emit('for (%s = %s; %s <= %s; %s += %s) {' % (v, lo, v, hi,
                                                            v, st))
        else:
            emit('for (%s = %s; %s <= %s; ++%s) {' % (v, lo, v, hi, v))
        do_stack.append(lab)
        ind[0] += 1
        return
    m = re.match(r'^DO\s+WHILE\s*\((.*)\)$', t)
    if m:
        emit('while (%s) {' % expr(tokenize(m.group(1)), r))
        do_stack.append(None)
        ind[0] += 1
        return
    m = re.match(r'^IF\s*\(', t)
    if m:
        toks = tokenize(t)
        j = matching(toks, 1)
        cond = expr(toks[2:j], r)
        rest = t[len(''.join(x[1] for x in toks[:j + 1])):]
        # Find the remainder after the condition in the source text.
        depth, k = 0, t.index('(')
        for k in range(t.index('('), len(t)):
            if t[k] == '(':
                depth += 1
            elif t[k] == ')':
                depth -= 1
                if depth == 0:
                    break
        rest = t[k + 1:].strip()
        inner = []
        body(rest, r, lambda s: inner.append(s.strip()), [0], do_stack)
        emit('if (%s) %s' % (cond, inner[0] if inner else '/*CHECK*/'))
        return
    m = re.match(r'^GO\s*TO\s+(\d+)$', t)
    if m:
        emit('goto L%s;' % m.group(1))
        return
    if t == 'RETURN':
        emit('return info;')
        return
    m = re.match(r'^CALL\s+(\w+)\s*\((.*)\)$', t)
    if m:
        toks = tokenize(m.group(2))
        emit(call_expr(m.group(1), split_args(toks), r) + ';')
        return
    m = re.match(r'^CALL\s+(\w+)$', t)
    if m:
        emit(m.group(1).capitalize() + '();')
        return
    # Assignment.
    toks = tokenize(t)
    depth = 0
    for k, tk in enumerate(toks):
        if tk[1] == '(':
            depth += 1
        elif tk[1] == ')':
            depth -= 1
        elif tk[1] == '=' and depth == 0:
            emit('%s = %s;' % (expr(toks[:k], r), expr(toks[k + 1:], r)))
            return
    emit('/*CHECK*/ ' + t)


if __name__ == '__main__':
    r, text = translate(sys.argv[1])
    print(text)
