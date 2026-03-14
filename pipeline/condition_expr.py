"""
Portable condition expression evaluator for the pipeline IR.

Expressions are small, safe, declarative predicates that can be stored in
JSON plan files and evaluated against a PipelineContext at runtime. They
replace the hardcoded ``condition_id -> Python function`` registry so that
the IR can carry executable guard logic without new Python code.

Every primitive or accessor resolves to a *signal*. Signals can be composed
with a small *calculator* that supports five basic math operators before the
final boolean guard is evaluated.

Grammar (case-insensitive keywords, case-sensitive attribute names)::

    expr            := or_expr
    or_expr         := and_expr ( 'OR' and_expr )*
    and_expr        := not_expr ( 'AND' not_expr )*
    not_expr        := 'NOT' not_expr | comparison_expr
    comparison_expr := signal_expr (
                           'IS_NONE'
                         | 'IS_NOT_NONE'
                         | 'CONTAINS' signal_expr
                         | ( '==' | '!=' | '>' | '>=' | '<' | '<=' ) signal_expr
                       )?
    signal_expr     := add_expr
    add_expr        := mul_expr ( ( '+' | '-' ) mul_expr )*
    mul_expr        := unary_expr ( ( '*' | '/' | '%' | 'MOD' ) unary_expr )*
    unary_expr      := ( '+' | '-' ) unary_expr | primary
    primary         := '(' expr ')'
                     | accessor
                     | primitive
                     | 'GATE_OVERRIDE'
    accessor        := IDENT ( '.' IDENT )*
    primitive       := STRING | INTEGER | FLOAT | 'TRUE' | 'FALSE' | 'NONE'
    STRING          := '"' ... '"'
    IDENT           := [a-zA-Z_][a-zA-Z0-9_]*

Security: only whitelisted dotted attribute access is performed on the
context object. No ``__dunder__`` access, no calls except to pre-approved
zero-arg methods, no mutation.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<STRING>"[^"]*")         |   # quoted string
    (?P<FLOAT>\d+\.\d+)        |   # float literal
    (?P<INT>\d+)                |   # integer literal
    (?P<OP>==|!=|>=|<=|>|<)    |   # comparison operators
    (?P<ARITH>[+\-*/%])         |   # calculator operators
    (?P<DOT>\.)                 |   # dot accessor
    (?P<LPAREN>\()              |   # left paren
    (?P<RPAREN>\))              |   # right paren
    (?P<COMMA>,)                |   # comma
    (?P<WORD>[a-zA-Z_][a-zA-Z0-9_]*) |  # keyword or identifier
    (?P<WS>\s+)                     # whitespace (skipped)
    """,
    re.VERBOSE,
)

_KEYWORDS = frozenset({
    "AND", "OR", "NOT",
    "IS_NONE", "IS_NOT_NONE",
    "CONTAINS", "MOD",
    "GATE_OVERRIDE", "TRUE", "FALSE", "NONE",
})

Token = Tuple[str, str]  # (type, value)


def _tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    text = str(expr or "").strip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise ExprSyntaxError(f"Unexpected character {text[pos]!r} at position {pos}")
        pos = m.end()
        if m.lastgroup == "WS":
            continue
        if m.lastgroup == "WORD":
            word = m.group()
            if word.upper() in _KEYWORDS:
                tokens.append(("KW", word.upper()))
            else:
                tokens.append(("IDENT", word))
        elif m.lastgroup == "STRING":
            tokens.append(("STRING", m.group()[1:-1]))  # strip quotes
        elif m.lastgroup == "INT":
            tokens.append(("INT", m.group()))
        elif m.lastgroup == "FLOAT":
            tokens.append(("FLOAT", m.group()))
        elif m.lastgroup == "OP":
            tokens.append(("OP", m.group()))
        elif m.lastgroup == "ARITH":
            tokens.append(("ARITH", m.group()))
        elif m.lastgroup == "DOT":
            tokens.append(("DOT", "."))
        elif m.lastgroup == "LPAREN":
            tokens.append(("LPAREN", "("))
        elif m.lastgroup == "RPAREN":
            tokens.append(("RPAREN", ")"))
        elif m.lastgroup == "COMMA":
            tokens.append(("COMMA", ","))
    return tokens


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

class _Expr:
    """Base class for AST nodes."""


class _SignalLit(_Expr):
    __slots__ = ("value",)
    def __init__(self, value: Any):
        self.value = value
    def __repr__(self):
        return f"SignalLit({self.value!r})"


class _GateOverride(_Expr):
    def __repr__(self):
        return "GateOverride()"


class _Accessor(_Expr):
    __slots__ = ("parts",)
    def __init__(self, parts: List[str]):
        self.parts = parts
    def __repr__(self):
        return f"Accessor({'.'.join(self.parts)})"


class _CalculatorUnary(_Expr):
    __slots__ = ("op", "child")
    def __init__(self, op: str, child: _Expr):
        self.op = op
        self.child = child


class _CalculatorBinary(_Expr):
    __slots__ = ("left", "op", "right")
    def __init__(self, left: _Expr, op: str, right: _Expr):
        self.left = left
        self.op = op
        self.right = right


class _Not(_Expr):
    __slots__ = ("child",)
    def __init__(self, child: _Expr):
        self.child = child


class _And(_Expr):
    __slots__ = ("children",)
    def __init__(self, children: List[_Expr]):
        self.children = children


class _Or(_Expr):
    __slots__ = ("children",)
    def __init__(self, children: List[_Expr]):
        self.children = children


class _IsNone(_Expr):
    __slots__ = ("signal",)
    def __init__(self, signal: _Expr):
        self.signal = signal


class _IsNotNone(_Expr):
    __slots__ = ("signal",)
    def __init__(self, signal: _Expr):
        self.signal = signal


class _Contains(_Expr):
    __slots__ = ("signal", "needle")
    def __init__(self, signal: _Expr, needle: _Expr):
        self.signal = signal
        self.needle = needle


class _Compare(_Expr):
    __slots__ = ("left", "op", "right")
    def __init__(self, left: _Expr, op: str, right: _Expr):
        self.left = left
        self.op = op
        self.right = right


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ExprSyntaxError(ValueError):
    """Raised when a condition expression cannot be parsed."""


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tok_type: str, tok_value: Optional[str] = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise ExprSyntaxError(f"Expected {tok_type} but reached end of expression")
        if tok[0] != tok_type or (tok_value is not None and tok[1] != tok_value):
            raise ExprSyntaxError(f"Expected {tok_type}({tok_value!r}) but got {tok}")
        return self._advance()

    def parse(self) -> _Expr:
        result = self._or_expr()
        if self.pos < len(self.tokens):
            raise ExprSyntaxError(f"Unexpected token after expression: {self.tokens[self.pos]}")
        return result

    def _or_expr(self) -> _Expr:
        children = [self._and_expr()]
        while self._peek() == ("KW", "OR"):
            self._advance()
            children.append(self._and_expr())
        return children[0] if len(children) == 1 else _Or(children)

    def _and_expr(self) -> _Expr:
        children = [self._not_expr()]
        while self._peek() == ("KW", "AND"):
            self._advance()
            children.append(self._not_expr())
        return children[0] if len(children) == 1 else _And(children)

    def _not_expr(self) -> _Expr:
        if self._peek() == ("KW", "NOT"):
            self._advance()
            return _Not(self._not_expr())
        return self._comparison_expr()

    def _comparison_expr(self) -> _Expr:
        left = self._signal_expr()
        nxt = self._peek()
        if nxt == ("KW", "IS_NOT_NONE"):
            self._advance()
            return _IsNotNone(left)
        if nxt == ("KW", "IS_NONE"):
            self._advance()
            return _IsNone(left)
        if nxt == ("KW", "CONTAINS"):
            self._advance()
            return _Contains(left, self._signal_expr())
        if nxt is not None and nxt[0] == "OP":
            op = self._advance()[1]
            return _Compare(left, op, self._signal_expr())
        return left

    def _signal_expr(self) -> _Expr:
        return self._add_expr()

    def _add_expr(self) -> _Expr:
        expr = self._mul_expr()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "ARITH" or tok[1] not in {"+", "-"}:
                return expr
            op = self._advance()[1]
            expr = _CalculatorBinary(expr, op, self._mul_expr())

    def _mul_expr(self) -> _Expr:
        expr = self._unary_expr()
        while True:
            tok = self._peek()
            if tok is None:
                return expr
            if tok == ("KW", "MOD"):
                op = self._advance()[1]
                expr = _CalculatorBinary(expr, op, self._unary_expr())
                continue
            if tok[0] == "ARITH" and tok[1] in {"*", "/", "%"}:
                op = self._advance()[1]
                expr = _CalculatorBinary(expr, op, self._unary_expr())
                continue
            return expr

    def _unary_expr(self) -> _Expr:
        tok = self._peek()
        if tok is not None and tok[0] == "ARITH" and tok[1] in {"+", "-"}:
            op = self._advance()[1]
            return _CalculatorUnary(op, self._unary_expr())
        return self._primary()

    def _primary(self) -> _Expr:
        tok = self._peek()
        if tok is None:
            raise ExprSyntaxError("Unexpected end of expression")
        if tok[0] == "LPAREN":
            self._advance()
            inner = self._or_expr()
            self._expect("RPAREN")
            return inner
        if tok == ("KW", "GATE_OVERRIDE"):
            self._advance()
            return _GateOverride()
        if tok[0] == "IDENT":
            return self._accessor()
        if tok[0] in {"STRING", "INT", "FLOAT"} or tok in {
            ("KW", "TRUE"),
            ("KW", "FALSE"),
            ("KW", "NONE"),
        }:
            return _SignalLit(self._value())
        raise ExprSyntaxError(f"Unexpected token {tok}")

    def _accessor(self) -> _Accessor:
        parts = [self._expect("IDENT")[1]]
        while self._peek() == ("DOT", "."):
            self._advance()
            parts.append(self._expect("IDENT")[1])
        return _Accessor(parts)

    def _value(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ExprSyntaxError("Expected value but reached end")
        if tok[0] == "STRING":
            self._advance()
            return tok[1]
        if tok[0] == "INT":
            self._advance()
            return int(tok[1])
        if tok[0] == "FLOAT":
            self._advance()
            return float(tok[1])
        if tok == ("KW", "TRUE"):
            self._advance()
            return True
        if tok == ("KW", "FALSE"):
            self._advance()
            return False
        if tok == ("KW", "NONE"):
            self._advance()
            return None
        raise ExprSyntaxError(f"Expected value, got {tok}")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

# Attributes that are NEVER accessible (security boundary).
_BLOCKED_NAMES = frozenset({"__class__", "__dict__", "__module__", "__init__"})

# Zero-arg methods that the evaluator is allowed to call.
_CALLABLE_WHITELIST = frozenset({
    "gate_override_enabled",
    "early_gates_passed",
    "all_base_gates_passed",
    "wave_stage_ready",
})


def _resolve_accessor(ctx: Any, parts: List[str]) -> Any:
    """Safely walk a dotted attribute path on *ctx*."""
    obj = ctx
    for part in parts:
        if part in _BLOCKED_NAMES or part.startswith("__"):
            raise ExprSecurityError(f"Access to {part!r} is blocked")
        val = getattr(obj, part, _SENTINEL)
        if val is _SENTINEL:
            # Try dict-style access for dict-like objects.
            if isinstance(obj, dict):
                val = obj.get(part, _SENTINEL)
            if val is _SENTINEL:
                return None
        # Auto-call whitelisted zero-arg methods.
        if callable(val) and part in _CALLABLE_WHITELIST:
            val = val()
        obj = val
    return obj


_SENTINEL = object()


class ExprSecurityError(RuntimeError):
    """Raised when an expression tries to access a blocked attribute."""


def _coerce_calculator_signal(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _apply_calculator_unary(op: str, value: Any) -> Any:
    numeric = _coerce_calculator_signal(value)
    if numeric is None:
        return None
    if op == "+":
        return numeric
    if op == "-":
        return -numeric
    return None


def _apply_calculator_binary(op: str, left: Any, right: Any) -> Any:
    lhs = _coerce_calculator_signal(left)
    rhs = _coerce_calculator_signal(right)
    if lhs is None or rhs is None:
        return None
    if op == "+":
        return lhs + rhs
    if op == "-":
        return lhs - rhs
    if op == "*":
        return lhs * rhs
    if op == "/":
        return None if rhs == 0 else (lhs / rhs)
    if op in {"MOD", "%"}:
        return None if rhs == 0 else (lhs % rhs)
    return None


def _eval_node(node: _Expr, ctx: Any) -> Any:
    if isinstance(node, _SignalLit):
        return node.value

    if isinstance(node, _GateOverride):
        fn = getattr(ctx, "gate_override_enabled", None)
        return bool(fn()) if callable(fn) else False

    if isinstance(node, _Accessor):
        return _resolve_accessor(ctx, node.parts)

    if isinstance(node, _CalculatorUnary):
        return _apply_calculator_unary(node.op, _eval_node(node.child, ctx))

    if isinstance(node, _CalculatorBinary):
        return _apply_calculator_binary(
            node.op,
            _eval_node(node.left, ctx),
            _eval_node(node.right, ctx),
        )

    if isinstance(node, _IsNone):
        val = _eval_node(node.signal, ctx)
        return val is None

    if isinstance(node, _IsNotNone):
        val = _eval_node(node.signal, ctx)
        return val is not None

    if isinstance(node, _Contains):
        val = _eval_node(node.signal, ctx)
        needle = _eval_node(node.needle, ctx)
        haystack_text = "" if val is None else str(val)
        needle_text = "" if needle is None else str(needle)
        return needle_text in haystack_text

    if isinstance(node, _Compare):
        return _compare(
            _eval_node(node.left, ctx),
            node.op,
            _eval_node(node.right, ctx),
        )

    if isinstance(node, _Not):
        return not bool(_eval_node(node.child, ctx))

    if isinstance(node, _And):
        return all(bool(_eval_node(c, ctx)) for c in node.children)

    if isinstance(node, _Or):
        return any(bool(_eval_node(c, ctx)) for c in node.children)

    raise ExprSyntaxError(f"Unknown AST node type: {type(node).__name__}")


def _compare(left: Any, op: str, right: Any) -> bool:
    try:
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
    except TypeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_condition_expr(expr: str) -> _Expr:
    """Parse a condition expression string into an AST.

    Raises :class:`ExprSyntaxError` on malformed input.
    """
    text = str(expr or "").strip()
    if not text:
        raise ExprSyntaxError("Empty expression")
    tokens = _tokenize(text)
    return _Parser(tokens).parse()


def evaluate_condition_expr(expr: str, ctx: Any) -> bool:
    """Parse and evaluate a condition expression against *ctx*.

    Returns True/False.  Empty/blank expressions return True (unconditional).
    """
    text = str(expr or "").strip()
    if not text:
        return True
    ast = parse_condition_expr(text)
    return bool(_eval_node(ast, ctx))


def validate_condition_expr(expr: str) -> Optional[str]:
    """Check whether *expr* is syntactically valid.

    Returns ``None`` if valid, or an error message string.
    """
    text = str(expr or "").strip()
    if not text:
        return None
    try:
        parse_condition_expr(text)
        return None
    except ExprSyntaxError as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Expression synthesis helpers — generate expression strings from Python state
# ---------------------------------------------------------------------------

def expr_for_condition_id(condition_id: str) -> str:
    """Return a portable expression string for a known condition_id.

    This bridges the old hardcoded condition registry to the new expression
    system.  Once all plans use ``condition_expr`` natively, this mapping
    can be retired.
    """
    _MAP = {
        "orchestration.mode_has_generator":
            'orchestration_mode CONTAINS "g"',
        "gates.pregestation_passed":
            "GATE_OVERRIDE OR gate_pregestation.passed",
        "gates.early_passed":
            "GATE_OVERRIDE OR early_gates_passed",
        "gates.all_base_passed":
            "GATE_OVERRIDE OR all_base_gates_passed",
        "gates.wave_stage_ready":
            "(GATE_OVERRIDE AND transformer IS_NOT_NONE) OR wave_stage_ready",
        "data.pregestation_rebuild_due":
            "data._preg_last_build_round < 0 OR "
            "(ctx.total_rounds_completed - data._preg_last_build_round) >= preg_cfg.rebuild_every_n_rounds",
        "data.gestation_rebuild_due":
            "(GATE_OVERRIDE OR gate_pregestation.passed) AND "
            "(data._gest_last_build_round < 0 OR "
            "(ctx.total_rounds_completed - data._gest_last_build_round) >= gest_cfg.rebuild_every_n_rounds)",
        "data.berkeley_refresh_due":
            "(GATE_OVERRIDE OR early_gates_passed) AND "
            "(data._bdata_last_build_round < 0 OR "
            "(ctx.total_rounds_completed - data._bdata_last_build_round) >= bdata_cfg.rebuild_every_n_rounds)",
    }
    return _MAP.get(str(condition_id or "").strip(), "")


def expr_for_node_should_run(node: Any) -> str:
    """Synthesize a ``run_condition_expr`` from a live PipelineNode.

    Inspects the node's class hierarchy and instance state to produce
    a declarative expression capturing the ``should_run()`` logic.
    """
    from pipeline.nodes.base import GatedNode, OneTimeNode

    parts: List[str] = []

    # GatedNode base: all required gates must have passed.
    if isinstance(node, GatedNode):
        gates = list(getattr(node, "required_gates", []) or [])
        for gate_attr in gates:
            parts.append(f"{gate_attr}.passed")

    # Node-specific extra predicates: inspect the concrete class.
    node_id = str(getattr(node, "node_id", "") or "")
    cls_name = type(node).__name__

    # PregestationEvalNode
    if cls_name == "PregestationEvalNode":
        parts.append("pregestation_eval_loader IS_NOT_NONE")
        parts.append("(classifier IS_NOT_NONE OR gate_classifier IS_NOT_NONE)")

    # GestationEvalNode
    elif cls_name == "GestationEvalNode":
        parts.append("gestation_eval_loader IS_NOT_NONE")
        parts.append("(classifier IS_NOT_NONE OR gate_classifier IS_NOT_NONE)")

    # BerkeleyGateNode
    elif cls_name == "BerkeleyGateNode":
        parts.append("payload_validation_loader IS_NOT_NONE")
        parts.append("(classifier IS_NOT_NONE OR gate_classifier IS_NOT_NONE)")

    # TransformerGateNode
    elif cls_name == "TransformerGateNode":
        parts.append("transformer IS_NOT_NONE")
        parts.append("classifier IS_NOT_NONE")

    # GeneratorGateNode
    elif cls_name == "GeneratorGateNode":
        parts.append("generator IS_NOT_NONE")
        parts.append("classifier IS_NOT_NONE")

    # WaveGateNode
    elif cls_name == "WaveGateNode":
        parts.append('orchestration_mode CONTAINS "w"')
        parts.append("classifier IS_NOT_NONE")

    # CheckpointSaveNode
    elif cls_name == "CheckpointSaveNode":
        period = int(getattr(node, "save_every_n_rounds", 1) or 1)
        if period > 1:
            parts.append(f"round_id MOD {period} == 0")

    if not parts:
        return ""

    return " AND ".join(parts)
