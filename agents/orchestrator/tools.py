"""
agents/orchestrator/tools.py
============================
Tools the ReAct agent can call. Both are deterministic and run locally.

  lookup[term]          sentences of the task context that mention the term
  calculate[expression] the value of an arithmetic expression

The calculator walks a whitelisted AST; it never calls eval(), so model
output cannot execute code.
"""

from __future__ import annotations

import ast
import math
import operator
import re

MAX_LOOKUP_SENTENCES = 3
MAX_EXPONENT = 64

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class ToolError(ValueError):
    """A tool could not run on the given input."""


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("exponent too large")
        return float(_BINARY[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_evaluate(node.operand)))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return float(_FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args)))
    raise ToolError(f"unsupported expression: {ast.dump(node)[:60]}")


def calculate(expression: str) -> str:
    """Evaluate arithmetic like '6 * 2.5 + 1', '20000 * 0.85**2' or 'sqrt(50)'."""
    cleaned = expression.replace(",", "").replace("×", "*").replace("^", "**").replace("÷", "/")
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", cleaned)
    try:
        value = _evaluate(ast.parse(cleaned.strip(), mode="eval"))
    except (SyntaxError, ArithmeticError, TypeError, ValueError) as err:
        raise ToolError(f"could not evaluate {expression!r}: {err}") from err
    if math.isnan(value) or math.isinf(value):
        raise ToolError(f"{expression!r} has no finite value")
    return f"{value:.10g}"


def lookup(term: str, context: str | None) -> str:
    """Return up to three context sentences mentioning ``term`` (case-insensitive)."""
    if not context or not context.strip():
        return "No context is available for this question."
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    words = [w for w in re.findall(r"\w+", term.lower()) if len(w) > 2] or [term.lower().strip()]
    scored = [
        (sum(word in sentence.lower() for word in words), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    hits = [s for s in sorted(scored, key=lambda x: (-x[0], x[1])) if s[0] > 0]
    if not hits:
        return f"No sentence in the context mentions {term!r}."
    chosen = sorted(hits[:MAX_LOOKUP_SENTENCES], key=lambda x: x[1])
    return " ".join(sentence for _, _, sentence in chosen)
