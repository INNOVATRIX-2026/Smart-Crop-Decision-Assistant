"""Computation traces that make every recommendation auditable.

The UI renders these objects; it never re-derives the arithmetic. That is the
whole point — an explanation panel that recomputes its own numbers will silently
drift out of sync with the engine it claims to explain. Here the engine emits the
trace as it works, so what the farmer reads is necessarily what the engine did.

Each :class:`Step` is one line of arithmetic with its units. Each
:class:`Explanation` bundles the inputs used, the ordered steps, the threshold
compared against, and the resulting conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Step:
    """One line of arithmetic in a derivation."""

    label: str          # what this step computes, e.g. "Total available water"
    expression: str     # how, with numbers substituted in
    value: float        # the result
    unit: str = ""

    def render(self) -> str:
        val = f"{self.value:,.1f}".rstrip("0").rstrip(".")
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.label}: {self.expression} = {val}{unit}"


@dataclass
class Explanation:
    """Full audit trail for a single recommendation."""

    inputs: dict[str, str] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    threshold: str | None = None
    conclusion: str = ""
    sources: list[str] = field(default_factory=list)

    # -- builder helpers (chainable) ------------------------------------
    def add_input(self, name: str, value: float | str, unit: str = "") -> Explanation:
        if isinstance(value, float):
            shown = f"{value:,.2f}".rstrip("0").rstrip(".")
        else:
            shown = str(value)
        self.inputs[name] = f"{shown} {unit}".strip()
        return self

    def add_step(self, label: str, expression: str, value: float, unit: str = "") -> Explanation:
        self.steps.append(Step(label=label, expression=expression, value=value, unit=unit))
        return self

    def set_threshold(self, text: str) -> Explanation:
        self.threshold = text
        return self

    def conclude(self, text: str) -> Explanation:
        self.conclusion = text
        return self

    def cite(self, *sources: str) -> Explanation:
        for s in sources:
            if s not in self.sources:
                self.sources.append(s)
        return self

    def render_lines(self) -> list[str]:
        """Flatten to display lines, for the Streamlit panel or a CLI dump."""
        lines: list[str] = []
        if self.inputs:
            lines.append("Inputs used:")
            lines += [f"  • {k} = {v}" for k, v in self.inputs.items()]
        if self.steps:
            lines.append("Computation:")
            lines += [f"  • {s.render()}" for s in self.steps]
        if self.threshold:
            lines.append(f"Threshold: {self.threshold}")
        if self.conclusion:
            lines.append(f"Conclusion: {self.conclusion}")
        if self.sources:
            lines.append(f"Sources: {', '.join(self.sources)}")
        return lines
