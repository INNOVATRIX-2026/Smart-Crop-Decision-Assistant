"""Agronomic decision engine.

Pure computation over value objects: no network calls, no file I/O, no clock
reads. Providers (``src.weather``, ``src.soil``) fetch; this package decides.
That separation is what makes the engine unit-testable against FAO-56 worked
examples and deterministic for the same inputs.
"""

from __future__ import annotations
