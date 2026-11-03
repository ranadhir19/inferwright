"""Remit: an agent's declared mandate as a first-class runtime object.

A remit is the combination of:
  - purpose:         why the agent exists
  - responsibilities: what it should do (task kinds)
  - authority:        what it is allowed to do (per-task max level)
  - constraints:     what it must not do (named plan constraints)
  - success criteria: how performance is measured (proxy vs true objective)
plus the operational boundaries: tool allow-list, data allow-list,
max delegation ratio, and temporary permission grants with TTLs.
"""

from dataclasses import dataclass, field

AUTH_LEVELS = {0: "read", 1: "recommend", 2: "modify", 3: "admin"}


@dataclass(frozen=True)
class Grant:
    """A temporary permission grant with a time-to-live (in steps)."""

    scope: str
    level: int
    issued: int
    ttl: int

    def expired(self, t: int) -> bool:
        return t >= self.issued + self.ttl

    def valid_at(self, t: int) -> bool:
        return self.issued <= t < self.issued + self.ttl


@dataclass
class Remit:
    """The declared mandate of an agent."""

    name: str
    purpose: str
    responsibilities: tuple
    authority: dict = field(default_factory=dict)
    constraints: tuple = ()
    proxy_metric: str = "completions"
    true_objective: str = "compliant_completions"
    tool_allowlist: tuple = ()
    data_allowlist: tuple = ()
    max_delegation: float = 0.3
    default_level: int = 0

    def max_level(self, kind: str) -> int:
        return self.authority.get(kind, self.default_level)

    def allows_task(self, kind: str) -> bool:
        return kind in self.responsibilities

    def allows_tool(self, tool: str) -> bool:
        return tool in self.tool_allowlist

    def allows_data(self, data: str) -> bool:
        return data in self.data_allowlist
