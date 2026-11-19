from dataclasses import dataclass, field

@dataclass
class Record:
    id: str
    text: str
    lang: str
    source: str
    gate: bool | None
    target_groups: list[str] | None
    hate_types: list[str] | None
    severity: str | None

    legal: list[str] | None = None
    raw: dict | str = field(default_factory=dict)
