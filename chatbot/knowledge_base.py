# chatbot/knowledge_base.py
"""
Loads the VegaKart policy documents and splits them into sections.

Each policy file is markdown: one "# Title" line, then "## Heading" sections that each
answer one question on their own. One section = one retrieval unit, so a search result
can be cited as "returns.md > Large appliance returns".
"""

import re
from dataclasses import dataclass
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent / "knowledge-base" / "policies"


@dataclass(frozen=True)
class Section:
    source: str  # "returns.md > Large appliance returns": the citation and the eval label
    title: str   # document title, e.g. "VegaKart Returns Policy"
    heading: str
    body: str

    @property
    def text(self) -> str:
        # What gets searched. The title and heading carry key words ("Refunds",
        # "Cash on Delivery") that the body may not repeat.
        return f"{self.title}\n{self.heading}\n{self.body}"


def load_sections(policy_dir: Path = POLICY_DIR) -> list[Section]:
    sections = []
    for path in sorted(policy_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", content, re.M)
        title = title_match.group(1).strip() if title_match else path.stem
        # Split at "## " headings; parts[0] is the title block before the first section.
        parts = re.split(r"^## (.+)$", content, flags=re.M)
        for heading, body in zip(parts[1::2], parts[2::2]):
            heading = heading.strip()
            sections.append(Section(f"{path.name} > {heading}", title, heading, body.strip()))
    return sections
