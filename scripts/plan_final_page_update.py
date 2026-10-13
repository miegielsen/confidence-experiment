"""Plan one guarded Page update from the observed development snapshot."""
import difflib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parents[1] / "reports/banking77-findings"


def main():
    old = json.loads((OUT / "page-update-content.development.json").read_text(encoding="utf-8"))
    new = json.loads((OUT / "page-update-content.json").read_text(encoding="utf-8"))
    page = json.loads((OUT / "page-before-final.json").read_text(encoding="utf-8"))
    blocks = page["content"]["blocks"]
    assert not any(b["kind"] == "agent_instructions" for b in blocks)
    figures = {}
    for index, block in enumerate(blocks):
        if "](visualize:" in block["markdown"]:
            matches = [key for key, value in old["captions"].items() if value == blocks[index + 1]["markdown"]]
            assert len(matches) == 1
            figures[matches[0]] = block
    def desired(payload):
        answer = []
        for paragraph in re.split(r"\n\s*\n", payload["markdown"].strip()):
            if paragraph.startswith("# "): continue
            if paragraph.startswith("## "):
                slug = re.sub("[^a-z0-9]+", "-", paragraph[3:].lower()).strip("-")
                for key in payload["inserts"].get(slug, []):
                    answer += [figures[key]["markdown"], payload["captions"][key]]
            answer.append(paragraph)
        return answer
    before = [b["markdown"] for b in blocks]
    assert before == desired(old), "Intervening Page edits require reconciliation"
    after = desired(new)
    operations = []
    for tag, i, j, a, z in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if tag == "equal": continue
        common = min(j - i, z - a)
        for offset in range(common):
            b = blocks[i + offset]
            operations.append({"op": "replace_block_markdown", "block_id": b["id"], "expected_hash": b["hash"], "markdown": after[a + offset]})
        for b in blocks[i + common:j]:
            operations.append({"op": "delete_block", "block_id": b["id"], "expected_hash": b["hash"]})
        if z - a > common:
            operations.append({"op": "insert_markdown", "at": {"kind": "between_blocks",
                "preceding_block_id": blocks[j-1]["id"] if j else None,
                "following_block_id": blocks[j]["id"] if j < len(blocks) else None},
                "markdown": "\n\n".join(after[a + common:z])})
    result = {"page_id": page["content"]["page_id"], "operations": operations,
              "expected_markdown_blocks": after, "captions": new["captions"],
              "figures": {key: value for key, value in figures.items() if key in new["captions"]}}
    (OUT / "page-final-update-plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"operations": len(operations), "expected_blocks": len(after), "figure_count": len(new["captions"])}))
    if len(operations) > 100: raise ValueError("Edit plan exceeds atomic operation limit")


if __name__ == "__main__": main()
