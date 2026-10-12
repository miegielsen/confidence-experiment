"""Build guarded Page operations from an observed snapshot and local report.

Planning only: does not call Pages or mutate a Page. Existing graphics keep their
block identities; changed graphics are replaced through the visualization tool.
"""
import difflib
import json
from pathlib import Path
import re

OUT=Path(__file__).resolve().parents[1]/'reports/banking77-findings'


def main():
    old=json.loads((OUT/'page-before-opus-content.json').read_text(encoding='utf-8'))
    page=json.loads((OUT/'page-opus-edit-snapshot.json').read_text(encoding='utf-8'))
    new=json.loads((OUT/'page-update-content.json').read_text(encoding='utf-8'))
    blocks=page['content']['blocks']
    assert not any(b['kind']=='agent_instructions' for b in blocks)
    figures={}
    for i,b in enumerate(blocks):
        if '](visualize:' in b['markdown']:
            caption=blocks[i+1]['markdown']
            matches=[key for key,value in old['captions'].items() if value==caption]
            assert len(matches)==1,(i,caption)
            figures[matches[0]]=b
    def desired(payload,include_figures):
        answer=[]
        for paragraph in re.split(r'\n\s*\n',payload['markdown'].strip()):
            if paragraph.startswith('# '):continue
            if paragraph.startswith('## '):
                slug=re.sub('[^a-z0-9]+','-',paragraph[3:].lower()).strip('-')
                for key in payload['inserts'].get(slug,[]):
                    if include_figures and key in figures:answer.append(figures[key]['markdown'])
                    answer.append(payload['captions'][key])
            answer.append(paragraph)
        return answer
    before=[b['markdown'] for b in blocks]
    assert before==desired(old,True),'Page content changed outside the saved report; reconcile before editing'
    after=desired(new,True)
    operations=[]
    for tag,i,j,a,z in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
        if tag=='equal':continue
        count=min(j-i,z-a)
        for offset in range(count):
            b=blocks[i+offset]
            assert '](visualize:' not in b['markdown']
            operations.append(dict(op='replace_block_markdown',block_id=b['id'],expected_hash=b['hash'],markdown=after[a+offset]))
        for b in blocks[i+count:j]:
            assert '](visualize:' not in b['markdown']
            operations.append(dict(op='delete_block',block_id=b['id'],expected_hash=b['hash']))
        if z-a>count:
            operations.append(dict(op='insert_markdown',at=dict(kind='between_blocks',preceding_block_id=blocks[j-1]['id'] if j else None,following_block_id=blocks[j]['id'] if j<len(blocks) else None),markdown='\n\n'.join(after[a+count:z])))
    assert len(operations)<=100
    result={'page_id':page['content']['page_id'],'operations':operations,
            'expected_native':desired(new,False),'figure_blocks':figures,
            'new_figures':[key for key in new['captions'] if key not in figures]}
    (OUT/'page-opus-update-plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'operations':len(operations),'native_blocks':len(result['expected_native']),'new_figures':result['new_figures']}))


if __name__=='__main__':main()
