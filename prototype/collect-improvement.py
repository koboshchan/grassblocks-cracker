"""Collect saved pixel-only ablations for reporting. Does not run or select solvers."""
import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    report={}
    for sub in ['improve','improve-refit','improve-combo','improve-extra']:
        for p in sorted((a.runs/sub).glob('*/*/result.json')):
            trial,image=p.parent.parent.name,p.parent.name
            r=json.loads(p.read_text())
            report.setdefault(trial,{})[image]={'tiles':len(r['blocks']),'seconds':r['seconds']}
    a.out.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
