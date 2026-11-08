"""Evaluate saved detector ablations. Calibration is evaluation-only, never runtime."""
import argparse
import json
from pathlib import Path
from evaluate import evaluation

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--hill-fixture',type=Path,required=True)
    ap.add_argument('--birch-fixture',type=Path,required=True)
    ap.add_argument('--river-reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    refs={
      'hill':(json.loads(args.hill_fixture.read_text())['blocks'],(3538,90,-454),0),
      'birch':(json.loads(args.birch_fixture.read_text())['blocks'],(-4133,63,-2049),3),
      'river':(json.loads(args.river_reference.read_text())['blocks'],(-525,65,143),2)
    }
    report={}
    for p in sorted(args.runs.glob('*/*/result.json')):
        variant,name=p.parent.parent.name,p.parent.name
        result=json.loads(p.read_text())
        metrics=evaluation(result,*refs[name]) if name in refs else {'tiles':len(result['blocks'])}
        metrics['seconds']=result.get('seconds')
        if name=='river':metrics['reference_kind']='prior unique bounded hash solution, not independent screenshot ground truth'
        report.setdefault(variant,{})[name]=metrics
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
