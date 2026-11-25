"""Portable MAGI ARC stages. See docs/MAGI_ARC.md for calibration order."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--base-eval-root',type=Path,required=True)
    p.add_argument('--arc-calibration',type=Path,required=True)
    p.add_argument('--teacher-root',type=Path)
    p.add_argument('--asset-dir',type=Path,help='Reusable original-weight SVD cache; independent of ARC statistics')
    p.add_argument('--method',choices=['none','teal','wina'],required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['stats','capture','search','extract','generate'],required=True)
    opt,rest=p.parse_known_args(argv)
    base=opt.base_eval_root.resolve()
    sys.path.insert(0,str(base))
    assets=(opt.asset_dir or opt.root/'svd').resolve()
    os.environ.update(ARC_BASE_EVAL=str(base),MAGI_ARC_CALIBRATION=str(opt.arc_calibration.resolve()),
        MAGI_ARC_METHOD=opt.method,MAGI_ARC_ASSETS=str(assets))
    if '--repo' in rest:
        count=int(os.environ.get('GIT_CONFIG_COUNT','0'))
        repo=str(Path(rest[rest.index('--repo')+1]).resolve())
        os.environ.update({f'GIT_CONFIG_KEY_{count}':'safe.directory',f'GIT_CONFIG_VALUE_{count}':repo,'GIT_CONFIG_COUNT':str(count+1)})
    if opt.stage=='generate':
        from integrations.magi_arc.generate import main as generate
        return generate(['--method',opt.method,'--asset-dir',str(assets),*rest])
    from integrations.magi_arc.calibrate import arguments,capture,search,extract
    selected,args=arguments([opt.stage,'--method',opt.method,'--root',str(opt.root),*rest])
    if opt.stage=='stats':
        if opt.teacher_root is None: p.error('stats requires --teacher-root with completed ordinary HiF4 trajectories')
        from integrations.magi_arc.calibration import collect
        return collect(selected,args,opt.teacher_root.resolve())
    if opt.method=='none' and opt.stage in ('search','extract'):
        p.error('Dense ARC has no sparse search/allocation')
    return {'capture':capture,'search':search,'extract':extract}[opt.stage](selected,args)


if __name__=='__main__': main()
