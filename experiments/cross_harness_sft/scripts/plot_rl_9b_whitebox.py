#!/usr/bin/env python3
"""One figure per metric; no synthetic step zero or stitched branches."""
import argparse,json
from pathlib import Path


def plot(run):
    path=run/'metrics.jsonl'
    if not path.exists():return
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r.get('record_type')=='metrics' and r['phase']=='train']
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'paper_figures';out.mkdir(exist_ok=True)
    series={
        'reward':('Verified trajectory reward',lambda r:r['optimizer']['mean_reward']),
        'entropy':('Actor token entropy',lambda r:r['optimizer'].get('actor/entropy_loss', r['optimizer'].get('actor/entropy'))),
        'zero_variance':('Zero-variance outcome groups',lambda r:r['zero_variance_group_rate']),
        'response_length':('Assistant response tokens',lambda r:r['optimizer']['mean_response_tokens']),
    }
    for name,(label,get) in series.items():
        values=[(r['step'],get(r)) for r in rows if get(r) is not None]
        if not values:continue
        fig,ax=plt.subplots(figsize=(6.4,4.))
        ax.plot(*zip(*values),color='#236a94',linewidth=1.7,marker='.' if len(values)<15 else None)
        ax.set(xlabel='Online sampling iteration',ylabel=label,title='Online RL | Controlled harness configurations')
        ax.grid(alpha=.2);fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=240)
        plt.close(fig)
    (out/'provenance.json').write_text(json.dumps({'source':str(path),'last_iteration':rows[-1]['step'],
        'recorded_iterations':[r['step'] for r in rows],'smoothing':False,'synthetic_step_zero':False},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();plot(args.run)
