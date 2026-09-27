"""Matched-seed deformation sweep with geometry interventions; writes CSV."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from dataset.generator import generate_dataset
from model.estimator import ChannelEstimator
from training.trainer import nmse_per_sample,nmse_db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--environments',type=int,default=20)
    parser.add_argument('--b',type=float,nargs='+',default=[0,.05,.1,.15,.3,.5])
    parser.add_argument('--seed',type=int,default=90401)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.environments < 2 or args.batch_size < 1 or any(not np.isfinite(b) or b<0 for b in args.b):
        parser.error('Need >=2 environments, positive batch size and finite nonnegative b.')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    config = json.loads((args.run_dir/'experiment_config.json').read_text())
    checkpoint = torch.load(args.run_dir/'best.pt',map_location='cpu',weights_only=True)
    model = ChannelEstimator(**checkpoint['model_config']).to(args.device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    simulation = config['simulation']
    pairs = simulation['n_subcarriers']-1
    rows = []
    with torch.no_grad():
        for b in args.b:
            y,g,t = generate_dataset(**simulation,n_channels=args.environments,b_min=b,b_max=b,seed=args.seed)
            # Move whole environment blocks: geometry is shared by adjacent subcarrier pairs.
            shuffled = np.roll(g.reshape(args.environments,pairs,*g.shape[1:]),1,axis=0).reshape(g.shape)
            sums = dict(normal=0.,shuffled_geometry=0.,disabled_modulation=0.,mean_reference=0.)
            for start in range(0,len(y),args.batch_size):
                sl = slice(start,start+args.batch_size)
                yy,gg,tt,gs = [torch.as_tensor(a[sl],device=args.device) for a in (y,g,t,shuffled)]
                reference = model.reconstruction.reference(yy)
                target = reference+tt
                predictions = dict(normal=model(yy,gg),shuffled_geometry=model(yy,gs),
                    disabled_modulation=model(yy,gg,modulation_strength=0),mean_reference=reference)
                for name,pred in predictions.items():
                    sums[name] += nmse_per_sample(pred,target).sum().item()
            row = dict(b_over_lambda=b,**{name+'_nmse_db':nmse_db(value/len(y)) for name,value in sums.items()})
            rows.append(row)
            print(row,flush=True)
    output = args.run_dir/f'relative_diagnostic_seed{args.seed}.csv'
    with output.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps(dict(seed=args.seed,environments=args.environments,
        b=args.b,run_dir=str(args.run_dir),rows=rows),indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
