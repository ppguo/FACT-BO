#!/usr/bin/env python3
"""Verify the committed public optimizer against one archived corrected SRAM run.

Only the initial observations are reused. Every adaptive point is simulated
again. Instrumentation observes scores/points without replacing either. A
trajectory difference is recorded, never repaired by substituting reference data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from ei_oracle import compare, expected_improvement, ranking_check
import algorithms._GITBO as optimizer
import algorithms.tabpfn_wrapper as wrapper_module
from fas_wca import LineProcessEvaluator, SramReadDelayProblem


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array(x):
    return x.detach().cpu().numpy()


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')


def write(path, row):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--initial-dir', type=Path, required=True)
    p.add_argument('--openyield-root', type=Path, required=True)
    p.add_argument('--meta', type=Path, required=True)
    p.add_argument('--worker-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--iterations', type=int, default=270)
    p.add_argument('--resume-prefix', type=Path,
                   help='Reuse a validated interrupted prefix only while regenerated points match exactly')
    args = p.parse_args()
    if not 1 <= args.iterations <= 270:
        p.error('iterations must be in [1,270]')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference = args.reference_run.resolve()
    source = json.loads((reference/'provenance.json').read_text())
    old = json.loads((reference/'result.json').read_text())
    rows = [json.loads(line) for line in (reference/'trace.jsonl').read_text().splitlines()]
    assert (reference/'RUN.done').exists() and len(rows)==300
    assert sha(reference/'trace.jsonl')==old['trace_sha256']
    assert source['circuit']=='sram' and source['direction']==1
    assert source['threshold']==290.6 and source['ball_k']==16.
    assert source['rank']==5 and source['scale']==1. and source['n_pending']==5000
    gpu = torch.cuda.get_device_name(torch.device(args.device))
    if torch.__version__ != source['torch']:
        raise RuntimeError(f"Reference Torch is {source['torch']}, current Torch is {torch.__version__}")
    seed = source['seed']
    initial = args.initial_dir/f'_trial_{seed}.pt'
    init = torch.load(initial, map_location='cpu', weights_only=False)
    assert tuple(init.shape)==(30,144)
    assert np.array_equal(array(init).astype(float), np.array([r['latent_u'] for r in rows[:30]]))
    assert all(not r['simulator']['error'] for r in rows[:30])
    os.chdir(ROOT)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    sources = [ROOT/'algorithms/_GITBO.py', ROOT/'algorithms/tabpfn_wrapper.py',
               ROOT/'tabpfn/model/bar_distribution.py',ROOT/'fas_wca/acquisition.py',
               ROOT/'fas_wca/sram.py',ROOT/'fas_wca/sigma_ball.py',
               ROOT/'tabpfn/model/tabpfn-v2-regressor.ckpt', Path(__file__),
               args.meta, args.openyield_root/'size_optimization/sample_yield_4x2.py', initial]
    hashes = {str(f.resolve()):sha(f) for f in sources}
    prefix = []
    prefix_provenance = None
    if args.resume_prefix is not None:
        previous = args.resume_prefix.resolve()
        prefix = [json.loads(line) for line in (previous/'trace.jsonl').read_text().splitlines()]
        prefix_provenance = json.loads((previous/'provenance.json').read_text())
        ledger = [json.loads(line) for line in (previous/'calls.jsonl').read_text().splitlines()]
        starts = [(r['kind'],r['call']) for r in ledger if r['event']=='started']
        ends = [(r['kind'],r['call']) for r in ledger if r['event']=='finished']
        assert starts==ends and len(starts)==len(set(starts)), 'unresolved prefix simulator calls'
        assert 30 <= len(prefix) <= 30+args.iterations
        assert prefix_provenance['seed']==seed and prefix_provenance['reference_trace_sha256']==old['trace_sha256']
        assert prefix_provenance['gpu']==gpu, 'resume on the same GPU model as the interrupted run'
        assert prefix[:30]==[dict(call=i+1,latent_u=r['latent_u'],objective_y=r['objective_y'],reused=True) for i,r in enumerate(rows[:30])]
        for f in sources:
            if f.resolve()==Path(__file__).resolve():
                continue  # The validation driver can gain recovery support; numerical sources cannot drift.
            suffix = str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f)
            matches = [h for name,h in prefix_provenance['hashes'].items() if name.endswith(suffix)]
            assert len(matches)==1 and matches[0]==sha(f), f'prefix source/input drift: {f}'
        for r in prefix[30:]:
            assert not r['reused'] and ('search',r['call']) in ends
            assert np.isfinite(r['objective_y']) and len(r['latent_u'])==144
    write(out/'provenance.json', dict(commit=commit, reference=str(reference),
        reference_trace_sha256=old['trace_sha256'], seed=seed, dimension=144,
        threshold_ps=290.6, radius=16., rank=5, scale=1., n_pending=5000,
        requested_evaluations=30+args.iterations, initial_reused=30, torch=torch.__version__,
        device=args.device, gpu=gpu, reference_gpu=source['gpu'],
        hardware_matches_reference=gpu==source['gpu'], hashes=hashes,
        resume_prefix=None if args.resume_prefix is None else str(args.resume_prefix.resolve()),
        resume_trace_sha256=None if args.resume_prefix is None else sha(args.resume_prefix/'trace.jsonl')))
    command = [str(args.worker_python), str(Path(__file__).with_name('openyield_worker.py')),
        '--openyield-root',str(args.openyield_root), '--meta',str(args.meta),
        '--workdir',str(out/'simulator')]
    evaluator = LineProcessEvaluator(command, out/'worker')
    physical = SramReadDelayProblem(evaluator)
    observed_rows, checks, capture = [], [], {}
    prefix_active = bool(prefix)
    prefix_reused = 0
    original_compute = optimizer.compute_acquisition_values
    original_wrapper = wrapper_module.VanillaDirectTabPFNRegressor
    original_load = torch.load
    checkpoint_loads = []

    def checked_load(path, *pos, **kw):
        if isinstance(path, (str, Path)) and str(path).endswith('.ckpt'):
            actual = Path(path).resolve()
            assert actual == (ROOT/'tabpfn/model/tabpfn-v2-regressor.ckpt').resolve()
            checkpoint_loads.append(str(actual))
        return original_load(path, *pos, **kw)

    class ObservedRegressor(original_wrapper):
        def predict_ei(self, logits, target):
            result = super().predict_ei(logits, target)
            capture.update(logits=array(logits),borders=array(self.bardist_.borders),
                mean=float(self.y_mean.item()),std=float(self.y_std.item()),target=float(target))
            return result

    def observed_compute(*pos, **kw):
        values, cons, gradient = original_compute(*pos, **kw)
        candidates = pos[9]
        std, mean, target = capture['std'],capture['mean'],capture['target']
        z = float(np.float32((np.float32(target)-np.float32(mean))/np.float32(std)))
        ref, error = expected_improvement(capture['logits'],capture['borders'],z)
        check = {**compare(array(values)/std,ref), **ranking_check(array(values)/std,ref)}
        i = len(checks); index = int(values[:,0].argmax())
        chosen = array(candidates[index,0]).astype(float)
        check.update(iteration=i, target_y=target, selected_index=index,
            selected_score=float(values[index,0]),
            candidate_pool_sha256=hashlib.sha256(array(candidates).tobytes()).hexdigest(),
            gradient_sha256=hashlib.sha256(array(gradient).tobytes()).hexdigest(),
            reference_candidate_exact=bool(np.array_equal(chosen,rows[30+i]['latent_u'])),
            quadrature_error_bound=float(error.max()))
        assert check['pass'] and check['argmax_pass']
        assert target==max(290.6,max(r['objective_y'] for r in observed_rows))
        capture['candidate'] = chosen
        np.savez_compressed(out/f'score_{i:03d}.npz', values=array(values),reference=ref,
            std=std, target=z, borders=capture['borders'], selected_logits=capture['logits'][index])
        append(out/'score_checks.jsonl',check);checks.append(check)
        return values,cons,gradient

    class Problem:
        dim=144
        def evaluate(self, x):
            nonlocal prefix_active, prefix_reused
            if not observed_rows:
                assert np.array_equal(array(x).astype(float),array(init).astype(float))
                for i,row in enumerate(rows[:30]):
                    item=dict(call=i+1,latent_u=row['latent_u'],objective_y=row['objective_y'],reused=True)
                    observed_rows.append(item);append(out/'trace.jsonl',item)
                return None,x.new_tensor([[r['objective_y']] for r in rows[:30]])
            assert len(x)==1 and np.array_equal(array(x[0]).astype(float),capture['candidate'])
            call=len(observed_rows)+1
            if prefix_active and call<=len(prefix):
                previous=prefix[call-1]
                if np.array_equal(array(x[0]).astype(float),previous['latent_u']):
                    item={**previous,'reused':True,'reuse_kind':'exact_adaptive_prefix'}
                    observed_rows.append(item);append(out/'trace.jsonl',item)
                    prefix_reused+=1
                    print(json.dumps(dict(call=call,reused='exact_adaptive_prefix')),flush=True)
                    return None,x.new_tensor([[previous['objective_y']]])
                prefix_active=False
                write(out/'PREFIX_DIVERGENCE.json',dict(call=call,reason='stop reuse; evaluate regenerated point and all following points'))
            append(out/'calls.jsonl',dict(call=call,event='started',kind='search'))
            start=time.monotonic()
            _,y=physical.evaluate(x)
            value=float(y.item())
            append(out/'calls.jsonl',dict(call=call,event='finished',kind='search',value=value,elapsed=time.monotonic()-start))
            item=dict(call=call,latent_u=array(x[0]).astype(float).tolist(),objective_y=value,reused=False)
            observed_rows.append(item);append(out/'trace.jsonl',item)
            print(json.dumps(dict(call=call,value=value,best=max(r['objective_y'] for r in observed_rows),
                reference_point_exact=checks[-1]['reference_candidate_exact'])),flush=True)
            return None,y

    try:
        random.seed(seed)
        torch.load=checked_load
        wrapper_module.VanillaDirectTabPFNRegressor=ObservedRegressor
        optimizer.compute_acquisition_values=observed_compute
        points,history=optimizer.GITBO(Problem(),seed,Trail_N=seed,N_iterations=args.iterations,
            Acquisition='ST-EFMI',threshold_y=290.6,INITIAL_DIR=str(args.initial_dir),
            SAVE_DIR=str(out/'results'),N_PENDING=5000,N_CANDIDATES=1,DEVICE=args.device,
            GPU_DEVICE=args.device,GI_SUBSPACE=True,rank_r=5,scale=1.)
        assert len(observed_rows)==30+args.iterations and len(checks)==args.iterations
        assert len(checkpoint_loads)==args.iterations
        assert np.array_equal(array(points).astype(float),[r['latent_u'] for r in observed_rows])
        first=next((r['call'] for r in observed_rows if r['objective_y']>290.6),None)
        worst=max(range(len(observed_rows)),key=lambda i:observed_rows[i]['objective_y'])
        rechecks=[]
        for i in sorted({worst}|({first-1} if first is not None else set())):
            append(out/'calls.jsonl',dict(call=i+1,event='started',kind='recheck'))
            _,y=physical.evaluate(points[i:i+1])
            value=float(y.item()); error=abs(value-observed_rows[i]['objective_y'])
            append(out/'calls.jsonl',dict(call=i+1,event='finished',kind='recheck',value=value))
            item=dict(call=i+1,value=value,error=error,pass_=error<=1e-6+1e-4*abs(value))
            rechecks.append(item);assert item['pass_']
        differences=[i+1 for i,r in enumerate(observed_rows) if not np.array_equal(r['latent_u'],rows[i]['latent_u'])]
        max_y_error=max(abs(r['objective_y']-rows[i]['objective_y']) for i,r in enumerate(observed_rows))
        best=observed_rows[worst]['objective_y']
        saved=list((out/'results').rglob('*.pt'));assert len(saved)==1
        payload=torch.load(saved[0],map_location='cpu',weights_only=False)
        assert np.array_equal(array(payload['trained_Y']).ravel(),[r['objective_y'] for r in observed_rows])
        assert all(not r['fallback'] for r in payload['acquisition_telemetry'])
        assert hashes=={str(f.resolve()):sha(f) for f in sources}
        result=dict(commit=commit,complete=True,full_budget=args.iterations==270,seed=seed,
            evaluations=len(observed_rows),initial_reused=30,adaptive_prefix_reused=prefix_reused,
            new_search_calls=args.iterations-prefix_reused,
            effective_search_evaluations=30+prefix_reused+(args.iterations-prefix_reused),
            recheck_calls=len(rechecks),simulator_errors=0,fallback_count=0,
            first_failure_call=first,reference_first_failure_call=old['first_failure_call'],
            final_worst_ps=best,reference_final_worst_ps=old['final_worst_metric'],
            all_points_exact=not differences,first_point_difference_call=next(iter(differences),None),
            max_paired_y_abs_error_ps=max_y_error,rechecks=rechecks,
            max_standardized_score_error=max(c['max_abs_error'] for c in checks),
            source_hashes_unchanged=True,trace_sha256=sha(out/'trace.jsonl'))
        result['gpu']=gpu
        result['reference_gpu']=source['gpu']
        result['hardware_matches_reference']=gpu==source['gpu']
        result['checkpoint_paths']=sorted(set(checkpoint_loads))
        result['matches_reference']=not differences and max_y_error==0 and first==old['first_failure_call']
        write(out/'result.json',result)
        (out/'RUN.done').write_text('Read result.json for reference agreement and full_budget.\n')
        print(json.dumps(result),flush=True)
    except BaseException:
        (out/'RUN.failed').write_text(traceback.format_exc())
        raise
    finally:
        evaluator.close()
        optimizer.compute_acquisition_values=original_compute
        wrapper_module.VanillaDirectTabPFNRegressor=original_wrapper
        torch.load=original_load


if __name__=='__main__':
    main()
