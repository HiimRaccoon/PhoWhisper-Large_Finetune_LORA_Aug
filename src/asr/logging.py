"""Append-only attempt logs; input transcripts/waveforms are never logged."""
from datetime import datetime,timezone
import json
from pathlib import Path


class RunLogger:
    def __init__(self,path,run_id,attempt_id):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.stream = path.open('a',encoding='utf-8',newline='\n')
        self.run_id,self.attempt_id = run_id,attempt_id
        self.path = path
        self.summary = {'views':0,'clean':0,'augmentation':0,'changed':0,'fallback_views':0,
                        'fallbacks':{},'optimizer_updates':0}

    def log(self,event,**fields):
        if event == 'optimizer_update':
            required = {'epoch','position','micro_step','global_update','loss','scaled_loss','window_size','lr',
                        'gradient_norm','dtype','scaler','gpu_memory','elapsed_seconds','throughput'}
            if not required.issubset(fields):
                raise ValueError('incomplete optimizer update log')
        record = {**fields,'event':event,'timestamp':datetime.now(timezone.utc).isoformat(),
                  'run_id':self.run_id,'attempt_id':self.attempt_id}
        self.stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False,default=str)+'\n')
        self.stream.flush()
        if event == 'optimizer_update':
            self.summary['optimizer_updates'] += 1
        if event == 'augmentation':
            branch = fields.get('branch')
            self.summary['views'] += 1
            if branch in {'clean','augmentation'}:
                self.summary[branch] += 1
            self.summary['changed'] += int(bool(fields.get('changed')))
            self.summary['fallback_views'] += int(bool(fields.get('fallbacks')))
            for reason in fields.get('fallbacks',[]):
                self.summary['fallbacks'][reason] = self.summary['fallbacks'].get(reason,0)+1

    def close(self):
        from .paths import write_json_atomic
        self.stream.close()
        write_json_atomic(self.path.with_suffix('.summary.json'),self.summary)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
