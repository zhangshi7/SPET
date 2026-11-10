"""Serial, auditable actual-execution oracle around unmodified BenchBase queries."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
PARAMETERS = (
    ('work_mem', 1024., 32768., 'kB', True),
    ('random_page_cost', 1., 8., '', False),
    ('cpu_tuple_cost', .001, .1, '', False),
    ('cpu_operator_cost', .00025, .025, '', False),
    ('effective_cache_size', 32., 2048., 'MB', True),
    ('hash_mem_multiplier', 1., 4., '', False),
)
DEFAULT_SETTINGS = {'work_mem':'4096kB', 'random_page_cost':'4', 'cpu_tuple_cost':'0.01',
                    'cpu_operator_cost':'0.0025', 'effective_cache_size':'2048MB',
                    'hash_mem_multiplier':'2'}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def canonical_digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def settings_from_x(x):
    x=np.asarray(x,float)
    if x.shape != (len(PARAMETERS),) or not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):
        raise ValueError('Decision must lie inside the registered normalized six-dimensional box')
    result={}
    for z,(key,lo,hi,unit,integer) in zip(x,PARAMETERS):
        value=lo*(hi/lo)**float(z)
        result[key]=(str(int(round(value))) if integer else f'{value:.12f}'.rstrip('0').rstrip('.'))+unit
    return result

def x_from_settings(settings):
    output=[]
    for key,lo,hi,unit,integer in PARAMETERS:
        value=str(settings[key])
        if unit and value.endswith(unit): value=value[:-len(unit)]
        output.append(np.log(float(value)/lo)/np.log(hi/lo))
    return np.asarray(output)

def setting_number(value):
    match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(kB|MB|GB|B|ms|s)?',str(value))
    if not match: raise ValueError(f'Cannot normalize setting {value!r}')
    number=float(match[1]);unit=match[2]
    return number*{'B':1,'kB':1024,'MB':1024**2,'GB':1024**3,'ms':1,'s':1000,None:1}[unit]

def validate_settings(requested,actual,timeout_ms):
    for key,value in requested.items():
        requested_number,actual_number=setting_number(value),setting_number(actual[key])
        # PostgreSQL SHOW prints real GUCs with six significant digits. Storage
        # quantities remain exact after unit conversion; real settings compare
        # the server's display precision, not nonexistent printed digits.
        match=(requested_number==actual_number) if key in ('work_mem','effective_cache_size') else (
            format(requested_number,'.6g')==format(actual_number,'.6g'))
        if not match:
            raise AssertionError(f'Session setting did not apply: {key}')
    for key,value in {'max_parallel_workers_per_gather':'0','jit':'off','plan_cache_mode':'force_custom_plan'}.items():
        if str(actual[key])!=value:raise AssertionError(f'Fixed session setting differs: {key}')
    if setting_number(actual['seq_page_cost'])!=1 or setting_number(actual['statement_timeout'])!=timeout_ms:
        raise AssertionError('Cost scale or statement timeout differs')

def locate_runtime():
    candidates=list((ROOT/'runtime/java25').glob('*/bin/java.exe'))
    if len(candidates)!=1: raise RuntimeError('Expected one isolated Java25 runtime')
    jars=list((ROOT/'runtime/benchbase').rglob('benchbase.jar'))
    if len(jars)!=1: raise RuntimeError('Expected one pinned BenchBase distribution')
    return candidates[0],jars[0]

def compile_harness():
    java,jar=locate_runtime()
    source=ROOT/'src/cases/java/com/oltpbenchmark/api/R2QueryHarness.java'
    classes=ROOT/'runtime/harness_classes'; classes.mkdir(parents=True,exist_ok=True)
    metadata=ROOT/'runtime/harness_build.json'
    expected={'source_sha256':sha(source),'jar_sha256':sha(jar),'java':str(java),'jar':str(jar)}
    target=classes/'com/oltpbenchmark/api/R2QueryHarness.class'
    if metadata.exists() and target.exists():
        old=read_json(metadata)
        if all(old.get(k)==v for k,v in expected.items()) and old.get('class_sha256')==sha(target):
            return java,jar,classes
    cmd=[str(java.with_name('javac.exe')),'-encoding','UTF-8','-cp',str(jar)+';'+str(jar.parent/'lib/*'),
         '-d',str(classes),str(source)]
    done=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',
                        creationflags=subprocess.CREATE_NO_WINDOW,timeout=120)
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'logs/harness_compile.log').write_text(done.stdout+'\n'+done.stderr,encoding='utf-8')
    if done.returncode: raise RuntimeError('Harness compilation failed; see logs/harness_compile.log')
    write_json(metadata,{**expected,'class_sha256':sha(target),'compiled_utc':now(),'command':cmd})
    return java,jar,classes

class DatabaseOracle:
    """Each request is serialized; the full underlying physical ledger is preserved.

    The JVM stays alive but prepared statements are new and prepareThreshold=0.
    Each query gets a savepoint and rollback, including Q15's temporary lifecycle
    of its original revenue0 view. No business-table data is changed.
    """
    def __init__(self, directory, *, scale=.1, timeout_ms=30000):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.scale=float(scale);self.timeout_ms=int(timeout_ms);self.calls=0;self.physical_queries=0
        self.attempt_id=uuid.uuid4().hex[:12];self.failed=False
        java,jar,classes=compile_harness()
        self.stderr=(self.directory/'harness.stderr.log').open('a',encoding='utf-8')
        command=[str(java),'-Xms64m','-Xmx512m','-cp',str(classes)+';'+str(jar)+';'+str(jar.parent/'lib/*'),
                 'com.oltpbenchmark.api.R2QueryHarness',str(ROOT/'runtime/db_connection.json')]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
        self.responses=queue.Queue()
        self.memory_peak={'db_private_plus_shared_bytes':0,'jvm_rss_bytes':0,'minimum_available_bytes':2**63-1}
        self.connection_settings=read_json(ROOT/'runtime/db_connection.json')
        def reader():
            try:
                for line in self.process.stdout:
                    if line.startswith('R2_ORACLE_RESPONSE '):
                        self.responses.put(json.loads(line[len('R2_ORACLE_RESPONSE '):]))
                    else:
                        with (self.directory/'harness.stdout.other.log').open('a',encoding='utf-8') as f:f.write(line)
            finally:self.responses.put(None)
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        p=psutil.Process(self.process.pid)
        write_json(self.directory/'oracle_lease.json',{'pid':p.pid,'create_time':p.create_time(),
            'owner_pid':os.getpid(),'started_utc':now(),'database_port':55432})
        try:self.health=self.request({'op':'health'},timeout=60)
        except BaseException:
            self.close();raise
        write_json(self.directory/'health.json',self.health)

    def request(self, request, *, timeout=600):
        if self.failed:raise RuntimeError('Oracle attempt has failed; reconstruct a new attempt')
        if self.process.poll() is not None:raise RuntimeError('Database harness exited')
        self.process.stdin.write(json.dumps(request,allow_nan=False)+'\n');self.process.stdin.flush()
        deadline=time.monotonic()+timeout
        while True:
            try:
                response=self.responses.get(timeout=min(.25,max(.01,deadline-time.monotonic())))
                break
            except queue.Empty:
                if time.monotonic()>=deadline:raise TimeoutError('Database oracle response deadline exceeded')
                self.check_resources()
        if response is None: raise RuntimeError('Database harness terminated; inspect its stderr log')
        if response.get('status')!='OK': raise RuntimeError(f'Oracle failure: {response}')
        if 'id' in request and response.get('id')!=request['id']:
            self.failed=True
            raise RuntimeError('Oracle response/request identity mismatch')
        return response

    def check_resources(self):
        available=psutil.virtual_memory().available
        self.memory_peak['minimum_available_bytes']=min(self.memory_peak['minimum_available_bytes'],available)
        memory=0
        data_dir=Path(self.connection_settings['data_dir'])
        postmaster_pid=int((data_dir/'postmaster.pid').read_text().splitlines()[0])
        postmaster=psutil.Process(postmaster_pid)
        if data_dir.as_posix().lower() not in ' '.join(postmaster.cmdline()).replace('\\','/').lower():
            raise RuntimeError('Postmaster identity mismatch')
        for process in [postmaster]+postmaster.children(recursive=True):
            try:
                info=process.memory_info();memory+=getattr(info,'private',info.rss)
            except psutil.NoSuchProcess:pass
        memory+=512*1024**2 # Shared buffer reservation charged once, separately from private bytes.
        jvm=psutil.Process(self.process.pid).memory_info().rss
        self.memory_peak['db_private_plus_shared_bytes']=max(self.memory_peak['db_private_plus_shared_bytes'],memory)
        self.memory_peak['jvm_rss_bytes']=max(self.memory_peak['jvm_rss_bytes'],jvm)
        if available<3*1024**3 or memory>4*1024**3:
            if hasattr(self,'health'):
                env=os.environ.copy();env.update(PGPASSWORD=self.connection_settings['password'],
                    PGHOST='127.0.0.1',PGPORT='55432',PGUSER=self.connection_settings['user'])
                subprocess.run([str(Path(self.connection_settings['pg_bin'])/'psql.exe'),'-d',
                    self.connection_settings['database'],'-At','-c',
                    f"SELECT pg_cancel_backend({int(self.health['backend_pid'])})"],
                    env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            raise RuntimeError('Resource guard: available memory <3 GiB or DB private+shared estimate >4 GiB')

    def run_queries(self,settings,queries,*,role,context,include_sql=False,checksum=False):
        if psutil.virtual_memory().available < 3*1024**3:
            raise RuntimeError('Resource guard: less than 3 GiB memory available')
        self.calls+=1
        request={'id':f'{self.attempt_id}-{role}-{self.calls:07d}','scale':self.scale,'settings':settings,
                 'queries':queries,'timeout_ms':self.timeout_ms,'checksum':bool(checksum),'include_sql':include_sql}
        started=now();tic=time.perf_counter()
        attempt={'status':'STARTED','request':request,'role':role,'context':context,'started_utc':started}
        attempt_path=self.directory/'attempts'/f'{request["id"]}.json'
        write_json(attempt_path,attempt)
        try:
            response=self.request(request,timeout=max(120,len(queries)*(self.timeout_ms/1000+3)+30))
        except BaseException as exc:
            self.failed=True
            write_json(attempt_path,{**attempt,'status':'FAILED_UNKNOWN_EXECUTION_COUNT',
                'error':str(exc),'ended_utc':now(),'physical_query_count':None})
            raise
        write_json(attempt_path,{**attempt,'status':'RESPONSE_RECEIVED','response':response,'received_utc':now()})
        try:validate_settings(settings,response['settings_actual'],self.timeout_ms)
        except BaseException as exc:
            self.failed=True
            write_json(attempt_path,{**attempt,'status':'FAILED_VALIDATION','response':response,
                'error':str(exc),'ended_utc':now(),'physical_query_count':len(response['queries']),
                'physical_sql_executions':sum(q['sql_executions'] for q in response['queries'])})
            raise
        if len(response['queries'])!=len(queries) or any(q['status']=='SQL_ERROR' for q in response['queries']):
            write_json(self.directory/f"failure_{self.calls:07d}.json",{'request':request,'response':response})
            raise RuntimeError('SQL failure: retained raw request and result')
        self.physical_queries+=len(queries)
        record={'request':request,'response':response,'role':role,'context':context,
                'started_utc':started,'completed_utc':now(),'wall_seconds':time.perf_counter()-tic,
                'available_memory_bytes':psutil.virtual_memory().available,
                'jvm_rss_bytes':psutil.Process(self.process.pid).memory_info().rss}
        record['memory_peaks_so_far']=dict(self.memory_peak)
        record['physical_sql_executions']=sum(q['sql_executions'] for q in response['queries'])
        write_json(attempt_path,{**attempt,'status':'COMPLETE','ended_utc':now(),
            'physical_query_count':len(response['queries']),'physical_sql_executions':record['physical_sql_executions']})
        with (self.directory/'physical_ledger.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
        return record

    def evaluate(self,x,query_specs,weights,*,context,role='feedback',warmup_specs=None,repeats=1):
        settings=settings_from_x(x);weights=np.asarray(weights,float)
        if len(query_specs)!=32 or weights.shape!=(32,) or not np.array_equal(weights,np.ones(32)):
            raise ValueError('Registered DB objective is the equal-weight actual mean of 32 executed requests')
        if set(q['query'] for q in query_specs)!=set(range(1,23)):
            raise ValueError('Every registered workload must cover the original 22 templates')
        records=[]
        if warmup_specs:
            records.append(self.run_queries(settings,warmup_specs,role='warmup',context=context))
        costs=[]
        for repeat in range(repeats):
            row=self.run_queries(settings,query_specs,role=role,context={**context,'repeat':repeat})
            latencies=np.asarray([q['elapsed_ms'] if q['status']=='OK' else self.timeout_ms
                                  for q in row['response']['queries']],float)
            costs.append(float(np.dot(weights,latencies)/weights.sum()));records.append(row)
        return float(np.mean(costs)),{'x':np.asarray(x).tolist(),'settings':settings,
             'query_specs':query_specs,'weights':weights.tolist(),'repeats':repeats,
             'objective_ms':float(np.mean(costs)),'measurement_costs_ms':costs,
             'logical_feedback_FE':1 if role=='feedback' else 0,'physical_queries':sum(
                 len(r['response']['queries']) for r in records),
             'physical_wall_seconds':sum(r['wall_seconds'] for r in records),
             'request_ids':[r['request']['id'] for r in records]}

    def close(self):
        if getattr(self,'process',None) and self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        if getattr(self,'stderr',None):self.stderr.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--health',action='store_true')
    args=parser.parse_args()
    with DatabaseOracle(ROOT/'results/oracle_health') as oracle:
        print(json.dumps(oracle.health))
