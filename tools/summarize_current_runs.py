# 作用：汇总本轮最新test.js负载结果，区分传输失败、参考标签不匹配、诊断和正式计时。
# 示例：python tools/summarize_current_runs.py /path/results --output /path/current-results.json
import argparse
import json
import re
from pathlib import Path


def value(metric, field, default=None):
    # 同时支持两类k6摘要格式；未知指标保留None。
    return metric.get('values',metric).get(field,default)


def classify(name):
    # 不将有profiler或早期缺标签异常的结果用于最终性能收益。
    if name.startswith('systems-') or name.startswith('compute-'):
        return 'diagnostic'
    if name.startswith('baseline-'):
        return 'initial_script_missing_label_error'
    if 'warmup' in name:
        return 'warmup'
    if name.startswith('budget-'):
        return 'exploratory_60s'
    return 'formal_or_confirmation'


def main():
    # 原test.js的checks包含业务参考标签，因此不能把所有checks失败当作HTTP失败。
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.directory)
    runs=[]
    for file in sorted(root.glob('*-summary.json')):
        data=json.loads(file.read_text(encoding='utf-8-sig'))
        m=data.get('metrics',{})
        if 'request_complete_ms' not in m or 'requests_successful' not in m:
            continue
        name=file.name.removesuffix('-summary.json')
        count=value(m['requests_successful'],'count')
        qps=value(m['requests_successful'],'rate')
        failed=value(m.get('requests_failed',{}),'count',0)
        done=value(m.get('stream_done',{}),'value',value(m.get('stream_done',{}),'rate'))
        p95=value(m['request_complete_ms'],'p(95)')
        log=root/(name+'.log')
        completion=None
        if log.exists():
            # 仅取终态行，不将含输入正文的日志复制进汇总。
            lines=[line for line in log.read_text(errors='replace').splitlines() if re.search(r'complete and [0-9]+ interrupted iterations',line)]
            completion=lines[-1] if lines else None
        run=dict(run=name,kind=classify(name),requests=count,qps=qps,p95_ms=p95,
                 p99_ms=value(m['request_complete_ms'],'p(99)'),mean_ms=value(m['request_complete_ms'],'avg'),
                 max_ms=value(m['request_complete_ms'],'max'),request_failures=failed,done_rate=done,
                 missing_reference=value(m.get('reference_label_missing',{}),'count',0),
                 exact_reference_rate=value(m.get('label_exact',{}),'value',value(m.get('label_exact',{}),'rate')),
                 prompt_tokens_mean=value(m.get('llm_prompt_tokens',{}),'avg'),
                 output_tokens_mean=value(m.get('llm_completion_tokens',{}),'avg'),completion_log=completion)
        run['slo_pass']=p95 is not None and p95<450 and failed==0 and done==1
        run['complete_uninterrupted']=completion is not None and bool(re.search(r'\b0 interrupted iterations',completion))
        runs.append(run)
    result=dict(latency_scope='full request function wall time, including complete SSE and original client preparation/label handling',
                slo='p95<450ms, zero request failures and all streams DONE; reference label checks are separate',runs=runs)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for run in runs:
        if run['kind']=='formal_or_confirmation':
            print(json.dumps(run,ensure_ascii=False))


if __name__=='__main__':
    main()
