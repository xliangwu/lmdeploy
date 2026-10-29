# 作用：按完整响应 p95<450ms、无失败/丢弃筛选压测结果，区分预热、扫描和确认运行。
# 示例：python tools/summarize_slo_runs.py /path/results --output /path/summary.json
import argparse
import json
from pathlib import Path


def value(metric,field,default=None):
    # 支持当前 k6 的平铺摘要和嵌套 values 摘要，缺失延迟保持 None。
    data=metric.get('values',metric)
    return data.get(field,default)


def main():
    # 仅解析真正包含完整请求耗时和成功计数的文件，不把旧 HTTP 指标混入本轮 SLO。
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=[]
    for path in sorted(Path(args.directory).glob('*summary.json')):
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        metrics=data.get('metrics',{})
        if 'request_complete_ms' not in metrics or 'requests_successful' not in metrics:
            continue
        p95=value(metrics['request_complete_ms'],'p(95)')
        failed=value(metrics.get('requests_failed',{}),'count',0)
        dropped=value(metrics.get('dropped_iterations',{}),'count',0)
        checks_failed=value(metrics.get('checks',{}),'fails',0)
        done=value(metrics.get('stream_done',{}),'value',value(metrics.get('stream_done',{}),'rate'))
        row=dict(run=path.stem.removesuffix('-summary'),p95_ms=p95,
                 p99_ms=value(metrics['request_complete_ms'],'p(99)'),
                 mean_ms=value(metrics['request_complete_ms'],'avg'),
                 success_qps=value(metrics['requests_successful'],'rate'),
                 requests=value(metrics['requests_successful'],'count'),failed=failed,dropped=dropped,
                 checks_failed=checks_failed,done_rate=done)
        row['slo_pass']=p95 is not None and p95<450 and failed==0 and dropped==0 and checks_failed==0 and done==1
        rows.append(row)
    result=dict(slo='complete SSE p95 <450 ms; zero request/check failures and zero dropped arrivals',runs=rows)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in rows:
        print(json.dumps(row,ensure_ascii=False))


if __name__=='__main__':
    main()
