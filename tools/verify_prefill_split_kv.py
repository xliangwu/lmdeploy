# 作用：以固定样本对比替换前后的真实服务输出，不进行吞吐基线或压力测试。
# 示例：python tools/verify_prefill_split_kv.py --help
# 捕获示例（Linux）：
# python tools/verify_prefill_split_kv.py \
#   --mode capture \
#   --dataset /path/cases.txt \
#   --reference /path/reference.json \
#   --url http://127.0.0.1:23333
import argparse
import concurrent.futures
import hashlib
import json
import random
import urllib.error
import urllib.request
from pathlib import Path


def make_body(case):
    # 保留原压测生成参数；非流式响应只用于稳定读取完整输出文本。
    return dict(model='qwen', messages=case['input']['messages'], max_tokens=3,
                temperature=0, top_p=1.0, repetition_penalty=1.0, top_k=1,
                stream=False, chat_template_kwargs=case.get('chat_template_kwargs') or {'enable_thinking': False})


def request_one(index, case, url):
    # 单条请求返回索引、输入摘要和输出，不保存或打印原始输入正文。
    payload = json.dumps(make_body(case), sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(payload).hexdigest()
    request = urllib.request.Request(url.rstrip('/')+'/v1/chat/completions',
                                     data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        return dict(index=index, input_sha256=digest, content=data['choices'][0]['message'].get('content', ''),
                    finish_reason=data['choices'][0].get('finish_reason'), usage=data.get('usage'))
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as error:
        # 错误计入验收失败，不能用空输出冒充有效结果。
        return dict(index=index, input_sha256=digest, error=type(error).__name__)


def main():
    # capture 使用固定种子；compare 复用参考索引并检查输入摘要，避免跨数据集比较。
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['capture', 'compare'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:23333')
    parser.add_argument('--count', type=int, default=128)
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    cases = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    reference = Path(args.reference)
    old = None if args.mode == 'capture' else json.loads(reference.read_text())
    indices = (random.Random(20261027).sample(range(len(cases)), min(args.count, len(cases)))
               if old is None else [r['index'] for r in old['results']])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(request_one, index, cases[index], args.url) for index in indices]
        results = [f.result() for f in futures]
    errors = [r['index'] for r in results if 'error' in r]
    summary = dict(mode=args.mode, requests=len(results), errors=errors)
    if old is None:
        # 失败时不覆盖参考，防止将异常结果作为后续正确答案。
        if not errors:
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_text(json.dumps(dict(results=results), ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        mismatches = []
        for before, after in zip(old['results'], results):
            if before['input_sha256'] != after['input_sha256']:
                raise RuntimeError('测试输入发生变化，拒绝跨数据集比较')
            if before.get('content') != after.get('content') or before.get('finish_reason') != after.get('finish_reason'):
                mismatches.append(after['index'])
        summary['output_mismatches'] = mismatches
        reference.with_suffix('.comparison.json').write_text(
            json.dumps(dict(summary=summary, results=results), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if errors or summary.get('output_mismatches') else 0


if __name__ == '__main__':
    raise SystemExit(main())
