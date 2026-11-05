# 作用：对比原始和候选 ByteLevel 增量反分词，覆盖全量真实输入、Unicode、特殊 token 与随机分块。
# 示例：python tools/verify_bytelevel_detokenize.py --help
import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from lmdeploy.model import HFChatTemplate


def load_module(path, name):
    # 独立加载两个版本，不覆盖正在运行服务的安装模块。
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summary(values):
    # 对 CPU 微测返回毫秒统计，不能用作服务吞吐数据。
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values), p95=ordered[int((len(ordered)-1)*.95)])


def main():
    # 候选必须确实启用经过限定的快路径，否则测试不能证明该路径的正确性。
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=5000)
    args = parser.parse_args()
    old_module = load_module(args.reference, 'tokenizer_reference')
    new_module = load_module(args.candidate, 'tokenizer_candidate')
    model = '/models/V166_mindspeed_mg2hf'
    original = old_module.Tokenizer(model)
    os.environ['LMDEPLOY_QWEN2_BYTELEVEL_TAIL'] = '1'
    candidate = new_module.Tokenizer(model)
    assert candidate.model._use_bytelevel_tail, '候选快路径未启用'
    template = HFChatTemplate(model_path=model)
    randomizer = random.Random(20261028)
    timing = {'reference': [], 'candidate': []}
    checks = 0

    def compare(prefix, generated, chunks, skip=True, spaces=True, timed=False):
        # 对每个到达分块比较文本，不能只比较最终拼接结果以掩盖流式边界变化。
        nonlocal checks
        old_state = old_module.DetokenizeState(len(prefix))
        new_state = new_module.DetokenizeState(len(prefix))
        position = 0
        for amount in chunks:
            position = min(position+amount, len(generated))
            ids = prefix+generated[:position]
            begin = time.perf_counter()
            expected, old_state = original.detokenize_incrementally(ids, old_state, skip, spaces)
            elapsed_old = (time.perf_counter()-begin)*1000
            begin = time.perf_counter()
            actual, new_state = candidate.detokenize_incrementally(ids, new_state, skip, spaces)
            elapsed_new = (time.perf_counter()-begin)*1000
            checks += 1
            if actual != expected or old_state.ids_offset != new_state.ids_offset:
                raise AssertionError(f'chunk mismatch: prefix={len(prefix)}, generated={position}, skip={skip}, expected={expected!r}, actual={actual!r}')
            if timed and position == min(chunks[0], len(generated)):
                timing['reference'].append(elapsed_old)
                timing['candidate'].append(elapsed_new)
            if position == len(generated):
                break

    suffix_texts = ['Z2', '你好🙂\n café', 'e\u0301 ABC', '\t 123\n', '中文结束。', '\ufffd', '👨\u200d👩\u200d👧\u200d👦', '<|im_end|>A']
    suffixes = [original.encode(text, add_bos=False) for text in suffix_texts]
    data = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    data = data[:args.count]
    for i, row in enumerate(data):
        prompt = template.messages2prompt(row['input']['messages'], **(row.get('chat_template_kwargs') or {'enable_thinking': False}))
        prefix = original.encode(prompt, add_bos=True)
        suffix = suffixes[i % len(suffixes)]
        compare(prefix, suffix, [1]*len(suffix), skip=i % 3 != 0, spaces=i % 2 == 0, timed=True)
        if i % 500 == 499:
            print(f'real_cases={i+1} chunk_checks={checks}', flush=True)

    # UTF-8 token 边界可能落在字节中间，逐个切分真实 Unicode 编码构造不完整前缀。
    unicode_ids = original.encode('A你好🙂é中👨\u200d👩\u200d👧\u200d👦\nB', add_bos=False)
    for end in range(len(unicode_ids)+1):
        for skip in (False, True):
            compare(unicode_ids[:end], unicode_ids[end:]+suffixes[0], [1]*(len(unicode_ids)+2), skip)

    specials = list(original.model.model.all_special_ids)
    vocab_size = len(original.model.model)
    for i in range(1000):
        prefix = [randomizer.randrange(vocab_size) for _ in range(randomizer.choice([0,1,4,16,64,256]))]
        if i % 5 == 0:
            prefix += specials*32
        if i % 7 == 0:
            prefix += [vocab_size+1]*20
        generated = [randomizer.randrange(vocab_size) for _ in range(12)]+specials+suffixes[i%len(suffixes)]
        chunks = [randomizer.randint(1,5) for _ in range(len(generated))]
        compare(prefix, generated, chunks, skip=i%2==0, spaces=i%3==0)

    # 环境开关关闭时必须回到原路径，保证发布可逆。
    os.environ['LMDEPLOY_QWEN2_BYTELEVEL_TAIL'] = '0'
    disabled = new_module.Tokenizer(model)
    assert not disabled.model._use_bytelevel_tail
    result = dict(real_cases=len(data), random_cases=1000, chunk_checks=checks, mismatches=0,
                  cpu_first_chunk_ms={name:summary(values) for name,values in timing.items()},
                  rollback_flag_verified=True, scope='tokenizer-only; synthetic generated IDs; no inference or GPU load')
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
