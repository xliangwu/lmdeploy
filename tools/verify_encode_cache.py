# 作用：逐token验证Qwen2片段编码与原版完全一致，覆盖真实输入、Unicode、选项回退、并发和内存淘汰。
# 示例：python verify_encode_cache.py --candidate-dir /path/candidate --dataset /path/cases.txt --output /path/result.json
import argparse
import concurrent.futures
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from lmdeploy.model import HFChatTemplate
from lmdeploy.tokenizer import Tokenizer as ReferenceTokenizer


def load_module(name, path):
    # 以独立模块名加载候选，不覆盖正在运行服务的安装文件。
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def stats(values):
    # CPU微测结果与服务性能分开报告。
    values=sorted(values)
    return dict(count=len(values),mean=sum(values)/len(values),p95=values[int((len(values)-1)*.95)])


def main():
    # 对照的环境开关关闭；候选必须实际进入受限快路径。
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-dir',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    directory=Path(args.candidate_dir)
    cache_module=load_module('lmdeploy.tokenizer_chunk_cache',directory/'tokenizer_chunk_cache.py')
    candidate_module=load_module('candidate_tokenizer',directory/'tokenizer.py')
    os.environ['LMDEPLOY_QWEN2_ENCODE_CACHE']='0'
    reference=ReferenceTokenizer('/models/V166_mindspeed_mg2hf')
    os.environ['LMDEPLOY_QWEN2_ENCODE_CACHE']='1'
    candidate=candidate_module.Tokenizer('/models/V166_mindspeed_mg2hf')
    assert candidate.model._chunk_encoder.supported
    template=HFChatTemplate(model_path='/models/V166_mindspeed_mg2hf')
    rows=[json.loads(x) for x in Path(args.dataset).read_text().splitlines() if x.strip()]
    timing={'reference':[],'candidate':[]}
    prompts=[]
    checks=0
    for i,row in enumerate(rows):
        text=template.messages2prompt(row['input']['messages'],**(row.get('chat_template_kwargs') or {'enable_thinking':False}))
        prompts.append(text)
        start=time.perf_counter()
        expected=reference.encode(text)
        timing['reference'].append((time.perf_counter()-start)*1000)
        start=time.perf_counter()
        actual=candidate.encode(text)
        timing['candidate'].append((time.perf_counter()-start)*1000)
        if expected!=actual:
            raise AssertionError('real token mismatch at '+str(i))
        checks+=1
    real_cache=candidate.model._chunk_encoder.cache_info()
    rng=random.Random(20261029)
    fragments=['中','文','e\u0301','🙂','👨‍👩‍👧‍👦','  ','\n','\r\n','\n  \n','\u2028','\u0301',"'s",'<|im_start|>','<|im_end|>','<|endoftext|>','123','!','\t','A']
    for i in range(1200):
        # 在边界前后放入组合字符、空白、CRLF和特殊token，强制产生多种候选切分位置。
        text=('x'*(1010+i%25))+''.join(rng.choice(fragments) for _ in range(160))+'\nA'+('z'*1024)+'\nB'
        for special in (False,True):
            expected=reference.encode(text,add_bos=False,add_special_tokens=special)
            actual=candidate.encode(text,add_bos=False,add_special_tokens=special)
            assert expected==actual, 'unicode boundary mismatch '+str(i)
            checks+=1
    # 非默认kwargs必须回退HF，尤其截断不能在每个片段单独应用。
    for opts in ({'truncation':True,'max_length':17},{'padding':'max_length','max_length':4096},{'return_token_type_ids':False}):
        text=prompts[0]
        assert reference.encode(text,**opts)==candidate.encode(text,**opts)
        checks+=1
    # 业务词表还包含非special的新增token；这些token同样不能跨换行边界匹配。
    for token in candidate.model.model.backend_tokenizer.get_added_tokens_decoder().values():
        text=('x'*1010)+token.content+'\n'+token.content+'中'+'\n'+('y'*1040)
        assert reference.encode(text)==candidate.encode(text)
        checks+=1
    reference.encode('reset')
    candidate.encode('reset')
    expected=[reference.encode(t) for t in prompts[:128]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        actual=list(pool.map(candidate.encode,prompts[:128]))
    assert expected==actual, 'concurrent encoding mismatch'
    checks+=len(actual)
    # 极小容量强制淘汰，并再次验证返回ID无变化。
    tiny=cache_module.Qwen2ChunkEncoder(candidate.model.model,max_entries=2,max_bytes=65536)
    for text in prompts[:32]:
        assert tiny.encode(text)==reference.encode(text)
        checks+=1
    assert tiny.bytes<=tiny.max_bytes and len(tiny.cache)<=tiny.max_entries
    # 改变预处理配置后必须回退，不能命中旧缓存；用独立实例避免影响上面的状态。
    from tokenizers.normalizers import Lowercase
    model=candidate.model.model
    saved=model.backend_tokenizer.normalizer
    try:
        model.backend_tokenizer.normalizer=Lowercase()
        text='ABC\nDEF'*300
        assert candidate.encode(text)==model.encode(text)
        checks+=1
    finally:
        model.backend_tokenizer.normalizer=saved
    result=dict(real_cases=len(rows),unicode_cases=1200,token_comparisons=checks,mismatches=0,
                timing_ms={k:stats(v) for k,v in timing.items()},real_input_cache=real_cache,
                final_cache=candidate.model._chunk_encoder.cache_info(),eviction_test=tiny.cache_info(),
                scope='CPU-only full token equality, no generation; stage timing is not QPS')
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
