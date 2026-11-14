# 作用：采集真实模型的输出 token，并离线逐块比较旧/新反分词；消除不同批次产生不同 token 的干扰。
# 示例：python verify_service_token_stream.py capture --dataset /path/cases.txt --output /path/tokens.json
import argparse
import concurrent.futures
import importlib.util
import json
import os
import random
import sys
import urllib.request
from pathlib import Path


def request_one(index, row, url):
    # 只请求额外的 output_ids 元数据；模型和生成参数与已有压测相同。
    body=dict(model='qwen',messages=row['input']['messages'],max_tokens=3,temperature=0,
              top_p=1.,top_k=1,repetition_penalty=1.,stream=False,return_token_ids=True,
              chat_template_kwargs=row.get('chat_template_kwargs') or {'enable_thinking':False})
    req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:
        data=json.load(response)
    choice=data['choices'][0]
    if choice['output_ids'] is None:
        raise RuntimeError('服务未返回 output_ids')
    return dict(index=index,ids=choice['output_ids'],content=choice['message']['content'],prompt_tokens=data['usage']['prompt_tokens'])


def load(path,name):
    # 两份源码以不同模块名加载，不修改运行服务的模块。
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    # capture 在宿主机访问现有服务，verify 在只使用 CPU 的容器中执行。
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['capture','verify'])
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--url',default='http://127.0.0.1:23333')
    parser.add_argument('--candidate')
    args=parser.parse_args()
    rows=[json.loads(s) for s in Path(args.dataset).read_text().splitlines() if s.strip()]
    target=Path(args.output)
    if args.mode=='capture':
        indices=random.Random(20261028).sample(range(len(rows)),512)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(request_one,i,rows[i],args.url) for i in indices]
            result=[f.result() for f in futures]
        target.write_text(json.dumps(result,ensure_ascii=False))
        print(json.dumps({'captured':len(result)}))
        return
    from lmdeploy.model import HFChatTemplate
    from lmdeploy import tokenizer as old_module
    new_module=load(args.candidate,'candidate_tokenizer')
    os.environ['LMDEPLOY_QWEN2_BYTELEVEL_TAIL']='1'
    old,new=old_module.Tokenizer('/models/V166_mindspeed_mg2hf'),new_module.Tokenizer('/models/V166_mindspeed_mg2hf')
    assert new.model._use_bytelevel_tail
    template=HFChatTemplate(model_path='/models/V166_mindspeed_mg2hf')
    chunks=0
    for item in json.loads(target.read_text()):
        row=rows[item['index']]
        prompt=template.messages2prompt(row['input']['messages'],**(row.get('chat_template_kwargs') or {'enable_thinking':False}))
        prefix=old.encode(prompt,add_bos=True)
        if len(prefix)!=item['prompt_tokens']:
            raise AssertionError('重建的输入 token 长度与服务不一致')
        a,b=old_module.DetokenizeState(len(prefix)),new_module.DetokenizeState(len(prefix))
        text=''
        for i in range(1,len(item['ids'])+1):
            ids=prefix+item['ids'][:i]
            expected,a=old.detokenize_incrementally(ids,a)
            actual,b=new.detokenize_incrementally(ids,b)
            assert expected==actual, f'生成 token 第{i}块文本不同'
            text+=actual
            chunks+=1
        assert text==item['content'], '离线重建文本与服务实际响应不同'
    result=dict(requests=512,chunks=chunks,mismatches=0,api_content_matches=True,fast_path_verified=True)
    target.with_suffix('.verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
