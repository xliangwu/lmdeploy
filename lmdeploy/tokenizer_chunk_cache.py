# Copyright (c) OpenMMLab. All rights reserved.
# 作用：在严格匹配的Qwen2 tokenizer上缓存独立文本片段的编码，减少重复提示词的CPU处理。
# 示例：LMDEPLOY_QWEN2_ENCODE_CACHE=1 lmdeploy serve api_server /models/V166_mindspeed_mg2hf
import json
import re
import sys
import threading
from collections import OrderedDict


class Qwen2ChunkEncoder:
    """仅在已验证的NFC、Qwen正则和无新增特殊token后处理下复用输入片段；不保存模型输出。"""

    _PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    _BOUNDARY = re.compile(r'[\r\n](?=\S)')

    def __init__(self, model, chunk_chars=1024, max_entries=2048, max_bytes=16*1024*1024):
        # 缓存按真实Python对象估算内存、限定条目数；锁只保护缓存，不串行化实际编码。
        self.model = model
        self.backend = model.backend_tokenizer
        self.chunk_chars = chunk_chars
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0
        self.supported = self._supports(model)
        self.signature = self._signature() if self.supported else None

    def _supports(self, model):
        # 正则/归一化/后处理必须精确匹配；不对其他分词器猜测边界可分性。
        if (not model.is_fast or type(model).__name__ not in ('Qwen2Tokenizer', 'Qwen2TokenizerFast')
                or not type(model).__module__.startswith('transformers.models.qwen2.')):
            return False
        state = json.loads(self.backend.to_str())
        expected_split = dict(type='Split', pattern={'Regex': self._PATTERN}, behavior='Isolated', invert=False)
        expected_bytelevel = dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=False)
        post = state.get('post_processor') or {}
        identity_single = [{'Sequence': {'id': 'A', 'type_id': 0}}]
        added = state.get('added_tokens', [])
        model_state = state.get('model') or {}
        return (model_state.get('type') == 'BPE' and model_state.get('dropout') is None
                and not model_state.get('continuing_subword_prefix') and not model_state.get('end_of_word_suffix')
                and state.get('normalizer') == {'type': 'NFC'}
                and state.get('pre_tokenizer') == dict(type='Sequence', pretokenizers=[expected_split, expected_bytelevel])
                and post.get('type') == 'TemplateProcessing' and post.get('single') == identity_single
                and not post.get('special_tokens')
                and all(t.get('content') and not t.get('normalized')
                        and not t.get('lstrip') and not t.get('rstrip') and not t.get('single_word')
                        and '\n' not in t['content'] and '\r' not in t['content'] for t in added))

    def _signature(self):
        # 检测外部变更，避免归一化、分割器或新增token变化后使用旧缓存。
        return (len(self.model), self.backend.model.dropout, self.backend.model.continuing_subword_prefix,
                self.backend.model.end_of_word_suffix, self.backend.normalizer.__getstate__(),
                self.backend.pre_tokenizer.__getstate__(), self.backend.post_processor.__getstate__())

    def _encode_chunk(self, text, add_special_tokens):
        # key包含特殊token开关；返回不可变tuple，调用者不会污染后续请求。
        key = (text, bool(add_special_tokens))
        with self.lock:
            found = self.cache.get(key)
            if found is not None:
                self.hits += 1
                self.cache.move_to_end(key)
                return found[0]
            self.misses += 1
        encoded = tuple(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)
        # 计入字符串、tuple、整数及键/字典元数据估算，限制总常驻大小。
        size = sys.getsizeof(text) + sys.getsizeof(encoded) + sum(sys.getsizeof(x) for x in encoded) + 256
        if size > self.max_bytes:
            return encoded
        with self.lock:
            # 并发首次编码同一片段时，不重复增加缓存占用。
            found = self.cache.get(key)
            if found is not None:
                self.cache.move_to_end(key)
                return found[0]
            while self.cache and (len(self.cache) >= self.max_entries or self.bytes + size > self.max_bytes):
                _, (_, removed_size) = self.cache.popitem(last=False)
                self.bytes -= removed_size
                self.evictions += 1
            self.cache[key] = (encoded, size)
            self.bytes += size
        return encoded

    def encode(self, text, add_special_tokens=True, **kwargs):
        # 带截断/填充/成对输入等选项时保留HF行为，任何不匹配均回退原路径。
        if (not self.supported or not isinstance(text, str) or kwargs
                or self.backend.truncation is not None or self.backend.padding is not None
                or self._signature() != self.signature):
            return self.model.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        result = []
        begin = 0
        while begin < len(text):
            # 只在换行后立即为非空白字符的位置切分：不会拆开Qwen正则中的空白段或特殊token。
            # NFC不能跨换行进行字符合成，且身份后处理不会为每一片段额外添加token。
            target = begin + self.chunk_chars
            match = self._BOUNDARY.search(text, target) if target < len(text) else None
            end = match.end() if match is not None else len(text)
            result.extend(self._encode_chunk(text[begin:end], add_special_tokens))
            begin = end
        return result

    def cache_info(self):
        # 返回聚合计数，绝不暴露缓存文本；供离线验收和诊断复核。
        with self.lock:
            return dict(supported=self.supported, hits=self.hits, misses=self.misses,
                        evictions=self.evictions, entries=len(self.cache), estimated_bytes=self.bytes,
                        max_entries=self.max_entries, max_bytes=self.max_bytes)
