# Copyright (c) OpenMMLab. All rights reserved.
# 作用：仅用CPU验证文本前处理不阻塞事件循环，并保留参数、输出和异常语义。
# 示例：python -m unittest discover -s tests/test_lmdeploy -p test_text_preprocess_async.py
import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from lmdeploy.serve.processors.multimodal import MultimodalProcessor


class TestTextPreprocessAsync(unittest.IsolatedAsyncioTestCase):
    """通过受控模板和tokenizer验证真实前处理方法，无需模型权重或GPU。"""

    def setUp(self):
        """每例使用独立模板及编码器，避免缓存或调用记录影响其他用例。"""
        self.template = SimpleNamespace(messages2prompt=Mock(return_value='rendered prompt'))
        self.tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2, 3]))
        self.processor = MultimodalProcessor(self.tokenizer, self.template)

    async def test_template_and_encode_leave_event_loop(self):
        """分别阻塞渲染和编码；事件循环必须仍能响应并释放worker。"""
        loop = asyncio.get_running_loop()
        event_loop_thread = threading.get_ident()
        for stage in ('template', 'encode'):
            with self.subTest(stage=stage):
                started = asyncio.Event()
                release = threading.Event()
                worker_threads = []
                result = 'rendered prompt' if stage == 'template' else [1, 2, 3]

                def block_worker(*args, **kwargs):
                    """通知事件循环worker已进入，再等待其释放；超时防止回归测试永久挂起。"""
                    worker_threads.append(threading.get_ident())
                    loop.call_soon_threadsafe(started.set)
                    if not release.wait(timeout=3):
                        raise TimeoutError('事件循环未能及时释放前处理worker')
                    return result

                self.template.messages2prompt.side_effect = block_worker if stage == 'template' else None
                self.tokenizer.encode.side_effect = block_worker if stage == 'encode' else None
                task = asyncio.create_task(self.processor._get_text_prompt_input('input', True, ''))
                try:
                    # 同步回归会阻塞loop并触发worker超时；正确实现可在worker等待时进入这里。
                    await asyncio.wait_for(started.wait(), timeout=1)
                    self.assertFalse(task.done())
                finally:
                    release.set()
                    actual = await asyncio.wait_for(task, timeout=4)
                self.assertEqual(actual, {'prompt': 'rendered prompt', 'input_ids': [1, 2, 3]})
                self.assertEqual(len(worker_threads), 1)
                self.assertNotEqual(worker_threads[0], event_loop_thread)

    async def test_prompt_options_and_none_content_preserved(self):
        """保留main的空content归一化、显式reasoning优先级及模板参数复制。"""
        messages = [{'role': 'assistant', 'content': None}]
        tools = [{'type': 'function', 'function': {'name': 'example'}}]
        options = {'reasoning_effort': 'low', 'enable_thinking': False}
        actual = await self.processor._get_text_prompt_input(
            messages, True, '', tools=tools, reasoning_effort='high', chat_template_kwargs=options)
        self.template.messages2prompt.assert_called_once_with(
            [{'role': 'assistant', 'content': ''}], tools=tools, reasoning_effort='high', enable_thinking=False)
        self.tokenizer.encode.assert_called_once_with('rendered prompt', add_bos=True)
        self.assertEqual(actual, {'prompt': 'rendered prompt', 'input_ids': [1, 2, 3]})
        self.assertEqual(options, {'reasoning_effort': 'low', 'enable_thinking': False})
        self.assertIsNone(messages[0]['content'])

    async def test_reasoning_option_fallback(self):
        """未提供独立reasoning参数时仍沿用chat_template_kwargs中的值。"""
        await self.processor._get_text_prompt_input(
            'input', True, '', chat_template_kwargs={'reasoning_effort': 'medium'})
        self.template.messages2prompt.assert_called_once_with('input', tools=None, reasoning_effort='medium')

    async def test_invalid_template_skips_encoding(self):
        """模板拒绝输入时保留ValueError，禁止把None交给tokenizer。"""
        self.template.messages2prompt.return_value = None
        with self.assertRaisesRegex(ValueError, 'base template'):
            await self.processor._get_text_prompt_input('input', True, '')
        self.tokenizer.encode.assert_not_called()

    async def test_worker_errors_propagate(self):
        """模板和编码异常必须由原协程抛出，不能转换为成功结果。"""
        for stage in ('template', 'encode'):
            with self.subTest(stage=stage):
                failure = RuntimeError(stage)
                self.template.messages2prompt.side_effect = failure if stage == 'template' else None
                self.tokenizer.encode.side_effect = failure if stage == 'encode' else None
                with self.assertRaises(RuntimeError) as caught:
                    await self.processor._get_text_prompt_input('input', True, '')
                self.assertIs(caught.exception, failure)


if __name__ == '__main__':
    unittest.main()
