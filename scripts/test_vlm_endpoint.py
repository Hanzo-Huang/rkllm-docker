#!/usr/bin/env python3
"""Exercise a local OpenAI-compatible VLM endpoint and save reproducible evidence.

The supplied image and expected words define a small smoke test, not a benchmark.
Uses only the Python standard library; model artifacts are never uploaded.
"""
import argparse
import base64
import json
import mimetypes
from pathlib import Path
import re
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18001')
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--expect', action='append', default=[])
    parser.add_argument('--expect-any', nargs='+', default=[])
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    results = {'model': args.model, 'base_url': base, 'tests': []}

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=args.timeout)

    def chat(content, stream=False):
        return request('/v1/chat/completions', {
            'model': args.model, 'messages': [{'role': 'user', 'content': content}],
            'stream': stream, 'temperature': 0, 'top_k': 1,
            'max_tokens': 128, 'enable_thinking': False,
        })

    def record(name, operation):
        start = time.monotonic()
        try:
            details = operation()
            result = {'name': name, 'passed': True, **details}
        except Exception as error:
            result = {'name': name, 'passed': False, 'error': str(error)}
        result['elapsed_seconds'] = round(time.monotonic() - start, 3)
        results['tests'].append(result)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)

    def health():
        with request('/health') as response:
            value = json.load(response)
        if value.get('status') != 'healthy':
            raise RuntimeError(f'Unhealthy server: {value}')
        return {'response': value}

    def text():
        with chat('What is 2 + 2? Answer with only the number.') as response:
            value = json.load(response)
        answer = value['choices'][0]['message']['content']
        if '4' not in answer:
            raise RuntimeError(f'Unexpected arithmetic answer: {answer!r}')
        return {'response': value}

    def vision():
        mime = mimetypes.guess_type(args.image.name)[0] or 'image/jpeg'
        image = 'data:' + mime + ';base64,' + base64.b64encode(args.image.read_bytes()).decode()
        content = [{'type': 'image_url', 'image_url': {'url': image}},
                   {'type': 'text', 'text': args.question}]
        with chat(content) as response:
            value = json.load(response)
        answer = value['choices'][0]['message']['content']
        missing_required = any(word.lower() not in answer.lower() for word in args.expect)
        missing_alternative = args.expect_any and not any(
            re.search(r'\b' + re.escape(word) + r'\b', answer, re.IGNORECASE)
            for word in args.expect_any)
        if not answer.strip() or missing_required or missing_alternative:
            raise RuntimeError(f'Vision answer failed required={args.expect}, any={args.expect_any}: {answer!r}')
        return {'response': value, 'image': str(args.image), 'question': args.question}

    def streaming():
        chunks, text, done = [], '', False
        with chat('Reply with the single word hello.', stream=True) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    done = True
                    break
                chunk = json.loads(data)
                if 'error' in chunk:
                    raise RuntimeError(str(chunk['error']))
                chunks.append(chunk)
                for choice in chunk.get('choices', []):
                    text += choice.get('delta', {}).get('content') or ''
        if not done or 'hello' not in text.lower():
            raise RuntimeError(f'Invalid stream: done={done}, answer={text!r}')
        return {'answer': text, 'done': done, 'chunks': chunks}

    for name, operation in [('health', health), ('text', text), ('vision', vision), ('streaming', streaming)]:
        record(name, operation)
    results['passed'] = all(test['passed'] for test in results['tests'])
    args.report.write_text(json.dumps(results, indent=2) + '\n')
    if not results['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
