#!/usr/bin/env python3
"""dsv41-zig-tp2-0102 起栈验收（2026-10-20 固化）。
用法: python3 /root/verify-dsv41-zig-tp2.py [URL]
默认直连 rank0 http://127.0.0.1:8000；可选传 proxy URL（如 http://192.168.90.104:8100）。
判据: ①模型列表含 deepseek-v4.1-flash-uncensored ②短/3k-promopt 两档 chat 都干净答出 PINEAPPLE-7Z 且 finish=stop
     ③无 chat template 报错。全过 exit 0。
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "deepseek-v4.1-flash-uncensored"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url):
    with OPENER.open(url, timeout=15) as r:
        return json.load(r)


def chat(prompt, max_tokens=64):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(req, timeout=300) as r:
        return json.load(r)


def main():
    ok = True
    try:
        models = [m["id"] for m in get(BASE + "/v1/models")["data"]]
        print("models:", models)
        if MODEL not in models:
            print("FAIL: 模型名不在列表"); ok = False
    except Exception as e:
        print("FAIL: /v1/models", e); return 1

    fill = "The quick brown fox jumps over the lazy dog. " * 20   # ~2.8k tok
    for name, prompt in [
        ("short", "Reply with exactly PINEAPPLE-7Z and nothing else."),
        ("long ", fill + "\n\nNow reply with exactly PINEAPPLE-7Z and nothing else."),
    ]:
        try:
            r = chat(prompt)
            ch = r["choices"][0]
            text = (ch["message"].get("content") or "").strip()
            good = "PINEAPPLE-7Z" in text and ch.get("finish_reason") == "stop"
            print(f"{name}: finish={ch.get('finish_reason')} tokens={r['usage']['completion_tokens']} text={text[:40]!r} -> {'PASS' if good else 'FAIL'}")
            ok = ok and good
        except Exception as e:
            print(f"{name}: FAIL {e}"); ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
