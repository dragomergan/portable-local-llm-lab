"""Check that KoboldCPP is up and that both the chat model and the embedding model work.

Usage:
    python check_server.py
    python check_server.py --server http://localhost:5001

Standard library only. Start KoboldCPP first (start_model script).
"""
import argparse
import json
import sys
import time
import urllib.request


def call(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def step(name, func):
    start = time.time()
    try:
        message = func()
        print("  OK    %-12s %s  (%.1f s)" % (name, message, time.time() - start))
        return True
    except Exception as e:
        print("  FAIL  %-12s %s" % (name, e))
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:5001")
    base = ap.parse_args().server.rstrip("/")

    print("Checking", base)
    results = []

    results.append(step("server", lambda: "model: " + str(call(base + "/v1/models")["data"][0]["id"])))

    def chat():
        reply = call(base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 60})["choices"][0]["message"]["content"]
        return '"' + " ".join(reply.split())[:80] + '"'
    results.append(step("chat model", chat))

    def embedding():
        vec = call(base + "/v1/embeddings", {"input": "Hello embeddings"})["data"][0]["embedding"]
        return "%d numbers per vector" % len(vec)
    results.append(step("embeddings", embedding))

    print()
    if all(results):
        print("Everything works. You can open rag_chat.html or run the test scripts.")
    else:
        print("Something failed. Is KoboldCPP running? Was it started with an embedding model "
              "(a file with 'embed' in its name in the models folder)?")
        sys.exit(1)


if __name__ == "__main__":
    main()
