"""Serve a local Laya checkpoint through the teacher evaluator's choice API.

Install Laya in a separate environment, then run this script with a pinned
Hugging Face snapshot directory as ``--model`` for reproducible evaluation.
The server binds to localhost only by default.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(agent, model_id: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": model_id})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("request must be 1-65536 bytes")
                body = json.loads(self.rfile.read(length))
                state = body["state"]
                question = body["questions"]["decision"]
                if not isinstance(state, str) or question.get("type") != "choice":
                    raise ValueError("expected a string state and choice question")
                if set(question.get("criteria", {})) != {"DOWN", "FLAT", "UP"}:
                    raise ValueError("expected DOWN/FLAT/UP criteria")
                answer = agent.predict(state, {"decision": question})["answers"]["decision"]
                self._send(200, {"answers": {"decision": answer}})
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                self._send(400, {"error": str(exc)})

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convaiinnovations/laya",
                        help="Hugging Face model ID or pinned local snapshot directory")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()

    import laya

    agent = laya.load(args.model, device=args.device)
    HTTPServer((args.host, args.port), make_handler(agent, args.model)).serve_forever()


if __name__ == "__main__":
    main()
