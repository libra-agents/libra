#!/usr/bin/env python3
"""Lightweight HTTP server for SciToolAgent tool execution inside a container."""

import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

TOOLS_DIR = os.environ.get("TOOLS_DIR", "/app/ToolsAgent")
PORT = int(os.environ.get("TOOL_SERVER_PORT", "8100"))
CALL_TIMEOUT = int(os.environ.get("TOOL_CALL_TIMEOUT", "120"))

_GLOBAL_SEED = os.environ.get("TOOL_RANDOM_SEED")
if _GLOBAL_SEED is not None:
    import random, numpy as np  # noqa: E401
    _seed = int(_GLOBAL_SEED)
    random.seed(_seed)
    np.random.seed(_seed)

sys.path.insert(0, TOOLS_DIR)
os.chdir(TOOLS_DIR)

from ToolsFuns.tools_dict import TOOLS_MAPPING  # noqa: E402

tool_categories: dict[str, str] = {}
for _cat, _names in TOOLS_MAPPING.items():
    for _n in _names:
        tool_categories[_n] = _cat

_tool_funcs: dict[str, object] = {}
_load_errors: list[str] = []


def _load_all_tools():
    """Best-effort load: try whole category first, then selective-import fallback."""
    for category, names in TOOLS_MAPPING.items():
        cat_base = category.replace("Tools", "")
        module_path = f"ToolsFuns.{cat_base}.tool_name_dict"
        dict_name = f"{cat_base.upper()}_TOOLS_DICT"
        try:
            mod = importlib.import_module(module_path)
            tools_dict = getattr(mod, dict_name, {})
            for name, func in tools_dict.items():
                if callable(func):
                    _tool_funcs[name] = func
            continue
        except Exception:
            pass

        # Fallback: exec each import line individually (skip failures),
        # then build the dict in the accumulated namespace.
        cat_dir = os.path.join(TOOLS_DIR, "ToolsFuns", cat_base)
        dict_path = os.path.join(cat_dir, "tool_name_dict.py")
        if not os.path.isfile(dict_path):
            _load_errors.append(f"{category}: tool_name_dict.py not found")
            continue

        with open(dict_path) as f:
            lines = f.readlines()

        ns: dict = {"__builtins__": __builtins__}
        import_lines = []
        dict_lines = []
        in_dict = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") and "import" in stripped:
                import_lines.append(line)
            elif dict_name in stripped and "=" in stripped:
                in_dict = True
                dict_lines.append(line)
            elif in_dict:
                dict_lines.append(line)

        for imp_line in import_lines:
            try:
                exec(imp_line, ns)
            except Exception as exc:
                _load_errors.append(f"{category}/{imp_line.strip()}: {exc}")

        # Build dict entry-by-entry, skipping entries with undefined names
        dict_source = "".join(dict_lines)
        if dict_source:
            try:
                exec(compile(dict_source, dict_path, "exec"), ns)
            except NameError:
                # Some names missing; build dict one entry at a time
                for name in names:
                    for line in dict_lines:
                        if f'"{name}"' in line or f"'{name}'" in line:
                            try:
                                entry_code = f"__entry = {{{line.strip().rstrip(',')}}}"
                                exec(compile(entry_code, dict_path, "exec"), ns)
                                for k, v in ns.pop("__entry", {}).items():
                                    if callable(v):
                                        _tool_funcs[k] = v
                            except Exception:
                                pass
                            break
                continue

        tools_dict = ns.get(dict_name, {})
        for name, func in tools_dict.items():
            if callable(func):
                _tool_funcs[name] = func


_load_all_tools()
print(f"  Loaded {len(_tool_funcs)}/{len(tool_categories)} tool functions", flush=True)
if _load_errors:
    print(f"  {len(_load_errors)} sub-module load failures (non-fatal)", flush=True)


def run_tool(name: str, input_str: str):
    func = _tool_funcs.get(name)
    if func is None:
        cat = tool_categories.get(name, "unknown")
        raise RuntimeError(f"Tool {name} unavailable (category {cat}): not loaded")
    return func(input_str)


_fork = multiprocessing.get_context("fork")


def _run_tool_child(send, name: str, input_str: str):
    try:
        result = run_tool(name, input_str)
        send.send((None, str(result) if result is not None else ""))
    except BaseException:
        send.send((traceback.format_exc(), None))
    finally:
        send.close()


def run_tool_with_timeout(name: str, input_str: str, timeout: int):
    receive, send = _fork.Pipe(duplex=False)
    process = _fork.Process(
        target=_run_tool_child,
        args=(send, name, input_str),
    )
    process.start()
    send.close()
    try:
        if not receive.poll(timeout):
            return None, f"tool {name} timed out after {timeout}s"
        try:
            error, output = receive.recv()
        except EOFError:
            return None, f"tool {name} exited without a result"
        return output, error
    finally:
        receive.close()
        if process.is_alive():
            process.kill()
        process.join()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json_response({
                "ok": True,
                "tools_registered": len(tool_categories),
                "tools_loaded": len(_tool_funcs),
                "load_errors": len(_load_errors),
            })
        elif self.path == "/tools":
            self._json_response({"tools": sorted(_tool_funcs.keys())})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/invoke":
            self._handle_invoke()
        elif self.path == "/exec":
            self._handle_exec()
        else:
            self._json_response({"error": "not found"}, 404)

    def _handle_invoke(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        tool_name = body.get("tool_name", "")
        input_str = body.get("input", "")
        timeout = body.get("timeout", CALL_TIMEOUT)

        if not tool_name:
            self._json_response({"output": None, "error": "missing tool_name"}, 400)
            return

        output, error = run_tool_with_timeout(tool_name, input_str, timeout)
        self._json_response(
            {"output": output, "error": error},
            504 if error and " timed out after " in error else 200,
        )

    def _handle_exec(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        code = body.get("code", "")
        timeout = body.get("timeout", CALL_TIMEOUT)

        if not code:
            self._json_response({"stdout": "", "stderr": "", "error": "missing code"}, 400)
            return

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                script_path = f.name
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=timeout,
            )
            os.unlink(script_path)
            self._json_response({
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "error": None,
            })
        except subprocess.TimeoutExpired:
            self._json_response({"stdout": "", "stderr": "", "error": f"script timed out after {timeout}s"}, 504)
        except Exception:
            self._json_response({"stdout": "", "stderr": "", "error": traceback.format_exc()}, 200)

    def _json_response(self, data, code=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        sys.stderr.write(f"[tool_server] {args[0]}\n")


if __name__ == "__main__":
    print(f"SciToolAgent tool server starting on 0.0.0.0:{PORT}", flush=True)
    print(f"  Tools directory: {TOOLS_DIR}", flush=True)
    print(f"  Available tools: {len(tool_categories)}", flush=True)
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
