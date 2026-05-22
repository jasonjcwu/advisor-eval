#!/usr/bin/env python3
"""SWE-bench: Sonnet+Opus advisor evaluation via PackyAPI."""

import json, os, subprocess, sys, time, uuid
from pathlib import Path

HERMES_ROOT = "/root/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

PACKYAPI_KEY = "sk-Y8H1iV4oTJ0sppszOAjDopIb4ht7LhSCdBHMHTxZZACrYrzY"
PACKYAPI_BASE = "https://www.packyapi.com"
EXECUTOR_MODEL = "claude-sonnet-4-6"
ADVISOR_MODEL = "claude-opus-4-7"

os.environ["HERMES_ADVISOR_MODEL"] = ADVISOR_MODEL
os.environ["HERMES_ADVISOR_API_KEY"] = PACKYAPI_KEY

from anthropic_adapter import AnthropicClient
from tools.advisor_tool import call_advisor, load_advisor_config

EXECUTOR_SYSTEM_PROMPT = """\
You are a coding agent solving software engineering tasks.

Tools: file_read, file_edit, bash_run, ask_advisor (consult Claude Opus).

**When to call ask_advisor:**
1. Explore first, then BEFORE writing code, call ask_advisor
2. Call ask_advisor when stuck or approach fails
3. Call ask_advisor before declaring done

Be concise. Focus on solving the task.
"""

SOLO_SYSTEM_PROMPT = """\
You are a coding agent solving software engineering tasks.

Tools: file_read, file_edit, bash_run.

Work methodically: explore, identify root cause, make minimal edits, verify.
Be concise.
"""

TOOLS = [
    {"type":"function","function":{"name":"file_read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"file_edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"]}}},
    {"type":"function","function":{"name":"bash_run","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer","default":30}},"required":["command"]}}},
    {"type":"function","function":{"name":"ask_advisor","description":"Consult Claude Opus for strategic guidance","parameters":{"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}}},
]

def execute_tool(name, args, workdir, messages):
    if name == "file_read":
        path = os.path.join(workdir, args["path"]) if not args["path"].startswith("/") else args["path"]
        try:
            with open(path) as f: content = f.read()
            return content[:10000] + (f"\n...(truncated {len(content)} chars)" if len(content)>10000 else "")
        except Exception as e: return f"Error: {e}"
    elif name == "file_edit":
        path = os.path.join(workdir, args["path"]) if not args["path"].startswith("/") else args["path"]
        try:
            with open(path) as f: c = f.read()
            if args["old_string"] not in c: return f"Error: old_string not found"
            with open(path,"w") as f: f.write(c.replace(args["old_string"], args["new_string"], 1))
            return f"Edited {path}"
        except Exception as e: return f"Error: {e}"
    elif name == "bash_run":
        try:
            r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=args.get("timeout",30), cwd=workdir)
            out = r.stdout + ("\nSTDERR:\n"+r.stderr if r.stderr else "") + (f"\nExit:{r.returncode}" if r.returncode else "")
            return out[:8000]
        except subprocess.TimeoutExpired: return "Timeout"
        except Exception as e: return f"Error: {e}"
    elif name == "ask_advisor":
        q = args.get("question") or args.get("message") or str(args)
        cfg = load_advisor_config()
        cfg.update(model=ADVISOR_MODEL, api_key=PACKYAPI_KEY, base_url=PACKYAPI_BASE, provider=None)
        print(f"      [Advisor] {q[:80]}...")
        raw = call_advisor(messages=messages, question=q, urgency="medium", config=cfg)
        try:
            p = json.loads(raw)
            if "error" in p: return f"Advisor error: {p['error']}"
            print(f"      [Advisor] -> {p.get('tokens_in',0)}/{p.get('tokens_out',0)} tokens")
            return f"[Opus]: {p['advice']}"
        except: return raw
    return f"Unknown: {name}"

def run_task(prompt, workdir, max_turns=12, solo=False):
    client = AnthropicClient(api_key=PACKYAPI_KEY, base_url=PACKYAPI_BASE, timeout=300)
    sp = SOLO_SYSTEM_PROMPT if solo else EXECUTOR_SYSTEM_PROMPT
    tools = [t for t in TOOLS if solo or t["function"]["name"]!="dummy"] if solo else TOOLS
    if solo: tools = [t for t in TOOLS if t["function"]["name"]!="ask_advisor"]
    messages = [{"role":"system","content":sp},{"role":"user","content":prompt}]
    adv = ti = to = 0; tcs = {}
    for turn in range(max_turns):
        try:
            resp = client.chat.completions.create(model=EXECUTOR_MODEL, messages=messages, tools=tools, max_tokens=4096, temperature=0.2)
        except Exception as e: print(f"      API error: {e}"); break
        u = resp.usage; ti += getattr(u,'input_tokens',0); to += getattr(u,'output_tokens',0)
        c = resp.choices[0]; m = c.message
        am = {"role":"assistant","content":m.content or ""}
        if getattr(m,'reasoning_content',None): am["reasoning_content"]=m.reasoning_content
        if m.tool_calls: am["tool_calls"]=m.tool_calls
        messages.append(am)
        if not m.tool_calls or c.finish_reason=="stop": break
        for tc in m.tool_calls:
            fn = tc["function"]["name"]; tcs[fn]=tcs.get(fn,0)+1
            try: ta = json.loads(tc["function"]["arguments"])
            except: ta = {}
            if fn=="ask_advisor": adv+=1
            print(f"      {fn}", end="")
            r = execute_tool(fn, ta, workdir, messages)
            print(f" -> {str(r)[:60].replace(chr(10),' ')}")
            messages.append({"role":"tool","tool_call_id":tc["id"],"content":str(r)})
    return {"advisor_calls":adv,"tool_calls":tcs,"tokens_in":ti,"tokens_out":to,"turns":turn+1}

def setup_workspace(inst, base="/tmp/swe-sonnet"):
    repo = inst.get("repo",""); bc = inst.get("base_commit",""); iid = inst["instance_id"]
    ws = os.path.join(base, iid.replace("/","__"))
    if os.path.exists(ws):
        try:
            subprocess.run(["git","checkout","--quiet",bc],check=True,timeout=30,capture_output=True,cwd=ws)
            subprocess.run(["git","clean","-fdq"],check=True,timeout=30,capture_output=True,cwd=ws)
        except: pass
        return ws
    os.makedirs(base,exist_ok=True)
    print(f"    Clone {repo}...")
    try:
        subprocess.run(["git","clone","--quiet",f"https://github.com/{repo}.git",ws],check=True,timeout=120,capture_output=True)
        subprocess.run(["git","checkout","--quiet",bc],check=True,timeout=30,capture_output=True,cwd=ws)
    except Exception as e:
        print(f"    Clone fail: {e}")
        os.makedirs(ws,exist_ok=True)
    return ws

def build_prompt(inst, solo=False):
    repo = inst.get("repo",""); issue = inst.get("problem_statement","")
    hints = inst.get("hints_text","")
    p = f"## Task: Fix bug in {repo}\n\n{issue}\n\n"
    if hints: p += f"## Hints\n{hints}\n\n"
    if solo:
        p += "## Instructions\n1. Explore codebase\n2. Identify root cause\n3. Make minimal edits\n4. Verify\n\nCodebase is in cwd. Start by listing files.\n"
    else:
        p += "## Instructions\n1. Explore codebase\n2. Identify root cause\n3. Make minimal edits\n4. Verify\n\nCodebase is in cwd. **IMPORTANT: Before making ANY edits, call ask_advisor FIRST to discuss your approach with Claude Opus.** Then list files and start.\n"
    return p

def get_diff(ws):
    try:
        if not os.path.exists(os.path.join(ws,".git")): return ""
        r = subprocess.run(["git","diff"],capture_output=True,text=True,cwd=ws,timeout=30)
        return r.stdout
    except: return ""

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set",default="/root/advisor-eval/eval_set_swe_hard6.json")
    p.add_argument("--limit",type=int)
    p.add_argument("--max-turns",type=int,default=12)
    p.add_argument("--output",default="/root/advisor-eval/swe_bench_results/sonnet-opus")
    p.add_argument("--solo",action="store_true")
    a = p.parse_args()
    with open(a.eval_set) as f: instances = json.load(f)
    if a.limit: instances = instances[:a.limit]
    if a.solo: a.output = a.output.replace("sonnet-opus","sonnet-solo")
    mode = "SOLO" if a.solo else f"+ {ADVISOR_MODEL}"
    print(f"Sonnet {mode} | {len(instances)} instances | {PACKYAPI_BASE}")
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    results = []
    for idx,inst in enumerate(instances):
        iid = inst["instance_id"]
        print(f"\n[{idx+1}/{len(instances)}] {iid}")
        ws = setup_workspace(inst)
        prompt = build_prompt(inst, solo=a.solo)
        t0 = time.time()
        try: info = run_task(prompt, ws, max_turns=a.max_turns, solo=a.solo)
        except Exception as e:
            import traceback; traceback.print_exc()
            info = {"error":str(e),"advisor_calls":0,"tool_calls":{},"turns":0,"tokens_in":0,"tokens_out":0}
        elapsed = time.time()-t0
        patch = get_diff(ws)
        r = {"instance_id":iid,"model_name_or_path":f"{EXECUTOR_MODEL}+{ADVISOR_MODEL}" if not a.solo else EXECUTOR_MODEL,"model_patch":patch,"advisor_calls":info.get("advisor_calls",0),"turns":info.get("turns",0),"tokens_in":info.get("tokens_in",0),"tokens_out":info.get("tokens_out",0)}
        results.append(r)
        print(f"    {'OK' if patch else '--'} patch={bool(patch)} adv={info.get('advisor_calls',0)} turns={info.get('turns','?')} {elapsed:.0f}s")
        with open(out/"predictions.jsonl","a") as f:
            f.write(json.dumps({"instance_id":iid,"model_name_or_path":r["model_name_or_path"],"model_patch":patch},ensure_ascii=False)+"\n")
        pat = sum(1 for x in results if x["model_patch"])
        with open(out/"metrics.json","w") as f:
            json.dump({"executor":EXECUTOR_MODEL,"advisor":ADVISOR_MODEL if not a.solo else None,"total":len(instances),"completed":len(results),"patches":pat,"adv_calls":sum(x["advisor_calls"] for x in results),"tokens_in":sum(x["tokens_in"] for x in results)},f,indent=2)
    print(f"\nDone: {sum(1 for x in results if x['model_patch'])}/{len(results)} patches | {sum(x['advisor_calls'] for x in results)} advisor calls")
