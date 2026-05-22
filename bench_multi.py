#!/usr/bin/env python3
"""
Multi-model advisor benchmark — SWE-bench hard6 evaluation.

Supported model pairs:
  claude-haiku+opus  : Haiku 4.5 executor + Opus 4.7 advisor (PackyAPI)
  claude-sonnet+opus : Sonnet 4.6 executor + Opus 4.7 advisor (PackyAPI)
  ds-flash+pro       : DeepSeek V4 Flash executor + V4 Pro advisor
  gpt-mini+gpt55     : GPT-5.4-mini executor + GPT-5.5 advisor (OpenAI — pending key)

Each pair runs both solo (no advisor) and advisor mode for comparison.
"""

import json, os, subprocess, sys, time, uuid
from pathlib import Path

HERMES_ROOT = "/root/hermes-agent"
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

# ── Model registry ───────────────────────────────────────────────────
MODELS = {
    # PackyAPI (Anthropic-format)
    "claude-haiku": {
        "model": "claude-haiku-4-5-20251001",
        "api_key": "sk-Y8H1iV4oTJ0sppszOAjDopIb4ht7LhSCdBHMHTxZZACrYrzY",
        "base_url": "https://www.packyapi.com",
        "api_mode": "anthropic",
    },
    "claude-sonnet": {
        "model": "claude-sonnet-4-6",
        "api_key": "sk-Y8H1iV4oTJ0sppszOAjDopIb4ht7LhSCdBHMHTxZZACrYrzY",
        "base_url": "https://www.packyapi.com",
        "api_mode": "anthropic",
    },
    "claude-opus": {
        "model": "claude-opus-4-7",
        "api_key": "sk-Y8H1iV4oTJ0sppszOAjDopIb4ht7LhSCdBHMHTxZZACrYrzY",
        "base_url": "https://www.packyapi.com",
        "api_mode": "anthropic",
    },
    # DeepSeek (OpenAI-compatible)
    "ds-flash": {
        "model": "deepseek-v4-flash",
        "api_key": None,  # auto-detected from hermes auth
        "base_url": "https://api.deepseek.com/v1",
        "api_mode": "openai",
    },
    "ds-pro": {
        "model": "deepseek-chat",  # V4 Pro
        "api_key": None,
        "base_url": "https://api.deepseek.com/v1",
        "api_mode": "openai",
    },
    # GPT (OpenAI — pending key)
    "gpt-mini": {
        "model": "gpt-5.4-mini",
        "api_key": "sk-54U6OgvJ5S4Xq4GS92G1K6RsQplq6tregYfTTE4wZNLHwzpP",
        "base_url": "https://www.packyapi.com/v1",
        "api_mode": "openai",
    },
    "gpt-55": {
        "model": "gpt-5.5",
        "api_key": "sk-54U...wzpP",
        "base_url": "https://www.packyapi.com/v1",
        "api_mode": "openai",
    },
    # GLM (智谱 — OpenAI-compatible via coding endpoint)
    "glm-air": {
        "model": "glm-4.5-air",
        "api_key": None,  # auto from hermes auth
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_mode": "openai",
    },
    "glm-5.1": {
        "model": "glm-5.1",
        "api_key": None,  # auto from hermes auth
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_mode": "openai",
    },
}

# ── Model pairs to evaluate ───────────────────────────────────────────
PAIRS = [
    ("claude-haiku", "claude-opus"),
    ("claude-sonnet", "claude-opus"),
    ("ds-flash", "ds-pro"),
    ("glm-air", "glm-5.1"),
    # ("gpt-mini", "gpt-55"),  # PackyAPI GPT returns 400 with tools
]

# ── Load DeepSeek keys from hermes auth ───────────────────────────────
def _load_ds_key():
    try:
        with open(os.path.expanduser("~/.hermes/auth.json")) as f:
            auth = json.load(f)
        pool = auth.get("credential_pool", {})
        for name in ["deepseek", "custom:deepseek"]:
            creds = pool.get(name, [])
            if creds:
                return creds[0].get("access_token", "")
    except Exception:
        pass
    return os.environ.get("DEEPSEEK_API_KEY", "")

DS_KEY = _load_ds_key()

def _load_glm_key():
    try:
        with open(os.path.expanduser("~/.hermes/auth.json")) as f:
            auth = json.load(f)
        pool = auth.get("credential_pool", {})
        for name in ["custom:glmcode", "glmcode"]:
            creds = pool.get(name, [])
            if creds:
                return creds[0].get("access_token", "")
    except Exception:
        pass
    return os.environ.get("GLMCODE_API_KEY", "")

GLM_KEY = _load_glm_key()

# ── Anthropic adapter ─────────────────────────────────────────────────
from anthropic_adapter import AnthropicClient

# ── Hermes advisor tool ──────────────────────────────────────────────
from tools.advisor_tool import call_advisor, load_advisor_config

# ── System prompts ────────────────────────────────────────────────────
EXECUTOR_PROMPT = """\
You are a coding agent solving software engineering tasks.

Tools: file_read, file_edit, bash_run, ask_advisor (consult a stronger model).

**When to call ask_advisor:**
1. Explore first, then BEFORE writing code, call ask_advisor
2. Call ask_advisor when stuck or approach fails
3. Call ask_advisor before declaring done

Be concise. Focus on solving the task.
"""

SOLO_PROMPT = """\
You are a coding agent solving software engineering tasks.

Tools: file_read, file_edit, bash_run.

Work methodically: explore, identify root cause, make minimal edits, verify.
Be concise.
"""

# ── Tool definitions ──────────────────────────────────────────────────
TOOLS = [
    {"type":"function","function":{"name":"file_read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"file_edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"]}}},
    {"type":"function","function":{"name":"bash_run","description":"Run a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer","default":30}},"required":["command"]}}},
    {"type":"function","function":{"name":"ask_advisor","description":"Consult a stronger model for strategic guidance","parameters":{"type":"object","properties":{"question":{"type":"string"}},"required":["question"]}}},
]


def execute_tool(name, args, workdir, messages, advisor_cfg):
    if name == "file_read":
        path = os.path.join(workdir, args["path"]) if not args["path"].startswith("/") else args["path"]
        try:
            with open(path) as f: c = f.read()
            return c[:10000] + (f"\n...(truncated {len(c)} total)" if len(c)>10000 else "")
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
        print(f"      🧠 advisor: {q[:80]}...")
        raw = call_advisor(messages=messages, question=q, urgency="medium", config=advisor_cfg)
        try:
            p = json.loads(raw)
            if "error" in p: return f"Advisor error: {p['error']}"
            print(f"      📋 -> {p.get('tokens_in',0)}↓/{p.get('tokens_out',0)}↑ tokens")
            return f"[Advisor]: {p['advice']}"
        except: return raw
    return f"Unknown: {name}"


def run_task(prompt, workdir, max_turns, solo, exec_cfg, advisor_cfg):
    """Run agent loop on one task."""
    api_mode = exec_cfg["api_mode"]
    
    if api_mode == "anthropic":
        client = AnthropicClient(api_key=exec_cfg["api_key"], base_url=exec_cfg["base_url"], timeout=300)
        use_openai = False
    else:
        import openai
        # Pick the right key: explicit > match by base_url > fallback chain
        key = exec_cfg["api_key"]
        if not key:
            base = exec_cfg.get("base_url", "")
            if "bigmodel" in base:
                key = GLM_KEY
            elif "deepseek" in base:
                key = DS_KEY
            else:
                key = DS_KEY or GLM_KEY
        client = openai.OpenAI(api_key=key, base_url=exec_cfg["base_url"])
        use_openai = True
    
    sp = SOLO_PROMPT if solo else EXECUTOR_PROMPT
    tools = [t for t in TOOLS if not solo or t["function"]["name"] != "ask_advisor"]
    
    messages = [{"role":"system","content":sp},{"role":"user","content":prompt}]
    adv = ti = to = 0; tcs = {}
    
    for turn in range(max_turns):
        try:
            if use_openai:
                extra = {}
                resp = client.chat.completions.create(
                    model=exec_cfg["model"], messages=messages, tools=tools,
                    max_tokens=4096, temperature=0.2, **extra)
                u = resp.usage; ti += u.prompt_tokens; to += u.completion_tokens
                m = resp.choices[0].message
                content = m.content or ""
                reasoning = getattr(m, "reasoning_content", None)
                tc_list = []
                if m.tool_calls:
                    tc_list = [{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in m.tool_calls]
                finish = resp.choices[0].finish_reason
            else:
                resp = client.chat.completions.create(
                    model=exec_cfg["model"], messages=messages, tools=tools,
                    max_tokens=4096, temperature=0.2)
                u = resp.usage; ti += getattr(u,'input_tokens',0); to += getattr(u,'output_tokens',0)
                m = resp.choices[0].message
                content = m.content or ""
                reasoning = getattr(m, "reasoning_content", None)
                tc_list = m.tool_calls or []
                finish = resp.choices[0].finish_reason
        except Exception as e:
            print(f"      ❌ API: {e}")
            break
        
        am = {"role":"assistant","content":content}
        if reasoning: am["reasoning_content"] = reasoning
        if tc_list: am["tool_calls"] = tc_list
        messages.append(am)
        
        if not tc_list or finish == "stop":
            break
        
        for tc in tc_list:
            fn = tc["function"]["name"]
            tcs[fn] = tcs.get(fn, 0) + 1
            try: ta = json.loads(tc["function"]["arguments"])
            except: ta = {}
            if fn == "ask_advisor": adv += 1
            r = execute_tool(fn, ta, workdir, messages, advisor_cfg)
            preview = str(r)[:60].replace("\n"," ")
            print(f"      {fn} → {preview}...")
            messages.append({"role":"tool","tool_call_id":tc["id"],"content":str(r)})
    
    return {"advisor_calls":adv, "tool_calls":tcs, "tokens_in":ti, "tokens_out":to, "turns":turn+1}


def setup_workspace(inst, base):
    repo = inst.get("repo",""); bc = inst.get("base_commit",""); iid = inst["instance_id"]
    ws = os.path.join(base, iid.replace("/","__"))
    if os.path.exists(ws):
        try:
            subprocess.run(["git","checkout","--quiet",bc], check=True, timeout=30, capture_output=True, cwd=ws)
            subprocess.run(["git","clean","-fdq"], check=True, timeout=30, capture_output=True, cwd=ws)
        except: pass
        return ws
    os.makedirs(base, exist_ok=True)
    try:
        subprocess.run(["git","clone","--quiet",f"https://github.com/{repo}.git",ws], check=True, timeout=120, capture_output=True)
        subprocess.run(["git","checkout","--quiet",bc], check=True, timeout=30, capture_output=True, cwd=ws)
    except Exception as e:
        print(f"      Clone fail: {e}")
        os.makedirs(ws, exist_ok=True)
    return ws


def build_prompt(inst, solo=False):
    repo = inst.get("repo",""); issue = inst.get("problem_statement","")
    hints = inst.get("hints_text","")
    p = f"## Task: Fix bug in {repo}\n\n{issue}\n\n"
    if hints: p += f"## Hints\n{hints}\n\n"
    if solo:
        p += "## Instructions\n1. Explore codebase\n2. Identify root cause\n3. Make minimal edits\n4. Verify\n\nCodebase is in cwd. Start by listing files.\n"
    else:
        p += "## Instructions\n1. Explore codebase\n2. Identify root cause\n3. Make minimal edits\n4. Verify\n\nCodebase is in cwd. **IMPORTANT: Before making ANY edits, call ask_advisor FIRST to discuss your approach.** Then list files and start.\n"
    return p


def get_diff(ws):
    try:
        if not os.path.exists(os.path.join(ws,".git")): return ""
        r = subprocess.run(["git","diff"], capture_output=True, text=True, cwd=ws, timeout=30)
        return r.stdout
    except: return ""


def run_pair(exec_name, adv_name, instances, max_turns, output_dir):
    """Run solo + advisor evaluation for one model pair."""
    exec_cfg = MODELS[exec_name]
    adv_cfg = MODELS[adv_name]
    
    results = {}
    
    for mode, solo in [("solo", True), ("advisor", False)]:
        label = f"{exec_name}-{mode}"
        print(f"\n{'='*60}")
        print(f"{label}: executor={exec_cfg['model']} advisor={adv_cfg['model'] if not solo else 'none'}")
        print(f"{'='*60}")
        
        out = Path(output_dir) / label
        out.mkdir(parents=True, exist_ok=True)
        
        # Setup advisor config for hermes-agent
        os.environ["HERMES_ADVISOR_MODEL"] = adv_cfg["model"]
        os.environ["HERMES_ADVISOR_API_KEY"] = adv_cfg["api_key"] or ""
        
        cfg = load_advisor_config()
        # Pick the right advisor key based on base_url
        adv_key = adv_cfg["api_key"]
        if not adv_key:
            adv_base = adv_cfg.get("base_url", "")
            if "bigmodel" in adv_base:
                adv_key = GLM_KEY
            elif "deepseek" in adv_base:
                adv_key = DS_KEY
            else:
                adv_key = DS_KEY or GLM_KEY
        cfg.update(model=adv_cfg["model"], api_key=adv_key,
                    base_url=adv_cfg["base_url"], provider=None)
        if adv_cfg["api_mode"] == "anthropic":
            cfg["provider"] = "anthropic"
        
        batch_results = []
        for idx, inst in enumerate(instances):
            iid = inst["instance_id"]
            print(f"\n  [{idx+1}/{len(instances)}] {iid}")
            
            ws = setup_workspace(inst, base=f"/tmp/swe-{label}")
            prompt = build_prompt(inst, solo=solo)
            
            t0 = time.time()
            try:
                info = run_task(prompt, ws, max_turns, solo, exec_cfg, cfg)
            except Exception as e:
                import traceback; traceback.print_exc()
                info = {"error":str(e),"advisor_calls":0,"tool_calls":{},"turns":0,"tokens_in":0,"tokens_out":0}
            
            elapsed = time.time() - t0
            patch = get_diff(ws)
            
            r = {"instance_id":iid, "patch":patch, "adv":info.get("advisor_calls",0),
                 "turns":info.get("turns",0), "tin":info.get("tokens_in",0),
                 "tout":info.get("tokens_out",0), "elapsed":elapsed}
            batch_results.append(r)
            
            print(f"    {'✓' if patch else '✗'} adv={r['adv']} turns={r['turns']} {elapsed:.0f}s")
            
            # Save predictions
            with open(out/"predictions.jsonl","a") as f:
                f.write(json.dumps({"instance_id":iid,"model_name_or_path":label,"model_patch":patch},ensure_ascii=False)+"\n")
            
            # Running metrics
            pat = sum(1 for x in batch_results if x["patch"])
            with open(out/"metrics.json","w") as f:
                json.dump({"exec":exec_cfg['model'],"adv":adv_cfg['model'] if not solo else None,
                           "mode":mode,"total":len(instances),"completed":len(batch_results),
                           "patches":pat,"adv_calls":sum(x["adv"] for x in batch_results),
                           "tokens_in":sum(x["tin"] for x in batch_results)}, f, indent=2)
        
        results[mode] = batch_results
    
    # Summary
    s_pat = sum(1 for x in results["solo"] if x["patch"])
    a_pat = sum(1 for x in results["advisor"] if x["patch"])
    a_calls = sum(x["adv"] for x in results["advisor"])
    print(f"\n  {exec_name}: solo={s_pat}/{len(instances)} advisor={a_pat}/{len(instances)} (calls={a_calls})")
    return results


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set", default="/root/advisor-eval/eval_set_swe_hard6.json")
    p.add_argument("--pairs", default=",".join(f"{e}+{a}" for e,a in PAIRS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--output", default="/root/advisor-eval/swe_bench_results/multi")
    args = p.parse_args()
    
    with open(args.eval_set) as f:
        instances = json.load(f)
    if args.limit:
        instances = instances[:args.limit]
    
    # Parse requested pairs from --pairs argument
    requested = []
    for pair_str in args.pairs.split(","):
        parts = pair_str.strip().split("+")
        if len(parts) == 2:
            requested.append((parts[0], parts[1]))
    if not requested:
        requested = PAIRS  # fallback to all
    
    print(f"Benchmark: {len(instances)} instances, {len(requested)} model pairs")
    for e, a in requested:
        print(f"  {e} + {a}")
    
    all_results = {}
    for exec_name, adv_name in requested:
        exec_cfg = MODELS[exec_name]
        adv_cfg = MODELS[adv_name]
        
        # Skip pairs with no API key
        if not exec_cfg["api_key"] and exec_cfg["api_mode"] == "openai" and not DS_KEY and not GLM_KEY:
            print(f"\n⚠️  Skipping {exec_name}+{adv_name}: no API key")
            continue
        
        r = run_pair(exec_name, adv_name, instances, args.max_turns, args.output)
        all_results[f"{exec_name}+{adv_name}"] = r
    
    # Final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pair':<25} {'Solo':>8} {'Advisor':>8} {'Δ':>6} {'Calls':>6}")
    print("-"*55)
    for pair_name, r in all_results.items():
        s = sum(1 for x in r["solo"] if x["patch"])
        a = sum(1 for x in r["advisor"] if x["patch"])
        c = sum(x["adv"] for x in r["advisor"])
        d = a - s
        print(f"{pair_name:<25} {s}/{len(instances):>3} {a}/{len(instances):>3} {d:+>5} {c:>6}")
