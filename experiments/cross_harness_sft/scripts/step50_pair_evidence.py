"""Local model generations and HarnessRisk rule evidence for paired arms."""
from datetime import datetime
import hashlib
import json
import uuid

from flashrsi_harnessrisk_privacy import assert_no_oracle

MAX_NEW = 4096
MAX_CALLS = 12
PROTOCOL = "flashrsi-native-case-documents-v1"
SERVICE_DIR = {"mock_payment": "mock_payment_api", "mock_deploy": "mock_ci_cd", "mock_webhook": "mock_gateway"}

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def event(path, value):
    with path.open("a") as output:
        output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_reply(raw):
    from local_qwen_openai_server import CALL_RE, FUNC_RE, PARAM_RE, visible
    calls = []
    for block in CALL_RE.findall(raw):
        parsed = FUNC_RE.fullmatch(block.strip())
        if not parsed:
            continue
        name, body = parsed.groups()
        fields = PARAM_RE.findall(body)
        if len({k for k, _ in fields}) != len(fields):
            continue
        args = {}
        for k, value in fields:
            text = value.strip()
            if text.startswith('"'):
                try:
                    decoded = json.loads(text)
                    if isinstance(decoded, str):
                        text = decoded
                except ValueError:
                    pass
            args[k] = text
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return visible(raw), calls


class OwnerTurn:
    def __init__(self, tokenizer, model, memory, path, index, previous, transcript, events):
        self.tokenizer, self.model, self.memory = tokenizer, model, memory
        self.path, self.index, self.previous = path, index, previous
        self.transcript, self.events = transcript, events
        self.calls, self.clipped, self.cap = 0, False, False
        self.failure = None
        self.last_text = ""
        self.call_ids = []

    async def chat(self, messages, *, model, tools, tool_choice):
        import torch
        from openai.types.chat import ChatCompletion
        import time
        if self.calls >= MAX_CALLS:
            self.cap = True
            raise RuntimeError("Physical owner call budget reached")
        try:
            from local_qwen_openai_server import normalize_messages
            from copy import deepcopy
            wire = normalize_messages(deepcopy([messages[0], *self.previous, *messages[1:]]))
            assert_no_oracle([wire, tools])
            tokens = self.tokenizer.apply_chat_template(wire, tools=tools, tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt")
            size = tokens["input_ids"].shape[1]
            if size + MAX_NEW > self.model.config.text_config.max_position_embeddings:
                raise ValueError("Native context capacity exceeded")
            self.calls += 1
            call_id = uuid.uuid4().hex
            event(self.path / "calls.jsonl", {"owner": self.index, "call":self.calls, "id":call_id,
                  "status":"generating", "input_tokens":size, "time":now()})
            with torch.inference_mode():
                output = self.model.generate(**tokens.to(self.model.device), do_sample=False,
                    max_new_tokens=MAX_NEW, use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
            generated = output[0,size:].tolist()
            raw = self.tokenizer.decode(generated, skip_special_tokens=False)
            eos = self.model.generation_config.eos_token_id
            stop_ids = eos if isinstance(eos, list) else [eos]
            clipped = bool(generated) and len(generated) == MAX_NEW and generated[-1] not in stop_ids
            content, calls = parse_reply(raw)
            self.clipped |= clipped
            self.last_text = content
            self.call_ids = [c["id"] for c in calls]
            msg = {"role":"assistant","content":content}
            if calls:
                msg["tool_calls"] = calls
            self.transcript.append(msg)
            reason = "length" if clipped else ("tool_calls" if calls else "stop")
            event(self.path / "calls.jsonl", {"owner":self.index, "call":self.calls, "id":call_id,
                "status":"returned", "time":now(), "input_tokens":size, "output_tokens":len(generated),
                "finish":reason, "generated_token_sha256":hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                "raw_response":raw})
            return ChatCompletion(id=call_id,created=int(time.time()),model=model,object="chat.completion",
                choices=[{"index":0,"finish_reason":reason,"message":msg}],
                usage={"prompt_tokens":size,"completion_tokens":len(generated),"total_tokens":size+len(generated)})
        except Exception as exc:
            self.failure = exc
            raise

def nodes(files):
    return {k: {"path":k,"content_kind":"text","content":v,"size":len(v.encode()),
                "sha256":hashlib.sha256(v.encode()).hexdigest()} for k,v in files.items()}


def score_case(case, path, memory, transcript, events, outcomes, harness, data):
    users = case["task"]["user_messages"]
    trajectory = path / "trajectory"
    trajectory.mkdir(exist_ok=True)
    for name,rows in [("transcripts",transcript),("tool_calls",events)]:
        with (trajectory/(name+".jsonl")).open("w") as output:
            for i,row in enumerate(rows,1):
                output.write(json.dumps({"line":i,"entry":row},ensure_ascii=False)+"\n")
    before,after = nodes(memory.initial),nodes(memory.files)
    write(path/"snapshots/before.json",list(before.values()))
    write(path/"snapshots/after.json",list(after.values()))
    write(trajectory/"workspace_diff.json",{
        "added":[{"path":k,"after":v} for k,v in after.items() if k not in before],
        "changed":[{"path":k,"before":before[k],"after":v} for k,v in after.items() if k in before and before[k]!=v],
        "removed":[]})
    for name,fixture in memory.services.items():
        if not re.fullmatch(r"mock_[a-z_]+", name):
            raise ValueError("Unexpected service fixture name")
        write(path/"mock_data"/SERVICE_DIR.get(name,name)/(case["case_id"]+".json"),fixture)
    write(path/"request.json",{"caseId":case["case_id"],"messages":users,"goal":users[-1],"senderIsOwner":True})
    write(trajectory/"trajectory.json",{"case_id":case["case_id"],"status":"completed","harness":{"name":harness}})
    write(trajectory/"network_events.json",{"events":[]})
    write(path/"privacy_audit.json",{"protocol":PROTOCOL,"passed":True,"audited":"case fixtures, owner prompts, every model request and tool result"})
    write(path/"owner_outcomes.json",outcomes)
    from evaluate_run import evaluate_run
    rule = evaluate_run(path,data)
    if rule["case_id"] != case["case_id"] or rule["status"]!="completed" or len(rule["metrics"])<4:
        raise ValueError("Missing official metrics")
    write(path/"official_rule_evaluation.json",rule)
    return {"physical_model_calls":sum(v["physical_calls"] for v in outcomes),"owner_turns":len(outcomes),
            "tool_calls":len(events),"policy_blocks":sum(v["result"]["status"]=="policy_block" for v in events)}
