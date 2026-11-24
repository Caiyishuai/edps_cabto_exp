"""Run a bounded CABTO method reproduction on the frozen Cover scene.
Commands: propose | sample | replay | perception. Originals are never overwritten.
"""
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from core.cover_cabto_grounding import (CONFIG, CP, INITIAL, TASKS, PAIRS, LocalModel, ModelLibrary,
    PolicyProgram, extract_source, action_id, propose, check_models, save, digest)
from core.cover_cabto_api import FineCover, API_DOCS, transition_check, project, intersect_plane


def keyfile(model):
    return model["name"] + ("_" + model["args"]["object"] if model["args"] else "")


def path_to(models, goal, start=INITIAL):
    queue = deque([(frozenset(start), [])]); seen = {frozenset(start)}
    while queue:
        state, path = queue.popleft()
        if set(goal) <= state: return path
        for m in models:
            if set(m["pre"]) <= state:
                nxt = frozenset((set(state)|set(m["add"]))-set(m["del"]))
                if nxt not in seen:
                    seen.add(nxt); queue.append((nxt, path+[m]))
    return None


def execute(fine, model, source):
    before = fine.state(); trace = []; error = None; executed = False
    valid = set(model["pre"]) <= before
    try:
        if not valid: raise RuntimeError("Precondition unsatisfied; invalid trial")
        api = fine.bind(model)
        program = PolicyProgram(source, API_DOCS[model["name"]])
        program.validate_api(api); executed = True
        trace = program.run(api)["trace"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; trace = getattr(exc,"policy_trace",[])
    after = fine.state()
    check = transition_check(model,before,after,error)
    unsafe = [v for v in fine.r.contacts.values() if v["min_distance_m"] < -.0003]
    check["effect_ok"] = check["effect_ok"] and not unsafe
    return {**check,"before":sorted(before),"after":sorted(after),"model":model,
            "code_executed":executed,"exception":error,"trace":trace,"forbidden_contacts":unsafe,
            "code_sha256":hashlib.sha256(source.encode()).hexdigest()}


def snapshot(out):
    from core import cover_cabto_api, cover_cabto_grounding, single_arm_rebuilt
    sources = [Path(__file__),Path(cover_cabto_api.__file__),Path(cover_cabto_grounding.__file__),Path(single_arm_rebuilt.__file__),CONFIG]
    out = Path(out); (out/"source_snapshot").mkdir(parents=True,exist_ok=True)
    for p in sources: shutil.copy2(p,out/"source_snapshot"/p.name)
    return {str(p):digest(p) for p in sources}


def sample(args):
    out = args.output; hashes = snapshot(out)
    proposal = json.loads(args.models.read_text()); models = proposal["models"]
    if proposal["config_sha256"] != digest(CONFIG): raise RuntimeError("Frozen config changed")
    accepted = {}; results = []; pending = models[:]
    external = json.loads(args.candidates.read_text()) if args.candidates else None
    client = None if external else LocalModel(args.model)
    config = json.loads(CONFIG.read_text())
    all_status = {"phase":"policy_sampling","model_id":external["origin"] if external else args.model,
                  "generation_channel":external.get("generation_channel") if external else "local_live_model",
                  "external_candidate_artifact_sha256":digest(args.candidates) if external else None,
                  "config_sha256":digest(CONFIG),"accepted":accepted,
                  "attempts":results,"success":False,"source_hashes":hashes,"perception":"oracle",
                  "notes":["Each accepted h has generated finite API source; no pick/place macro and no no-op adapter", "oracle localization baseline; Qwen is a separate perception experiment", "preconditions initialized by reset plus real accepted-policy prefix, not object pose teleportation"]}
    try:
        while pending:
            established = [m for m in models if action_id(m) in accepted]
            selected = next(((m,path_to(established,m["pre"])) for m in pending if path_to(established,m["pre"]) is not None),None)
            if selected is None:
                all_status["blocked"] = "No valid physical precondition prefix using accepted actions"; break
            h, prelude = selected
            action_dir = out/keyfile(h); action_dir.mkdir(parents=True,exist_ok=True)
            previous = []
            for i in range(args.attempts):
                path = action_dir/f"attempt{i+1}"; path.mkdir(parents=True,exist_ok=True)
                fine = FineCover(config,seed=0,render=True,perception="oracle",out=path)
                before_image = path/"before.png"; before_image_saved = False
                row = {"action":action_id(h),"attempt":i+1,"model":h,"prefix":[]}
                try:
                    for m in prelude:
                        rec = execute(fine,m,accepted[action_id(m)]["source"]); row["prefix"].append(rec)
                        if not rec["effect_ok"]: raise RuntimeError("Accepted prefix failed; do not blame candidate")
                    fine.r.env.save_png(str(before_image),"overview"); before_image_saved = True
                    prompt = {"instruction":"Generate ONLY a Python def policy(api) implementing this ONE grounded STRIPS action, using real fine-grained API calls. No imports, loops, other functions, simulator objects or success flags. No hardcoded xyz. The return value is not the success criterion. All motion must be explicitly called. Do not change the model.",
                              "action_model":h,"observed_before":sorted(fine.state()),"APIs":API_DOCS[h["name"]],
                              "scene":config["specification"], "object_roles":h["args"],
                              "previous_failed_programs_and_execution_feedback":previous}
                    if external:
                        source = external["programs"][keyfile(h)]
                        row["candidate_origin"] = {"channel":external["generation_channel"],"artifact":str(args.candidates),
                                                   "sha256":digest(args.candidates),"new_model_call_in_this_process":False}
                        row["validation_context"] = prompt
                    else:
                        call = client.call(json.dumps(prompt,ensure_ascii=False),[before_image],max_tokens=850,temperature=min(i*.2,.6))
                        row["call"] = call
                        source = extract_source(call["raw"])
                    row["source"] = source
                    (path/"policy.py").write_text(source)
                    row["evaluation"] = execute(fine,h,source)
                    row["accepted"] = all(row["evaluation"][k] for k in ("valid_trial","code_executed","goal_ok","effect_ok"))
                except Exception as exc:
                    row.update(accepted=False,error=f"{type(exc).__name__}: {exc}")
                finally:
                    fine.r.env.save_png(str(path/"after.png"),"overview")
                    save(path/"perception.json",fine.perception.records)
                    save(path/"events.json",fine.r.events)
                    row["simulation_seconds"] = float(fine.r.env.d.time)
                    fine.close()
                save(path/"feedback.json",row); results.append(row)
                print("POLICY",keyfile(h),i+1,row["accepted"],row.get("error") or row.get("evaluation",{}).get("exception"),flush=True)
                if row["accepted"]:
                    accepted[action_id(h)]={"source":source,"sha256":hashlib.sha256(source.encode()).hexdigest(),"source_attempt":str(path),"model":h}
                    (action_dir/"accepted_policy.py").write_text(source)
                    pending.remove(h); save(out/"policies.json",all_status); break
                ev = row.get("evaluation",{})
                previous.append({"generated_code":row.get("source",row.get("call",{}).get("raw")),
                                 "error":row.get("error") or ev.get("exception"),"violations":ev.get("violations"),"api_trace":ev.get("trace"),"forbidden_contacts":ev.get("forbidden_contacts"),
                                 "instruction":"Repair the candidate CODE, keeping the immutable action model and API contracts."})
            else:
                # A bounded failure is not a proof of impossibility. Expose both
                # planning role and execution evidence for optional cross-level work.
                all_status["blocked"] = {"h":h,"status":"policy_budget_exhausted_unresolved_not_proven_inconsistent", "evidence":previous}
                save(out/"cross_level_context.json", {"failed_model":h,"planning":check_models(models),"execution":previous,
                     "status":"not automatically patching h for a code or perception failure"})
                break
        all_status["success"]=len(accepted)==len(models)
        if accepted: all_status["grounded_library_check"]=check_models([a["model"] for a in accepted.values()],out/"grounded_trees")
    finally:
        all_status["sources_unchanged"]=all(digest(p)==h for p,h in hashes.items())
        save(out/"policies.json",all_status)
    if not all_status["success"]: raise SystemExit(2)


def replay(args):
    out=args.output; hashes=snapshot(out)
    proposals=json.loads(args.models.read_text()); models=proposals["models"]
    policy_file=json.loads(args.policies.read_text()); policy=policy_file["accepted"]
    if proposals["config_sha256"] != digest(CONFIG) or policy_file["config_sha256"] != digest(CONFIG):
        raise RuntimeError("Proposal/policy belongs to a different frozen configuration")
    for key, entry in policy.items():
        if entry["sha256"] != hashlib.sha256(entry["source"].encode()).hexdigest():
            raise RuntimeError(f"Policy source hash mismatch: {key}")
    validated=[m for m in models if action_id(m) in policy]
    if any(policy[action_id(m)]["model"] != m for m in validated):
        raise RuntimeError("Policy contract differs from the currently proposed model")
    complete=check_models(validated,out/"symbolic_trees")
    if not complete["complete_for_P"]: raise RuntimeError("Validated library incomplete on P")
    config=json.loads(CONFIG.read_text())
    client=LocalModel(args.model) if args.perception!="oracle" else None
    episodes=[]
    for task in TASKS:
        if args.task and task["id"] != args.task: continue
        path=out/task["id"];path.mkdir(parents=True,exist_ok=True)
        fine=FineCover(config,args.seed,render=True,perception=args.perception,out=path/"perception",client=client)
        goal=set(task["goal"]);tree=ModelLibrary(validated).build(fine.state(),goal)
        (path/"tree.json").write_text(tree.export());(path/"tree.dot").write_text(tree.export("dot"))
        (path/"scene.xml").write_text(fine.r.xml)
        fine.r.env.save_png(str(path/"before.png"),"overview")
        episode={"task":task,"seed":args.seed,"perception":args.perception,"config_sha256":digest(CONFIG),"steps":[],"success":False}
        try:
            if args.perception == "qwen_zoom":
                needed = [s for o,s in PAIRS if f"at_target({o})" in goal]
                fine.perception.prepare_static_supports(needed)
                episode["initial_static_support_cache"] = {k:v.tolist() for k,v in fine.perception.static_cache.items()}
            for k in range(24):
                status,h=tree.tick(fine.state())
                if status!="running": break
                rec=execute(fine,h,policy[action_id(h)]["source"]);episode["steps"].append(rec)
                save(path/f"step{k:02d}.json",rec)
                fine.r.env.save_png(str(path/f"step{k:02d}.png"),"overview")
                print("REPLAY",args.perception,task["id"],k,action_id(h),rec["effect_ok"],rec["exception"],flush=True)
                if not rec["effect_ok"]: raise RuntimeError("Action failed independent transition or safety check")
            fine.r.mark("final_stability")
            for _ in range(math_ceil(config["control"]["stable_seconds"]/(fine.r.dt*fine.r.steps))):
                fine.r.wait(fine.r.dt*fine.r.steps,1.)
                fine.r.stability_ok=fine.r.stability_ok and goal<=fine.state()
            episode.update(fine.final_check(goal))
            status,_=tree.tick(fine.state());episode["bt_status"]=status
            episode["success"]=episode["success"] and status=="success" and all(x["effect_ok"] for x in episode["steps"])
        except Exception as exc:
            episode.update(exception=f"{type(exc).__name__}: {exc}",success=False)
            episode["observed_state"]=sorted(fine.state())
            episode["metrics"]=fine.r.assess()
        finally:
            fine.r.env.save_png(str(path/"final.png"),"overview")
            fine.r.env.save_video(str(path/"rollout.mp4"),fps=30)
            episode["simulation_seconds"]=float(fine.r.env.d.time)
            episode["weld_active"]=fine.r.active_welds()
            episode["perception_records"]=fine.perception.records
            save(path/"events.json",fine.r.events)
            save(path/"result.json",episode);episodes.append(episode);fine.close()
    result={"complete_for_declared_P":complete,"episodes":episodes,"success_count":sum(e["success"] for e in episodes),
            "total":len(episodes),"source_hashes":hashes,"sources_unchanged":all(digest(p)==h for p,h in hashes.items()),
            "mode":"finite_empirical_consistency_only","Qwen_fallback_to_oracle":False,
            "proposal_input_sha256":digest(args.models),"policies_input_sha256":digest(args.policies),
            "proposal_origin":proposals.get("proposal_model"),"policy_origin":policy_file.get("model_id"),
            "policy_generation_channel":policy_file.get("generation_channel"),
            "Qwen_geometry_prior":"known support surface z; oracle initial payload z plus grasp offset",
            "control_feedback":"oracle payload tracking in move_object; proprioceptive IK",
            "observation_protocol":"pre-observed static support cache plus model-centered crops" if args.perception=="qwen_zoom" else "two live cameras, no static cache"}
    save(out/"result.json",result)
    if not all(e["success"] for e in episodes): raise SystemExit(2)


def math_ceil(x):
    import math
    return math.ceil(x)


def perception(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=json.loads(CONFIG.read_text());client=LocalModel(args.model)
    fine=FineCover(config,args.seed,render=True,perception="qwen",out=out,client=client)
    calibration=[];results=[]
    try:
        # Algebraic GT calibration has no VLM role and no hidden scene mutation.
        for name,z in [(o,float(fine.r.initial[o][2])+config["grip_offset_z"][o]) for o,_ in PAIRS]:
            for cam in ("overview","front"):
                size=fine.r.env.render_cam(cam).shape[0]
                gt=np.r_[fine.r.pos(name)[:2],z];uv=project(fine.r.env,gt,cam,size)
                back=intersect_plane(fine.r.env,uv,cam,size,z)
                calibration.append({"name":name,"cam":cam,"roundtrip_error_m":float(np.linalg.norm(back-gt))})
        for name,z,kind in ([(o,float(fine.r.initial[o][2])+config["grip_offset_z"][o],"grasp_center") for o,_ in PAIRS]+
                            [(s,config["table_top"]+config["support_surface_z"][s],"support_floor_center") for _,s in PAIRS]):
            try:
                p=fine.perception.locate(name,z,kind);results.append({"name":name,"accepted":True,"xyz":p.tolist()})
            except Exception as exc:results.append({"name":name,"accepted":False,"error":str(exc)})
            print("POINT",results[-1],flush=True)
        save(out/"result.json",{"model_id":args.model,"calibration":calibration,"points":results,"records":fine.perception.records,
             "no_robot_motion":True,"no_fallback":True,"known_z_plane_prior":True,"coordinate_scale":"absolute pixels for Qwen2.5-VL; explicit0..1000 for Qwen3.5"})
    finally:fine.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=["propose","sample","replay","perception"])
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--models",type=Path)
    ap.add_argument("--policies",type=Path)
    ap.add_argument("--candidates",type=Path,help="Explicitly sourced external/session model candidates, never relabeled as a local model call")
    ap.add_argument("--model",default="mlx-community/Qwen3.5-0.8B-4bit")
    ap.add_argument("--attempts",type=int,default=3)
    ap.add_argument("--perception",choices=["oracle","qwen","qwen_zoom"],default="oracle")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--task",choices=["C1","C2","C3"])
    a=ap.parse_args();a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):
        ap.error("Output must be new/empty: preserve previous attempts and evidence")
    a.output.mkdir(parents=True,exist_ok=True)
    if a.phase=="propose":propose(a.output,a.model,a.attempts)
    elif a.phase=="sample":sample(a)
    elif a.phase=="replay":replay(a)
    else:perception(a)


if __name__=="__main__":main()
