"""CABTO transfer experiments for four frozen scenes, with honest provenance.
Blocks transfers genuine Cover model schemas/programs; dual tasks use an explicit
reference-initialized LM proposal and new fine-grained program sampling. No hidden
whole pick/place wrappers. Dual control is oracle + declared weld assistance.
"""
import argparse,ast,hashlib,inspect,json,sys
from collections import deque
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;PACKAGE=ROOT.parent
sys.path.insert(0,str(PACKAGE/"CABTO/exp4_bt_tasks/stage3_cabto"))
from formal_bt import ModelLibrary,action_id,validate_model,symbolic_dry_run
from policy_codegen import PolicyProgram,extract_source
from core.cover_cabto_grounding import LocalModel,parse_json,save,digest
from core.cover_cabto_api import transition_check,API_DOCS
from core.remaining_blocks_api import FineBlocks,task_set,transferred_models

CONFIGS={"blocks":"blocks_rebuilt_v1.json","pour":"held_basin_pour_smooth_v2.json","handover":"handover_tea_box_v3.json","storage":"packing_dual_lift_shelf_storage_v3.json"}

def config_path(task):return ROOT/"scenes"/CONFIGS[task]

def make(task,path,seed=0,client=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);config=json.loads(config_path(task).read_text())
    if task=="blocks":return FineBlocks(config,seed,True,"qwen" if client else "oracle",path/"perception",client)
    if task=="handover":
        from core.remaining_handover_api import factory
        return factory(config,path,seed)
    from core.remaining_dual_api import factory
    return factory(task,config,path,seed)

def refs(task):
    if task=="blocks":return transferred_models(PACKAGE)[0]
    if task=="handover":
        from core.remaining_handover_api import reference_models
        return reference_models()
    from core.remaining_dual_api import reference_models
    return reference_models(task)

def docs(task,m):
    if task=="blocks":return API_DOCS[m["name"]]
    if task=="handover":
        from core.remaining_handover_api import DOCS
        return DOCS[m["program_kind"]]
    from core.remaining_dual_api import docs_for
    return docs_for(task,m)

def sources_snapshot(out):
    # All project/core source plus imported underlying stage2 and formal modules;
    # snapshots are immutable and verified after each run.
    files=list((ROOT/"core").glob("*.py"))+[Path(__file__)]+list(ROOT.glob("probe_*.py"))
    files+=list((PACKAGE/"CABTO/exp4_bt_tasks/stage2_scripted").glob("*.py"))
    files+=list((PACKAGE/"CABTO/exp4_bt_tasks/stage3_cabto").glob("*.py"))
    hashes={str(p):digest(p) for p in files}
    p=Path(out)/"source_snapshot";p.mkdir(parents=True,exist_ok=True)
    for f in files:
        dst=p/f.relative_to(PACKAGE);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes())
    return hashes

def plancheck(models,tasks,out=None):
    for m in models:validate_model(m)
    result=[]
    for task in tasks:
        s=set(task["initial"]);g=set(task["goal"])
        try:
            bt=ModelLibrary(models).build(s,g);r=symbolic_dry_run(bt,s,g,100)
            if out:
                p=Path(out)/task["id"];p.mkdir(parents=True,exist_ok=True);(p/"tree.json").write_text(bt.export());(p/"tree.dot").write_text(bt.export("dot"))
            result.append({"task":task["id"],**r})
        except Exception as exc:result.append({"task":task["id"],"reached_goal":False,"error":str(exc)})
    return {"complete_for_declared_P":all(r["reached_goal"] for r in result),"tasks":result,"scope":"finite_P_symbolic_only"}

def prepare(args):
    hashes=sources_snapshot(args.output);runtime=make(args.task,args.output/"initial")
    try:
        initial=sorted(runtime.state());goal=sorted(runtime.goal);runtime.env.save_png(str(args.output/"before.png"))
        tasks=task_set() if args.task=="blocks" else [{"id":args.task.upper(),"initial":initial,"goal":goal}]
        library=refs(args.task)
        provenance={"model":"reference", "mode":"explicit_reference_initialization_not_unconstrained_discovery"}
        if args.task=="blocks":
            _,origin=transferred_models(PACKAGE)
            provenance={"model":"mlx-community/Qwen3.5-9B-4bit","mode":"unmodified_model_schema_transfer_cover_to_blocks","source":str(origin),"source_sha256":digest(origin)}
        else:
            client=LocalModel(args.model);proposed=[]
            for j,m in enumerate(library):
                prompt=("Review this single STRIPS action for a robotic task and propose its pre/add/del. It is initialized from an engineer's existing reference, NOT a blank action discovery. "
                    "Return ONLY JSON with pre,add,del lists. Preserve the named conditions and safety requirements when they are consistent; do not add unsupported predicates. "
                    "pre/add and add/del must be disjoint. Do not give control code.\nReference action: "+json.dumps(m,ensure_ascii=False)+"\nTask goal: "+json.dumps(goal))
                record={"reference":m,"mode":"reference_initialized_local_model_proposal"}
                for attempt in range(args.attempts):
                    call=client.call(prompt,max_tokens=850,temperature=.1*attempt);record["call"]=call
                    try:
                        answer=parse_json(call["raw"]);new={**m,**{k:answer[k] for k in ("pre","add","del")}};validate_model(new)
                        allowed=set().union(*(set(a[k]) for a in library for k in ("pre","add","del")))
                        if set().union(*(set(new[k]) for k in ("pre","add","del")))-allowed:raise ValueError("Unknown CP predicate")
                        record.update(model=new,accepted=True);save(args.output/f"proposal_{j:02d}_{attempt}.json",record);break
                    except Exception as exc:
                        record.update(accepted=False,error=str(exc));save(args.output/f"proposal_{j:02d}_{attempt}.json",record)
                        prompt+="\nYour raw answer: "+call["raw"]+"\nValidation error: "+str(exc)
                else:raise RuntimeError("Proposal failed; no host effects patch")
                proposed.append(new);print("PROPOSE",args.task,j,m["name"],flush=True)
            library=proposed;provenance={"model":args.model,"mode":"reference_initialized_local_model_proposal","reference_source":"existing validated engineering domain supplied verbatim as warm start"}
        check=plancheck(library,tasks,args.output/"planning")
        save(args.output/"models.json",{"task":args.task,"models":library,"P":tasks,"initial":initial,"goal":goal,"check":check,"provenance":provenance,
             "config_sha256":digest(config_path(args.task)),"assist":"strict_no_weld" if args.task=="blocks" else "declared_grasp_weld_assisted",
             "perception_scope":"blocks_vision_stage_available; dual_policy_sampling_uses_oracle_geometry","source_hashes":hashes})
        if not check["complete_for_declared_P"]:raise RuntimeError("Proposed library incomplete")
    finally:runtime.close()

def execute(runtime,task,m,source,path):
    path.mkdir(parents=True,exist_ok=True);before=runtime.state();exc=None;trace=[];executed=False
    runtime.env.save_png(str(path/"before.png"));(path/"policy.py").write_text(source)
    try:
        if not set(m["pre"])<=before:raise ValueError("Physical precondition does not hold")
        api=runtime.api(m);program=PolicyProgram(source,docs(task,m));program.validate_api(api);executed=True
        trace=program.run(api,call_limit=60)["trace"]
    except Exception as error:exc=f"{type(error).__name__}: {error}";trace=getattr(error,"policy_trace",[])
    after=runtime.state();check=transition_check(m,before,after,exc)
    # Original dual observers expose additional overlapping status predicates.
    # Check declared effects and frame drift separately; do NOT fake Eq1 evidence.
    contract_ok=not any(check["violations"][k] for k in ("missing_pre","missing_add","remaining_del","missing_preserved_pre"))
    safety=True
    if hasattr(runtime,"collision_free"):safety=runtime.collision_free()
    if task=="blocks":safety=not any(r["min_distance_m"]<-.0003 for r in runtime.r.contacts.values())
    ok=bool(contract_ok and exc is None and executed and safety)
    if task=="blocks":ok=ok and check["strict_CP_transition_match"]
    result={**check,"effect_ok":ok,"code_executed":executed,"exception":exc,"trace":trace,"before":sorted(before),"after":sorted(after),"model":m,
            "safety_ok":bool(safety),"contract_effect_ok":contract_ok,"code_sha256":hashlib.sha256(source.encode()).hexdigest(),
            "consistency_scope":"full_CP" if task=="blocks" else "declared_pre_add_del_and_independent_final_goal; extra_observer_frame_drift_reported"}
    save(path/"feedback.json",result);runtime.env.save_png(str(path/"after.png"));return result

def sample(args):
    data=json.loads(args.models.read_text());models=data["models"];accepted={};attempts=[];out=args.output;hashes=sources_snapshot(out)
    previous=json.loads(args.resume.read_text()) if args.resume else None
    if previous:accepted=deepcopy(previous["accepted"]);attempts=deepcopy(previous["attempts"])
    # Only genuine generated Cover programs are transferred, never session candidates.
    cover=json.loads((PACKAGE/"outputs/cover_cabto_fix/sampling_9b_complete/policies.json").read_text()) if args.task=="blocks" else None
    client=LocalModel(args.model) if args.task!="blocks" else None
    runtime=make(args.task,out/"episode");prefix=[];goal=set(data["goal"]);tree=ModelLibrary(models).build(runtime.state(),goal)
    try:
        for tick in range(40):
            status,m=tree.tick(runtime.state())
            if status!="running":break
            key=action_id(m);kind=m.get("program_kind",m["name"]);feedback=[]
            if key in accepted:
                rec=execute(runtime,args.task,m,accepted[key]["source"],out/f"prefix_{tick:02d}")
                if not rec["effect_ok"]:raise RuntimeError("Previously accepted program failed live prefix")
                prefix.append((m,accepted[key]["source"]));continue
            transferred=next((v for v in accepted.values() if v["model"].get("program_kind")==kind),None)
            if cover:transferred=next(v for v in cover["accepted"].values() if v["model"]["name"]==kind)
            for trial in range(args.attempts):
                p=out/f"action_{tick:02d}"/f"attempt{trial+1}";p.mkdir(parents=True,exist_ok=True)
                row={"action_id":key,"model":m,"attempt":trial+1}
                image=p/"context.png";runtime.env.save_png(str(image))
                source=None
                try:
                    if trial==0 and transferred:
                        source=transferred["source"];row["transfer"]={"source_sha256":transferred["sha256"],"source_attempt":transferred["source_attempt"],"mode":"unmodified_actual_model_source_schema_transfer","new_call":False}
                    else:
                        if client is None:client=LocalModel(args.model)
                        prompt=("Write ONLY def policy(api): implementing ONE action, not the whole task. No imports, loops, tuple unpacking, simulator access or success flags. "
                            "Use local coordinates read from API. All required movement, closing/opening, attachment/detachment and withdrawal must be EXPLICIT. "
                            "Do not change the STRIPS model; do not return a success claim.\nAction model: "+json.dumps(m,ensure_ascii=False)
                            +"\nOnly available API signatures (not a required sequence):\n"+"\n".join(docs(args.task,m)[k] for k in sorted(docs(args.task,m)))
                            +"\nActual previous failure: "+json.dumps(feedback[-1:],ensure_ascii=False))
                        call=client.call(prompt,[image],max_tokens=1100,temperature=.15*trial);row["call"]=call;source=extract_source(call["raw"])
                    row["source"]=source;rec=execute(runtime,args.task,m,source,p);row["evaluation"]=rec;row["accepted"]=rec["effect_ok"]
                except Exception as exc:row.update(accepted=False,error=str(exc))
                attempts.append(row);save(p/"attempt.json",row)
                print("SAMPLE",args.task,tick,m["name"],trial+1,row["accepted"],row.get("error") or row.get("evaluation",{}).get("exception"),flush=True)
                if row["accepted"]:
                    accepted[key]={"source":source,"sha256":hashlib.sha256(source.encode()).hexdigest(),"model":m,"source_attempt":str(p)};prefix.append((m,source));break
                feedback.append({"source":source or row.get("call",{}).get("raw"),"error":row.get("error"),"observations":row.get("evaluation")})
                # Reset to identical physical start and replay only accepted actual programs.
                runtime.close();runtime=make(args.task,out/f"reset_{tick:02d}_{trial+1}")
                for k,(pm,ps) in enumerate(prefix):
                    pr=execute(runtime,args.task,pm,ps,out/f"reset_{tick:02d}_{trial+1}"/f"prefix{k}")
                    if not pr["effect_ok"]:raise RuntimeError("Accepted precondition prefix failed after reset")
            else:raise RuntimeError("Policy budget exhausted")
        if not goal<=runtime.state():raise RuntimeError("Final task state not reached")
        stable=runtime.stable_goal();status,_=tree.tick(runtime.state())
        success=stable and status=="success"
    except Exception as exc:
        success=False;failure=f"{type(exc).__name__}: {exc}";print("BLOCKED",args.task,failure,flush=True)
    finally:
        result={"task":args.task,"model_id":args.model,"accepted":accepted,"attempts":attempts,"success":success,"error":locals().get("failure"),
                "config_sha256":digest(config_path(args.task)),"model_input_sha256":digest(args.models),"model_source":str(args.models),
                "source_hashes":hashes,"sources_unchanged":all(digest(p)==h for p,h in hashes.items()),
                "final_state":sorted(runtime.state()),"perception":"oracle","assist":data["assist"],"task_success_is_not_all_schema_validation":True}
        # One task may not traverse all library actions; accept final only when all needed grounded actions exercised.
        result["unvalidated_models"]=[action_id(m) for m in models if action_id(m) not in accepted]
        result["complete_validated_library"]=plancheck([v["model"] for v in accepted.values()],data["P"])
        save(out/"policies.json",result);runtime.save(out);runtime.close()
    if not success:raise SystemExit(2)

def replay(args):
    data=json.loads(args.models.read_text());cache=json.loads(args.policies.read_text());out=args.output;hashes=sources_snapshot(out)
    assert cache["config_sha256"]==data["config_sha256"]==digest(config_path(args.task))
    accepted=cache["accepted"];models=[m for m in data["models"] if action_id(m) in accepted]
    check=plancheck(models,data["P"],out/"planning");assert check["complete_for_declared_P"]
    for v in accepted.values():assert hashlib.sha256(v["source"].encode()).hexdigest()==v["sha256"]
    client=LocalModel("mlx-community/Qwen2.5-VL-3B-Instruct-4bit") if args.task=="blocks" and args.vision else None
    results=[]
    for t in data["P"]:
        p=out/t["id"];p.mkdir(parents=True,exist_ok=True);runtime=make(args.task,p,args.seed,client);runtime.goal=set(t["goal"])
        bt=ModelLibrary(models).build(runtime.state(),runtime.goal);(p/"tree.json").write_text(bt.export());(p/"tree.dot").write_text(bt.export("dot"))
        episode={"task":t,"seed":args.seed,"steps":[],"success":False}
        try:
            for i in range(40):
                status,m=bt.tick(runtime.state())
                if status!="running":break
                rec=execute(runtime,args.task,m,accepted[action_id(m)]["source"],p/f"step{i:02d}");episode["steps"].append(rec)
                print("REPLAY",args.task,t["id"],args.seed,i,m["name"],rec["effect_ok"],rec["exception"],flush=True)
                if not rec["effect_ok"]:raise RuntimeError("Independent action contract failed")
            episode["success"]=bool(runtime.goal<=runtime.state() and runtime.stable_goal())
            episode["final_state"]=sorted(runtime.state())
        except Exception as exc:episode["error"]=str(exc)
        finally:
            runtime.save(p)
            episode["final_state"]=sorted(runtime.state());episode["full_frame_all_match"]=all(s["strict_CP_transition_match"] for s in episode["steps"])
            episode["assist"]=data["assist"];episode["perception"]="Qwen_RGB" if client else "oracle"
            if hasattr(runtime,"events"):save(p/"events.json",runtime.events)
            elif hasattr(runtime,"scout"):save(p/"events.json",runtime.scout.events)
            if args.task=="handover":episode["geometry"]=runtime.placement_geometry();episode["handover_verified"]=runtime.handover_verified();episode["collision_free"]=runtime.collision_free()
            if args.task=="pour":episode["source_return"]=runtime.return_geometry("canL");episode["basin_return"]=runtime.return_geometry("basin");episode["collision_free"]=runtime.collision_free()
            if args.task=="storage":episode["goal_now"]=runtime.scout.goal_now()
            save(p/"result.json",episode);results.append(episode);runtime.close()
    result={"task":args.task,"episodes":results,"success_count":sum(r["success"] for r in results),"total":len(results),"complete":check,
            "model_input":str(args.models),"model_sha256":digest(args.models),"policies_input":str(args.policies),"policies_sha256":digest(args.policies),
            "config_sha256":digest(config_path(args.task)),"source_hashes":hashes,"sources_unchanged":all(digest(p)==h for p,h in hashes.items())}
    save(out/"result.json",result)
    if not all(r["success"] for r in results):raise SystemExit(2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=["prepare","sample","replay"]);ap.add_argument("task",choices=CONFIGS)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--models",type=Path);ap.add_argument("--policies",type=Path);ap.add_argument("--resume",type=Path)
    ap.add_argument("--model",default="mlx-community/Qwen3.5-9B-4bit");ap.add_argument("--attempts",type=int,default=4);ap.add_argument("--seed",type=int,default=0);ap.add_argument("--vision",action="store_true")
    a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):ap.error("Use a new empty output to preserve evidence")
    a.output.mkdir(parents=True,exist_ok=True)
    {"prepare":prepare,"sample":sample,"replay":replay}[a.phase](a)

if __name__=="__main__":main()
