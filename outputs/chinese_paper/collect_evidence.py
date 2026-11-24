"""Extract paper tables and figure assets from frozen experiment records.
No model calls, simulation, policy modification or result rewriting.
Run from this directory; optional argument points to edps_cabto_exp.
"""
from pathlib import Path
import hashlib,json,shutil,sys

HERE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[1]
OUT=ROOT/"outputs";ASSETS=HERE/"assets";ASSETS.mkdir(exist_ok=True)
SAMPLES={"pour":"sampling_v5","handover":"sampling_v3","storage":"sampling_v2"}
manifest=[]

def read(p):
    p=Path(p);b=p.read_bytes();manifest.append({"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(b).hexdigest()});return json.loads(b)
def cp(src,name):
    src=Path(src);dst=ASSETS/name;shutil.copy2(src,dst)
    manifest.append({"path":str(src.relative_to(ROOT)),"sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"paper_asset":"assets/"+name})

scenes=[];all_episodes=[];patches=[];witnesses=[]
for task in ("cover","blocks","pour","handover","storage"):
    if task=="cover":
        base=OUT/"cover_cabto_fix";runs=[read(base/b/"result.json") for b in ("final_visual","final_holdout1","final_holdout2")]
        model=read(base/"proposal_final_v2/models.json");sample=read(base/"sampling_9b_complete/policies.json")
    else:
        base=OUT/"remaining_cabto"/task;runs=[read(base/f"final_seed{s}/result.json") for s in (0,1)]
        model=read(base/("proposal" if task=="blocks" else "refinement")/"models.json")
        sample=read(base/("sampling" if task=="blocks" else SAMPLES[task])/"policies.json")
    ep=[e for d in runs for e in d["episodes"]];steps=[s for e in ep for s in e["steps"]]
    assert all(e["success"] for e in ep) and all(s["effect_ok"] and s["strict_CP_transition_match"] for s in steps)
    sr={"scene":task,"goals":len(model["P"]),"grounded_actions":len(model["models"]),"episodes":len(ep),"successful":sum(e["success"] for e in ep),
        "transitions":len(steps),"full_matches":sum(s["strict_CP_transition_match"] for s in steps),"step_lengths":[len(e["steps"]) for e in ep],
        "accepted_policies":len(sample["accepted"]),"retained_sampling_attempts":len(sample["attempts"]),"perception":"VLM_RGB_with_priors" if task in ("cover","blocks") else "oracle",
        "grasp":"strict" if task in ("cover","blocks") else "weld_assisted"}
    scenes.append(sr);all_episodes+=ep
    if task in SAMPLES:
        for key,item in sample["accepted"].items():
            row=next(a for a in reversed(sample["attempts"]) if a["accepted"] and a["action_id"]==key)
            w=row["evaluation"];witnesses.append({"scene":task,"action_id":key,"old_full_match":w["strict_CP_transition_match"],"old_declared_effects_pass":w["effect_ok"],"violations":w["violations"]})
        for p in sorted((base/"refinement").glob("patch*.json")):
            d=read(p)
            if d["accepted"]:patches.append({"scene":task,"action_id":d["action_id"],"patch":d["patch"],"model":d["model"],"before":d["witness"]["before"],"after":d["witness"]["after"]})
    if task=="cover":imagebase=base/"final_visual/C3";before=imagebase/"before.png"
    else:imagebase=base/"final_seed0"/("B3" if task=="blocks" else task.upper());before=imagebase/"step00/before.png"
    cp(before,f"{task}_initial.png");cp(imagebase/"final.png",f"{task}_final.png")

witness_summary=[]
for task in SAMPLES:
    w=[x for x in witnesses if x["scene"]==task]
    witness_summary.append({"scene":task,"witnesses":len(w),"old_full_matches":sum(x["old_full_match"] for x in w),"patched_models":sum(x["scene"]==task for x in patches),"postrepair_final_matches":next(x["transitions"] for x in scenes if x["scene"]==task)})

camera=read(OUT/"unified_cameras/verification.json")
points=read(OUT/"unified_cameras/pointing_v3/results.json")["point_results"]
point_rows=[]
for p in points:point_rows.append({k:p.get(k) for k in ("task","capture_index","target","camera","status","pixel_error","conditional_true_plane_xy_error_mm","reference_visibility")})
point_counts=[]
for task in SAMPLES:
    pp=[p for p in points if p["task"]==task]
    point_counts.append({"scene":task,"calls":len(pp),"returned":sum(p["status"]=="point_returned_not_validated" for p in pp),"abstained":sum(p["status"]=="model_abstained" for p in pp)})

for name,file in [("storage_wrist_item.png","storage/frames/002_after_align_xy/rig_wrist_right.png"),("storage_global_item.png","storage/frames/002_after_align_xy/rig_global.png"),("storage_old_item.png","storage/frames/002_after_align_xy/overview.png"),("storage_wrist_slot.png","storage/frames/004_after_move_above/rig_wrist_right.png"),("handover_wrist.png","handover/frames/005_after_approach/rig_wrist_right.png"),("pour_global.png","pour/frames/008_after_align_receiver/rig_global.png")]:cp(OUT/"unified_cameras/selected_v3"/file,name)
for name,file in [("cover_bowl_overlay.png","000_bowl_overlay.png"),("cover_apple_overlay.png","006_apple_overlay.png")]:
    src=OUT/"cover_cabto_fix/final_visual/C3/perception"/file
    if src.is_file():cp(src,name)
cp(OUT/"remaining_cabto/storage/final_seed0/STORAGE/step08/before.png","storage_carry.png")
cp(OUT/"remaining_cabto/handover/final_seed0/HANDOVER/step03/before.png","handover_dual_hold.png")
cp(OUT/"remaining_cabto/pour/final_seed0/POUR/step04/before.png","pour_caught.png")

# Capture exact generated policy examples without editing their contents.
cover=read(OUT/"cover_cabto_fix/sampling_9b_complete/policies.json")
source=next(v["source"] for v in cover["accepted"].values() if v["model"]["name"]=="pick")
(HERE/"assets/generated_pick.py").write_text(source+"\n")

for name in ["project/probe_free_tray_packing.py","project/core/cover_cabto_grounding.py","project/core/cover_cabto_api.py","project/run_remaining_cabto.py","project/refine_remaining_frames.py"]:
    path=ROOT/name;manifest.append({"path":name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"purpose":"operational_definitions_and_verification_semantics"})

results={"scenes":scenes,"totals":{"goals":sum(x["goals"] for x in scenes),"actions":sum(x["grounded_actions"] for x in scenes),"episodes":len(all_episodes),"transitions":sum(x["transitions"] for x in scenes)},
         "paired_training_witnesses":witness_summary,"model_patches":patches,"camera_summary":camera,"camera_point_counts":point_counts,"camera_point_rows":point_rows,
         "interpretation":"Selected engineering outcomes, not iid samples or equal-budget baselines. Original runs and camera compatibility runs kept separate."}
(HERE/"evidence.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
(HERE/"source_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({"scenes":scenes,"totals":results["totals"],"paired":witness_summary,"camera_point_counts":point_counts},ensure_ascii=False,indent=2))
