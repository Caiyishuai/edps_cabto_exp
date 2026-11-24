"""Staged (per-action) evidence and incremental refinement prompts for P2."""
from __future__ import annotations

from collections import defaultdict
import json

from blocks_p1_candidates import (ACTION_ARGS, PREDICATES, CandidateRejected, OfflineProposer,
                                  compile_candidate, parse_proposal, recipe, validate_candidate)


def action_name(model):
    return model["name"] if isinstance(model, dict) else model


def staged_evidence(result, evidence_id, current_offset):
    """Group observed steps by action name and label each with its role."""
    by_action = defaultdict(list)
    for index, step in enumerate(result["steps"]):
        check = step.get("check", {})
        name = action_name(step["model"])
        role = ("grasp" if name in ("pick_table", "unstack")
                else "placement" if name in ("stack", "put_table") else "other")
        entry = {"index": index, "action": name, "role": role, "model": step["model"],
                 "before": step["before"]["state"], "after": step.get("after", {}).get("state"),
                 "exception": step.get("exception"), "violations": check.get("violations"),
                 "success": bool(check.get("effect_ok", False))}
        if not entry["success"]:
            entry["api_trace"] = step["api_trace"]
            entry["finger_forces_N"] = step.get("after", {}).get("finger_forces_N")
        by_action[name].append(entry)
    return {"evidence_id": evidence_id, "current_grasp_offset_z_m": current_offset,
            "success": result["success"], "initial_state": result.get("initial", {}).get("state"),
            "actions": {k: v for k, v in sorted(by_action.items())}, "exception": result["exception"]}


def describe_failure(result):
    """Return which action first failed and its observed violations (no correct patch)."""
    for step in result.get("steps", []):
        if not step.get("check", {}).get("effect_ok", False):
            name = action_name(step["model"])
            return {"failed_action": name,
                    "role": ("grasp" if name in ("pick_table", "unstack") else
                             "placement" if name in ("stack", "put_table") else "other"),
                    "exception": step.get("exception"),
                    "violations": step.get("check", {}).get("violations"),
                    "finger_forces_N": step.get("after", {}).get("finger_forces_N"),
                    "before_state": step["before"]["state"],
                    "after_state": step.get("after", {}).get("state")}
    if not result.get("success"):
        return {"failed_action": None, "role": "final_goal", "exception": result.get("exception"),
                "final_check": result.get("final_check", {}).get("goal_reached")}
    return None


def proposal_messages_staged(evidence, protocol, accumulated=None, last_failure=None, max_candidates=1):
    language = {"predicates": PREDICATES, "action_arguments": {k: sorted(v) for k, v in ACTION_ARGS.items()},
                "offset_bounds_m": protocol["offset_bounds_m"], "max_candidates": max_candidates,
                "max_pre_add": protocol["max_pre_add"]}
    system = (
        "You repair a robot action library and a shared grasp parameter from measured execution evidence. "
        "Observations are grouped by action name and labeled by role (grasp= pick_table/unstack, placement= stack/put_table). "
        "Diagnose each role separately: a grasp failure (zero finger force) is a parameter/controller problem; "
        "a placement failure on an occupied support is a missing precondition (clear support) or a missing clearing action. "
        "Do not mix a placement failure into a grasp parameter. Return exactly one JSON object; no Python, markdown, or simulator writes. "
        "Do not alter effects, goals, safety thresholds, objects, or controls. Propose reusable lifted edits, not object-color exceptions. "
        "If an accumulated patch is present, it is already retained; propose only the ADDITIONAL edit that fixes the REMAINING failure, "
        "and do not repeat an already-applied edit. Use decision=abstain if evidence does not justify any supported repair. "
        "Output schema: {\"decision\":\"propose\" or \"abstain\",\"reason\":string,\"candidates\":["
        "{\"id\":short_string,\"pre_add\":[{\"action\":action_name,\"predicate\":predicate_name,\"args\":[bound_argument_name]}],"
        "\"offset_z_m\":number_or_null,\"reason\":short_string}]}. "
        "pre_add appends positive preconditions to every grounded action with that name; args use only its object/support variables. "
        "offset_z_m is the REPLACEMENT offset from the read-only grasp_point(), in metres, shared by pick_table and unstack; "
        "null means preserve each original parameter. Predicate names: empty = released hand; holding = carried object; "
        "ontable = physical table support; on = direct object support; clear = no other block supported on the object; "
        "home_completed = actuated HOME. Existing unstack and put_table are real available skills; the planner may use them. "
        "Any predicates not declared here are unsupported. Each candidate can change either or both layers."
    )
    payload = {"language_contract": language, "observed_executions": evidence}
    if accumulated is not None:
        payload["accumulated_patch"] = accumulated
    if last_failure is not None:
        payload["last_execution_failure"] = last_failure
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def accumulate(candidate, accumulated):
    """Merge a newly proposed candidate into the retained patch (union pre_add, latest offset)."""
    result = {"pre_add": list(accumulated.get("pre_add", [])), "offset_z_m": accumulated.get("offset_z_m")}
    seen = {(e["action"], e["predicate"], tuple(e["args"])) for e in result["pre_add"]}
    for edit in candidate.get("pre_add", []):
        key = (edit["action"], edit["predicate"], tuple(edit["args"]))
        if key not in seen:
            result["pre_add"].append(edit)
            seen.add(key)
    if candidate.get("offset_z_m") is not None:
        result["offset_z_m"] = candidate["offset_z_m"]
    return result


def accumulated_candidate(accumulated):
    return {"id": "accumulated", "pre_add": accumulated.get("pre_add", []),
            "offset_z_m": accumulated.get("offset_z_m"), "reason": "incremental accumulated patch"}
