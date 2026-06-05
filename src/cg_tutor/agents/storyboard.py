"""Storyboard Agent.

Narrative → Storyboard (concrete camera/object/keyframe spec per shot).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cg_tutor.agents.base import load_prompt, save_artifact
from cg_tutor.config import get_agent_model
from cg_tutor.llm_client import LLMClient
from cg_tutor.scene_ir import build_scene_ir, verify_scene_ir
from cg_tutor.schemas import CameraKey, Narrative, OverlayZone, SceneObject, Storyboard
from cg_tutor.scene_profiles import SceneProfile
from cg_tutor._logging import get_logger


log = get_logger(__name__)


AGENT = "storyboard"


class ShotPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: list[CameraKey] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    overlay_zone: OverlayZone | None = None


def to_storyboard(
    narrative: Narrative,
    *,
    model: str | None = None,
    out_dir: Path | None = None,
    candidates: int = 2,
    style_addendum: str = "",
    scene_profile: SceneProfile | None = None,
) -> Storyboard:
    mode = os.environ.get("CG_TUTOR_STORYBOARD_MODE", "patch").strip().lower()
    if mode in ("full", "legacy"):
        return _to_storyboard_full(
            narrative,
            model=model,
            out_dir=out_dir,
            candidates=candidates,
            style_addendum=style_addendum,
            scene_profile=scene_profile,
        )
    if mode not in ("patch", "patches"):
        raise ValueError("CG_TUTOR_STORYBOARD_MODE must be 'patch' or 'full'")
    return _to_storyboard_patch(
        narrative,
        model=model,
        out_dir=out_dir,
        candidates=candidates,
        style_addendum=style_addendum,
        scene_profile=scene_profile,
    )


def _to_storyboard_full(
    narrative: Narrative,
    *,
    model: str | None = None,
    out_dir: Path | None = None,
    candidates: int = 2,
    style_addendum: str = "",
    scene_profile: SceneProfile | None = None,
) -> Storyboard:
    cfg = get_agent_model(AGENT)
    if model:
        chain = (model,)
    else:
        chain = cfg.chain or ("codex-cli/gpt-5.5",)
    candidate_slugs = _storyboard_candidate_slugs(candidates, model, chain)
    system = load_prompt("storyboard")
    user = _storyboard_user_prompt(narrative, style_addendum=style_addendum)
    errors: list[str] = []
    valid: list[tuple[tuple, int, str, dict, Storyboard]] = []

    # Validate after each provider attempt. A provider can return parseable
    # JSON that is still the wrong schema (for example, a single shot object
    # instead of a full Storyboard); that should fall through just like an
    # API/timeout failure.
    for idx, slug in enumerate(candidate_slugs):
        client_chain = (
            (slug,) if model else _storyboard_client_chain(slug, chain)
        )
        for provider_slug in client_chain:
            safe_slug = provider_slug.replace("/", "_").replace(":", "_")
            artifact_prefix = f"storyboard.cand{idx:02d}.{safe_slug}"
            try:
                client = LLMClient.from_chain((provider_slug,), max_retries=1)
                raw, sb = _complete_and_validate_candidate(
                    client=client,
                    system=system,
                    user=user,
                    narrative=narrative,
                    out_dir=out_dir,
                    artifact_prefix=artifact_prefix,
                    style_addendum=style_addendum,
                )
                score = _storyboard_quality_key(sb, narrative, scene_profile)
                valid.append((score, idx, provider_slug, raw, sb))
                if out_dir is not None:
                    ir_report = verify_scene_ir(
                        build_scene_ir(
                            narrative, sb, scene_profile=scene_profile,
                        )
                    )
                    save_artifact(
                        out_dir,
                        f"{artifact_prefix}.score.json",
                        json.dumps({
                            "model": provider_slug,
                            "score_key": list(score),
                            "shots": len(sb.shots),
                            "duration_sec": sb.total_duration,
                            "scene_ir_block": ir_report.block_count,
                            "scene_ir_warn": ir_report.warn_count,
                        }, indent=2),
                    )
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{provider_slug}: {e}")
                if out_dir is not None:
                    error_path = out_dir / f"{artifact_prefix}.error.txt"
                    if not error_path.exists():
                        save_artifact(out_dir, error_path.name, str(e))
                log.info(f"candidate={idx} provider={provider_slug} "
                    f"failed validation/call: {e}",
                )
                continue

    if valid:
        score, idx, slug, raw, sb = max(valid, key=lambda item: item[0])
        if out_dir is not None:
            body = json.dumps(raw, indent=2)
            save_artifact(out_dir, "storyboard.raw.json", body)
            normalized_for_diff = _normalize_relative_times(
                _normalize_shot_start_times(raw)
            )
            normalization_audit = _record_normalization(raw, normalized_for_diff)
            save_artifact(
                out_dir,
                "storyboard.selection.json",
                json.dumps({
                    "selected_candidate": idx,
                    "selected_model": slug,
                    "score_key": list(score),
                    "candidate_count": len(candidate_slugs),
                    "valid_count": len(valid),
                    "normalization": normalization_audit,
                }, indent=2),
            )
            save_artifact(out_dir, "storyboard.json", sb.model_dump_json(indent=2))
        if len(candidate_slugs) > 1:
            log.info(f"selected candidate={idx} provider={slug} "
                f"from {len(valid)}/{len(candidate_slugs)} valid",
            )
        return sb

    if os.environ.get("CG_TUTOR_STORYBOARD_DETERMINISTIC_FALLBACK", "1") != "0":
        raw = _deterministic_storyboard_raw(
            narrative,
            scene_profile=scene_profile,
        )
        motion_audit = _ensure_storyboard_motion(raw)
        sb = validate_storyboard(raw)
        _validate_storyboard_against_narrative(sb, narrative)
        if out_dir is not None:
            score = _storyboard_quality_key(sb, narrative, scene_profile)
            save_artifact(
                out_dir,
                "storyboard.deterministic_fallback.json",
                json.dumps({
                    "reason": "all storyboard providers failed",
                    "errors": errors,
                    "score_key": list(score),
                    "motion_postprocess": motion_audit,
                    "raw": raw,
                }, indent=2),
            )
            save_artifact(out_dir, "storyboard.raw.json", json.dumps(raw, indent=2))
            save_artifact(
                out_dir,
                "storyboard.selection.json",
                json.dumps({
                    "selected_candidate": -1,
                    "selected_model": "deterministic_fallback",
                    "score_key": list(score),
                    "candidate_count": len(candidate_slugs),
                    "valid_count": 0,
                    "fallback": True,
                    "motion_postprocess": motion_audit,
                    "normalization": _record_normalization(raw, raw),
                }, indent=2),
            )
            save_artifact(out_dir, "storyboard.json", sb.model_dump_json(indent=2))
        log.info("all storyboard providers failed; using deterministic fallback")
        return sb

    raise RuntimeError("all storyboard providers failed:\n" + "\n".join(errors))


def _storyboard_max_tokens() -> int:
    return int(os.environ.get("CG_TUTOR_STORYBOARD_MAX_TOKENS", "9000"))


def _storyboard_patch_max_tokens() -> int:
    return int(os.environ.get("CG_TUTOR_STORYBOARD_PATCH_MAX_TOKENS", "3500"))


def _to_storyboard_patch(
    narrative: Narrative,
    *,
    model: str | None = None,
    out_dir: Path | None = None,
    candidates: int = 2,
    style_addendum: str = "",
    scene_profile: SceneProfile | None = None,
) -> Storyboard:
    cfg = get_agent_model(AGENT)
    chain = (model,) if model else (cfg.chain or ("anthropic/claude-sonnet-4.6",))
    base_raw = _deterministic_storyboard_raw(
        narrative,
        scene_profile=scene_profile,
    )
    patch_records: list[dict[str, Any]] = []
    for shot_idx, node in enumerate(narrative.nodes):
        patch, record = _generate_shot_patch(
            narrative=narrative,
            shot_idx=shot_idx,
            base_shot=base_raw["shots"][shot_idx],
            chain=chain,
            style_addendum=style_addendum,
            scene_profile=scene_profile,
            out_dir=out_dir,
        )
        patch_records.append(record)
        if patch is None:
            continue
        _merge_patch_into_shot(base_raw["shots"][shot_idx], patch)

    motion_audit = _ensure_storyboard_motion(base_raw)
    sb = validate_storyboard(base_raw)
    _validate_storyboard_against_narrative(sb, narrative)
    if out_dir is not None:
        score = _storyboard_quality_key(sb, narrative, scene_profile)
        save_artifact(
            out_dir,
            "storyboard.patch_records.json",
            json.dumps(patch_records, indent=2),
        )
        save_artifact(out_dir, "storyboard.raw.json", json.dumps(base_raw, indent=2))
        save_artifact(
            out_dir,
            "storyboard.selection.json",
            json.dumps({
                "selected_candidate": -1,
                "selected_model": "patch_merge",
                "score_key": list(score),
                "candidate_count": max(1, int(candidates)),
                "valid_count": sum(1 for r in patch_records if r.get("ok")),
                "mode": "patch",
                "motion_postprocess": motion_audit,
                "normalization": _record_normalization(base_raw, base_raw),
            }, indent=2),
        )
        save_artifact(out_dir, "storyboard.json", sb.model_dump_json(indent=2))
    return sb


def _generate_shot_patch(
    *,
    narrative: Narrative,
    shot_idx: int,
    base_shot: dict[str, Any],
    chain: tuple[str, ...],
    style_addendum: str,
    scene_profile: SceneProfile | None,
    out_dir: Path | None,
) -> tuple[ShotPatch | None, dict[str, Any]]:
    node = narrative.nodes[shot_idx]
    record: dict[str, Any] = {
        "node_id": node.id,
        "ok": False,
        "provider": None,
        "errors": [],
        "fallback": "deterministic_base",
    }
    for provider_slug in chain:
        safe_slug = provider_slug.replace("/", "_").replace(":", "_")
        artifact_prefix = f"storyboard.patch.{node.id}.{safe_slug}"
        try:
            client = LLMClient.from_chain((provider_slug,), max_retries=1)
            raw = client.complete_json(
                system=_storyboard_patch_system_prompt(),
                user=_storyboard_patch_user_prompt(
                    narrative=narrative,
                    shot_idx=shot_idx,
                    base_shot=base_shot,
                    style_addendum=style_addendum,
                    scene_profile=scene_profile,
                ),
                max_tokens=_storyboard_patch_max_tokens(),
            )
            if out_dir is not None:
                save_artifact(
                    out_dir,
                    f"{artifact_prefix}.raw.json",
                    json.dumps(raw, indent=2),
                )
            patch = _validate_shot_patch(raw, base_shot)
            record.update({
                "ok": True,
                "provider": provider_slug,
                "fallback": None,
                "camera_keys": len(patch.camera),
                "objects": len(patch.objects),
            })
            return patch, record
        except Exception as e:  # noqa: BLE001
            record["errors"].append(f"{provider_slug}: {e}")
            if out_dir is not None:
                save_artifact(out_dir, f"{artifact_prefix}.error.txt", str(e))
            continue
    return None, record


def _storyboard_patch_system_prompt() -> str:
    return (
        "You fill one shot patch for a Blender storyboard. Return ONLY a JSON "
        "object with keys camera, objects, overlay_zone. Do not include "
        "concept_id, fps, resolution, shots, node_id, start_sec, duration_sec, "
        "formula, or caption. Prefer a storyboard that visibly changes over "
        "time: use at least two camera keys and add object keyframes for "
        "teaching-relevant points, rays, vectors, moving objects, or lights."
    )


def _storyboard_patch_user_prompt(
    *,
    narrative: Narrative,
    shot_idx: int,
    base_shot: dict[str, Any],
    style_addendum: str,
    scene_profile: SceneProfile | None,
) -> str:
    node = narrative.nodes[shot_idx]
    parts = [
        "Fill ONLY this ShotPatch JSON object:",
        json.dumps({
            "camera": base_shot["camera"],
            "objects": base_shot["objects"],
            "overlay_zone": base_shot["overlay_zone"],
        }, indent=2),
        "",
        "Patch schema:",
        "- camera: list of global-time camera keys with time_sec, position, look_at, fov.",
        "- objects: list of visible SceneObject entries for this shot, including key_light.",
        "- For each shot, animate at least one teaching-relevant object or light with two keyframes.",
        "- Keep static anchors such as floor grids, screens, walls, and frames stable.",
        "- overlay_zone: object {x,y,w,h} or null. If present, keep it inside "
        "a conservative screen-safe area: x>=0.06, y>=0.06, x+w<=0.94, "
        "y+h<=0.94. Prefer w<=0.40 and h<=0.22 for formulas so the overlay "
        "does not touch frame edges. Do not place storyboard objects or 3D "
        "text in the same area.",
        "- Do not return root Storyboard keys or shot timing keys.",
        "",
        "Current narrative node:",
        node.model_dump_json(indent=2),
        "",
        "Full narrative context:",
        narrative.model_dump_json(indent=2),
    ]
    if scene_profile is not None:
        parts.extend([
            "",
            "Scene profile JSON:",
            scene_profile.model_dump_json(indent=2),
            "",
            "If persistent_anchors exist, include them by exact name in objects.",
        ])
    if style_addendum:
        parts.extend(["", "Style addendum:", style_addendum])
    return "\n".join(parts)


def _validate_shot_patch(raw: dict, base_shot: dict[str, Any]) -> ShotPatch:
    if not isinstance(raw, dict):
        raise ValueError("ShotPatch must be a JSON object")
    forbidden = {
        "concept_id", "fps", "resolution", "shots",
        "node_id", "start_sec", "duration_sec", "formula", "caption",
    }
    present_forbidden = sorted(forbidden & set(raw))
    if present_forbidden:
        raise ValueError(
            "ShotPatch included forbidden structural keys: "
            + ", ".join(present_forbidden)
        )
    patch = ShotPatch.model_validate(_strip_known_extras(raw))
    if not patch.camera:
        patch.camera = [CameraKey.model_validate(item) for item in base_shot["camera"]]
    if not patch.objects:
        patch.objects = [SceneObject.model_validate(item) for item in base_shot["objects"]]
    return patch


def _merge_patch_into_shot(shot: dict[str, Any], patch: ShotPatch) -> None:
    shot["camera"] = [item.model_dump(mode="json") for item in patch.camera]
    shot["objects"] = [item.model_dump(mode="json") for item in patch.objects]
    shot["overlay_zone"] = (
        patch.overlay_zone.model_dump(mode="json")
        if patch.overlay_zone is not None else None
    )


def _ensure_storyboard_motion(raw: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {"changed": False, "shots": []}
    for shot_idx, shot in enumerate(raw.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        shot_audit = _ensure_shot_motion(shot, shot_idx)
        if shot_audit["changed"]:
            audit["changed"] = True
        audit["shots"].append(shot_audit)
    return audit


def _ensure_shot_motion(shot: dict[str, Any], shot_idx: int) -> dict[str, Any]:
    start = float(shot.get("start_sec", 0.0) or 0.0)
    duration = max(0.0, float(shot.get("duration_sec", 0.0) or 0.0))
    audit: dict[str, Any] = {
        "node_id": shot.get("node_id"),
        "changed": False,
        "camera": False,
        "objects": [],
    }
    if duration <= 0.25:
        return audit

    camera = shot.get("camera")
    if isinstance(camera, list) and not _camera_has_motion(camera):
        _add_camera_motion(camera, start, duration, shot_idx)
        audit["changed"] = True
        audit["camera"] = True

    objects = shot.get("objects")
    if not isinstance(objects, list) or _shot_has_object_motion(objects):
        return audit

    for obj in sorted(objects, key=_motion_candidate_key):
        if not isinstance(obj, dict) or _is_static_motion_anchor(obj):
            continue
        if _add_object_motion(obj, start, duration, shot_idx):
            audit["changed"] = True
            audit["objects"].append(obj.get("name"))
            if len(audit["objects"]) >= 2:
                break
    return audit


def _camera_has_motion(camera: list[Any]) -> bool:
    keys = [item for item in camera if isinstance(item, dict)]
    if len(keys) < 2:
        return False
    first = keys[0]
    last = keys[-1]
    pos_delta = _vec_delta(first.get("position"), last.get("position"))
    look_delta = _vec_delta(first.get("look_at"), last.get("look_at"))
    fov_delta = abs(float(first.get("fov", 50.0)) - float(last.get("fov", 50.0)))
    return pos_delta >= 0.35 or look_delta >= 0.2 or fov_delta >= 3.0


def _add_camera_motion(
    camera: list[Any],
    start: float,
    duration: float,
    shot_idx: int,
) -> None:
    if not camera or not isinstance(camera[0], dict):
        return
    first = dict(camera[0])
    first["time_sec"] = round(start, 6)
    last = dict(camera[-1]) if isinstance(camera[-1], dict) else dict(first)
    pos = _vec3(last.get("position"), [0.0, -7.0, 3.2])
    look = _vec3(last.get("look_at"), [0.0, 0.0, 0.9])
    direction = -1.0 if shot_idx % 2 else 1.0
    last["time_sec"] = round(start + duration, 6)
    last["position"] = [
        round(pos[0] + direction * 0.55, 4),
        round(pos[1] + 0.12, 4),
        round(pos[2] + 0.06, 4),
    ]
    last["look_at"] = [
        round(look[0] + direction * 0.08, 4),
        round(look[1], 4),
        round(look[2], 4),
    ]
    last["fov"] = float(last.get("fov", first.get("fov", 50.0)))
    camera[:] = [first, last]


def _shot_has_object_motion(objects: list[Any]) -> bool:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        times = {
            round(float(k.get("time_sec", 0.0)), 4)
            for k in obj.get("keyframes", [])
            if isinstance(k, dict)
        }
        if len(times) >= 2:
            return True
    return False


def _motion_candidate_key(obj: Any) -> tuple[int, str]:
    if not isinstance(obj, dict):
        return (999, "")
    name = str(obj.get("name", "")).lower()
    primitive = str(obj.get("primitive", "")).lower()
    obj_type = str(obj.get("type", "")).lower()
    score = 80
    if any(w in name for w in ("ray", "vector", "arrow", "projected")):
        score = 0
    elif any(w in name for w in ("point", "object", "marker", "sample")):
        score = 10
    elif primitive in {"arrow", "curve_polyline", "sphere", "cube", "cone"}:
        score = 20
    elif obj_type == "light":
        score = 40
    elif obj_type in {"text", "annotation"}:
        score = 70
    return (score, name)


def _is_static_motion_anchor(obj: dict[str, Any]) -> bool:
    name = str(obj.get("name", "")).lower()
    dynamic = ("ray", "vector", "arrow", "projected", "point", "object")
    if any(w in name for w in dynamic):
        return False
    static = (
        "floor", "grid", "bench", "wall", "window", "frame", "screen",
        "plane", "aperture", "camera_center", "axis", "axes",
    )
    return any(w in name for w in static)


def _add_object_motion(
    obj: dict[str, Any],
    start: float,
    duration: float,
    shot_idx: int,
) -> bool:
    if obj.get("keyframes"):
        return False
    end = round(start + max(duration * 0.82, 0.25), 6)
    name = str(obj.get("name", "")).lower()
    obj_type = str(obj.get("type", "")).lower()
    primitive = str(obj.get("primitive", "")).lower()
    loc = _vec3(obj.get("location"), [0.0, 0.0, 0.0])
    sign = -1.0 if shot_idx % 2 else 1.0
    if obj_type == "light":
        obj["keyframes"] = [
            {"time_sec": round(start, 6), "attr": "location", "value": loc},
            {
                "time_sec": end,
                "attr": "location",
                "value": [
                    round(loc[0] + sign * 0.45, 4),
                    round(loc[1] + 0.2, 4),
                    round(loc[2], 4),
                ],
            },
        ]
        return True
    if primitive in {"arrow", "curve_polyline"} or any(
        w in name for w in ("ray", "vector", "arrow")
    ):
        obj["keyframes"] = [
            {
                "time_sec": round(start, 6),
                "attr": "scale",
                "value": [0.72, 0.72, 0.72],
            },
            {"time_sec": end, "attr": "scale", "value": [1.08, 1.08, 1.08]},
        ]
        return True
    obj["keyframes"] = [
        {"time_sec": round(start, 6), "attr": "location", "value": loc},
        {
            "time_sec": end,
            "attr": "location",
            "value": [
                round(loc[0] + sign * 0.28, 4),
                round(loc[1] + 0.18, 4),
                round(loc[2] + 0.06, 4),
            ],
        },
    ]
    return True


def _vec_delta(a: Any, b: Any) -> float:
    av = _vec3(a, [0.0, 0.0, 0.0])
    bv = _vec3(b, [0.0, 0.0, 0.0])
    return sum((x - y) ** 2 for x, y in zip(av, bv)) ** 0.5


def _vec3(value: Any, default: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            pass
    return [float(default[0]), float(default[1]), float(default[2])]


def _storyboard_user_prompt(
    narrative: Narrative,
    *,
    style_addendum: str = "",
) -> str:
    parts = [
        "NARRATIVE JSON:",
        narrative.model_dump_json(indent=2),
        "",
        "FILL THIS STORYBOARD SKELETON:",
        "Return the same top-level object shape. Do not remove root keys, do "
        "not return a single shot, and do not change the number/order of "
        "shots. Missing any skeleton shot is invalid. Replace placeholder "
        "camera/objects fields with concrete schema-valid values while "
        "preserving node_id, start_sec, "
        "duration_sec, formula, and caption unless formula simplification is "
        "needed for one-line mathtext.",
        json.dumps(_storyboard_skeleton(narrative), indent=2),
    ]
    if style_addendum:
        parts.extend(["", "---", style_addendum])
    return "\n".join(parts)


def _complete_and_validate_candidate(
    *,
    client,
    system: str,
    user: str,
    narrative: Narrative,
    out_dir: Path | None,
    artifact_prefix: str,
    style_addendum: str = "",
) -> tuple[dict, Storyboard]:
    raw = client.complete_json(
        system=system,
        user=user,
        max_tokens=_storyboard_max_tokens(),
    )
    raw = _fill_storyboard_defaults(raw, narrative)
    _ensure_storyboard_motion(raw)
    if out_dir is not None:
        save_artifact(
            out_dir,
            f"{artifact_prefix}.raw.json",
            json.dumps(raw, indent=2),
        )
    try:
        sb = validate_storyboard(raw)
        _validate_storyboard_against_narrative(sb, narrative)
        return raw, sb
    except Exception as first_error:  # noqa: BLE001
        if out_dir is not None:
            save_artifact(
                out_dir,
                f"{artifact_prefix}.invalid.json",
                json.dumps(raw, indent=2),
            )
        try:
            repair_raw = client.complete_json(
                system=system,
                user=_storyboard_repair_user_prompt(
                    narrative, raw, first_error, style_addendum,
                ),
                max_tokens=_storyboard_max_tokens(),
            )
            repair_raw = _fill_storyboard_defaults(repair_raw, narrative)
            _ensure_storyboard_motion(repair_raw)
            if out_dir is not None:
                save_artifact(
                    out_dir,
                    f"{artifact_prefix}.repair.raw.json",
                    json.dumps(repair_raw, indent=2),
                )
            sb = validate_storyboard(repair_raw)
            _validate_storyboard_against_narrative(sb, narrative)
            return repair_raw, sb
        except Exception as repair_error:  # noqa: BLE001
            if out_dir is not None:
                if "repair_raw" in locals():
                    save_artifact(
                        out_dir,
                        f"{artifact_prefix}.repair.invalid.json",
                        json.dumps(repair_raw, indent=2),
                    )
                save_artifact(
                    out_dir,
                    f"{artifact_prefix}.error.txt",
                    "Initial validation failed:\n"
                    f"{first_error}\n\n"
                    "Repair validation/call failed:\n"
                    f"{repair_error}",
                )
            raise RuntimeError(
                "initial storyboard validation failed and schema repair "
                f"also failed: {first_error}; repair: {repair_error}"
            ) from repair_error


def _fill_storyboard_defaults(raw: dict, narrative: Narrative) -> dict:
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    out.setdefault("concept_id", narrative.concept_id)
    out.setdefault("fps", 24)
    out.setdefault("resolution", [960, 540])
    return out


def _validate_storyboard_against_narrative(
    storyboard: Storyboard,
    narrative: Narrative,
) -> None:
    """Reject schema-valid storyboards that dropped or reordered nodes."""
    expected_ids = [node.id for node in narrative.nodes]
    actual_ids = [shot.node_id for shot in storyboard.shots]
    if actual_ids != expected_ids:
        raise ValueError(
            "storyboard shots must exactly match narrative node order: "
            f"expected {expected_ids}, got {actual_ids}"
        )

    eps = 1e-3
    duration_error = abs(storyboard.total_duration - narrative.total_duration)
    if duration_error > eps:
        raise ValueError(
            "storyboard total_duration must match narrative total_duration: "
            f"expected {narrative.total_duration:.3f}s, "
            f"got {storyboard.total_duration:.3f}s"
        )

    for idx, (shot, node) in enumerate(zip(storyboard.shots, narrative.nodes)):
        if abs(shot.duration_sec - node.duration_sec) > eps:
            raise ValueError(
                f"storyboard shot {idx} ({shot.node_id}) duration_sec must "
                f"match narrative: expected {node.duration_sec:.3f}s, "
                f"got {shot.duration_sec:.3f}s"
            )


def _storyboard_skeleton(narrative: Narrative) -> dict[str, Any]:
    shots: list[dict[str, Any]] = []
    cursor = 0.0
    for node in narrative.nodes:
        duration = float(node.duration_sec)
        formula = node.formulas[0] if node.formulas else None
        shots.append({
            "node_id": node.id,
            "start_sec": round(cursor, 6),
            "duration_sec": duration,
            "camera": "<fill with at least 2 global-time camera keys>",
            "objects": "<fill with visible objects for this shot, including key_light>",
            "overlay_zone": (
                {"x": 0.56, "y": 0.07, "w": 0.36, "h": 0.18}
                if formula else None
            ),
            "formula": formula,
            "caption": node.description,
        })
        cursor += duration
    return {
        "concept_id": narrative.concept_id,
        "fps": 24,
        "resolution": [960, 540],
        "shots": shots,
    }


def _deterministic_storyboard_raw(
    narrative: Narrative,
    *,
    scene_profile: SceneProfile | None = None,
) -> dict[str, Any]:
    shots: list[dict[str, Any]] = []
    anchors = _fallback_anchor_names(scene_profile)
    cursor = 0.0
    for idx, node in enumerate(narrative.nodes):
        duration = float(node.duration_sec)
        formula = node.formulas[0] if node.formulas else None
        shots.append({
            "node_id": node.id,
            "start_sec": round(cursor, 6),
            "duration_sec": duration,
            "camera": _fallback_camera_keys(cursor, duration),
            "objects": _fallback_objects_for_shot(anchors, idx),
            "overlay_zone": (
                {"x": 0.56, "y": 0.07, "w": 0.36, "h": 0.18}
                if formula else None
            ),
            "formula": formula,
            "caption": node.description,
        })
        cursor += duration
    return {
        "concept_id": narrative.concept_id,
        "fps": 24,
        "resolution": [960, 540],
        "shots": shots,
    }


def _fallback_camera_keys(start_sec: float, duration_sec: float) -> list[dict[str, Any]]:
    end_sec = start_sec + duration_sec
    return [
        {
            "time_sec": round(start_sec, 6),
            "position": [0.0, -7.0, 3.2],
            "look_at": [0.0, 0.0, 0.9],
            "fov": 48.0,
        },
        {
            "time_sec": round(end_sec, 6),
            "position": [0.25, -7.0, 3.2],
            "look_at": [0.0, 0.0, 0.9],
            "fov": 48.0,
        },
    ]


def _fallback_anchor_names(scene_profile: SceneProfile | None) -> list[str]:
    anchors = [
        _safe_object_name(item)
        for item in (scene_profile.persistent_anchors if scene_profile else [])
    ]
    anchors = [a for a in anchors if a]
    if not anchors:
        base = scene_profile.base_profile if scene_profile else ""
        if base == "transformation_demo":
            anchors = [
                "hero_mesh",
                "source_shape_A",
                "target_shape_B",
                "sample_vertex_marker",
                "trajectory_trace",
                "parameter_panel",
            ]
        elif base == "curve_construction":
            anchors = [
                "control_point_A",
                "control_point_B",
                "control_polygon",
                "interpolation_point",
                "curve_trace",
            ]
        elif base == "vector_teaching":
            anchors = [
                "teaching_floor",
                "main_teaching_object",
                "point_A",
                "point_B",
                "vector_cue",
                "label_panel",
            ]
        else:
            anchors = [
                "ground_plane",
                "main_subject",
                "background_reference",
                "small_hud_panel",
            ]
    out: list[str] = []
    for anchor in anchors:
        if anchor not in out:
            out.append(anchor)
    return out[:10]


def _safe_object_name(text: str) -> str:
    name = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _fallback_objects_for_shot(anchors: list[str], shot_idx: int) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = [
        {
            "name": "key_light",
            "type": "light",
            "primitive": None,
            "location": [0.0, -3.5, 5.0],
            "properties": {
                "light_kind": "AREA",
                "energy": 650,
                "size": 5.0,
                "light_color": [1.0, 0.96, 0.9],
            },
            "keyframes": [],
        }
    ]
    for idx, anchor in enumerate(anchors):
        objects.append(_fallback_object_for_anchor(anchor, idx, shot_idx))
    return objects


def _fallback_object_for_anchor(anchor: str, idx: int, shot_idx: int) -> dict[str, Any]:
    low = anchor.lower()
    if any(w in low for w in ("ray", "line", "axis", "vector", "trace", "trajectory", "curve")):
        return {
            "name": anchor,
            "type": "mesh",
            "primitive": "curve_polyline",
            "location": [0.0, 0.0, 0.0],
            "properties": {
                "points": [[2.0, 0.25, 1.35], [0.0, 0.0, 1.0], [-2.0, -0.25, 0.95]],
                "bevel_depth": 0.012,
                "color": [0.2, 0.75, 1.0],
                "emission": 0.6,
            },
            "keyframes": [],
        }
    if any(w in low for w in ("panel", "screen")):
        return {
            "name": anchor,
            "type": "primitive",
            "primitive": "plane",
            "location": [2.4, 0.05, 1.45],
            "properties": {
                "size": 1.2,
                "color": [0.08, 0.11, 0.14],
                "alpha": 0.35,
            },
            "keyframes": [],
        }
    if any(w in low for w in ("plane", "screen", "grid", "floor")):
        location = [-2.0, 0.0, 1.0] if "image" in low or "screen" in low else [0.0, 0.0, 0.0]
        return {
            "name": anchor,
            "type": "primitive",
            "primitive": "plane",
            "location": location,
            "properties": {
                "size": 3.0 if location[2] > 0 else 6.0,
                "color": [0.55, 0.75, 1.0],
                "alpha": 0.35 if location[2] > 0 else 0.18,
            },
            "keyframes": [],
        }
    if any(w in low for w in ("box", "cube", "object", "pyramid", "mesh", "shape", "source", "target")):
        return {
            "name": anchor,
            "type": "primitive",
            "primitive": "cube",
            "location": [2.0 + 0.15 * shot_idx, 0.0, 1.0],
            "properties": {
                "size": 0.8,
                "color": [0.95, 0.35, 0.25],
            },
            "keyframes": [],
        }
    if any(w in low for w in ("label", "text")):
        return {
            "name": anchor,
            "type": "annotation",
            "primitive": "empty",
            "location": [-1.6 + 0.25 * idx, 0.0, 1.9],
            "properties": {
                "text": anchor.replace("_", " "),
                "size": 0.18,
                "color": [1.0, 1.0, 1.0],
            },
            "keyframes": [],
        }
    return {
        "name": anchor,
        "type": "primitive",
        "primitive": "sphere",
        "location": _fallback_point_location(low, idx, shot_idx),
        "properties": {
            "radius": 0.08,
            "color": [0.2 + 0.12 * (idx % 5), 0.65, 1.0 - 0.1 * (idx % 4)],
        },
        "keyframes": [],
    }


def _fallback_point_location(anchor: str, idx: int, shot_idx: int) -> list[float]:
    if "projected" in anchor:
        return [-2.0, -0.35 + 0.18 * (idx % 4), 0.9 + 0.12 * (idx % 3)]
    if "pinhole" in anchor or "center" in anchor or anchor.endswith("_c"):
        return [0.0, 0.0, 1.0]
    if "point" in anchor:
        return [2.0 + 0.15 * shot_idx, -0.45 + 0.25 * (idx % 4), 1.0 + 0.16 * (idx % 3)]
    return [-1.2 + 0.45 * (idx % 6), 0.35 * ((idx % 3) - 1), 1.0 + 0.08 * (idx % 4)]


def _storyboard_repair_user_prompt(
    narrative: Narrative,
    raw: dict,
    error: BaseException,
    style_addendum: str = "",
) -> str:
    parts = [
        "Your previous response was invalid for the Storyboard schema.",
        "Repair ONLY the JSON shape and missing fields. Return ONE complete "
        "Storyboard JSON object and nothing else.",
        "",
        "Critical requirements:",
        "- The top-level object must contain concept_id, fps, resolution, shots.",
        "- The top-level object must NOT contain node_id.",
        "- shots must be a non-empty list with one shot per narrative node.",
        f"- shots must contain exactly {len(narrative.nodes)} item(s).",
        "- The shot node_id sequence must exactly match the narrative node "
        "sequence.",
        "- Do not return a single shot, a single narrative node, or shots: [].",
        "- If the invalid response only covered the first node, synthesize "
        "the missing shots from the skeleton below instead of copying the "
        "partial response.",
        "- Preserve the narrative timing.",
        "- Preserve narrative formulas only when a node has formulas. If a "
        "node has formulas: [], set shot.formula to null and overlay_zone to null.",
        "",
        "Validation error:",
        str(error),
        "",
        "Minimum correct root skeleton to fill:",
        json.dumps(_storyboard_skeleton(narrative), indent=2),
        "",
        "Original narrative JSON:",
        narrative.model_dump_json(indent=2),
        "",
        "Your invalid JSON response:",
        json.dumps(raw, indent=2),
    ]
    if style_addendum:
        parts.extend([
            "",
            "Style profile that the repaired Storyboard must obey:",
            style_addendum,
        ])
    return "\n".join(parts)


def _storyboard_candidate_slugs(
    count: int,
    model: str | None,
    chain: tuple[str, ...],
) -> tuple[str, ...]:
    """Provider policy for planned storyboard generation.

    Default policy: rotate through configured providers across candidates.
    Earlier versions used Sonnet for every planned candidate, but single-shot
    root failures are often model-correlated; starting later candidates on a
    different provider gives the selection pool real diversity.

    Explicit ``model=...`` means the caller wants that model, so repeat it
    across candidates.
    """
    n = max(1, int(count))
    if model:
        return tuple(model for _ in range(n))
    unique_chain: list[str] = []
    for slug in chain or ("anthropic/claude-sonnet-4.6",):
        if slug and slug not in unique_chain:
            unique_chain.append(slug)
    return tuple(unique_chain[i % len(unique_chain)] for i in range(n))


def _storyboard_client_chain(
    primary_slug: str,
    configured_chain: tuple[str, ...],
) -> tuple[str, ...]:
    """Use a single planned candidate, but keep configured fallbacks alive."""
    out: list[str] = []
    for slug in (primary_slug, *configured_chain):
        if slug and slug not in out:
            out.append(slug)
    return tuple(out)


def _storyboard_quality_key(
    storyboard: Storyboard,
    narrative: Narrative,
    scene_profile: SceneProfile | None = None,
) -> tuple[int, int, int, float, int]:
    """Deterministic quality key for selecting among valid storyboards."""
    ir_report = verify_scene_ir(
        build_scene_ir(narrative, storyboard, scene_profile=scene_profile)
    )
    node_mismatch = abs(len(storyboard.shots) - len(narrative.nodes))
    duration_error = abs(storyboard.total_duration - narrative.total_duration)
    object_count = sum(len(shot.objects) for shot in storyboard.shots)
    return (
        -ir_report.block_count,
        -ir_report.warn_count,
        -node_mismatch,
        -duration_error,
        object_count,
    )


# Harmless commentary fields LLMs often append at any level of the
# storyboard tree. Schema is now `extra="forbid"` so a stray field would
# fail validation outright; we strip these explicitly so the strict mode
# only catches *unknown* keys (the ones we actually want to surface as
# typos like `start_seconds` or `camara`).
_KNOWN_HARMLESS_EXTRAS: frozenset[str] = frozenset({
    "description", "notes", "purpose", "summary", "intent", "rationale",
    "comment", "_comment", "explanation", "title",
})


def _strip_known_extras(node):
    if isinstance(node, dict):
        return {
            k: _strip_known_extras(v)
            for k, v in node.items()
            if k not in _KNOWN_HARMLESS_EXTRAS
        }
    if isinstance(node, list):
        return [_strip_known_extras(item) for item in node]
    return node


def _record_normalization(original: dict, normalized: dict) -> dict:
    """Diff that survives in storyboard.selection.json so ablation can tell
    whether the LLM actually produced contiguous start_sec or we papered
    over it. Only top-level + per-shot start_sec are recorded."""
    diff: dict = {"shot_start_sec_normalized": False, "extras_stripped": []}
    orig_shots = original.get("shots", []) if isinstance(original, dict) else []
    new_shots = normalized.get("shots", []) if isinstance(normalized, dict) else []
    if len(orig_shots) == len(new_shots):
        for orig, new in zip(orig_shots, new_shots):
            if not isinstance(orig, dict) or not isinstance(new, dict):
                continue
            if float(orig.get("start_sec", 0.0) or 0.0) != float(new.get("start_sec", 0.0) or 0.0):
                diff["shot_start_sec_normalized"] = True
                break
    def _collect_extras(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _KNOWN_HARMLESS_EXTRAS:
                    diff["extras_stripped"].append(f"{path}.{k}".lstrip("."))
                _collect_extras(v, f"{path}.{k}".lstrip("."))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _collect_extras(item, f"{path}[{i}]")
    _collect_extras(original)
    return diff


def validate_storyboard(raw: dict) -> Storyboard:
    """Normalize common LLM storyboard variants, then validate schema."""
    return Storyboard.model_validate(
        _strip_known_extras(
            _normalize_relative_times(
                _normalize_shot_start_times(_normalize_single_shot_root(raw))
            )
        )
    )


def _normalize_single_shot_root(raw: dict) -> dict:
    """Recover the common failure where the model returns one Shot at root.

    The storyboard prompt strongly forbids this, but some providers still emit
    a top-level object shaped like a Shot plus ``shots: []``. If the shape is
    unambiguous, wrap that shot in a valid Storyboard root so the provider
    fallback chain is not wasted on a recoverable formatting error.
    """
    if not isinstance(raw, dict):
        return raw
    if "node_id" not in raw:
        return raw
    if not {"duration_sec", "camera", "objects"}.issubset(raw):
        return raw
    shots = raw.get("shots")
    if shots not in (None, []):
        return raw

    shot_keys = {
        "node_id",
        "start_sec",
        "duration_sec",
        "camera",
        "objects",
        "overlay_zone",
        "formula",
        "caption",
    }
    shot = {k: v for k, v in raw.items() if k in shot_keys}
    root = {
        "concept_id": raw.get("concept_id") or "unknown_concept",
        "fps": raw.get("fps") or 24,
        "resolution": raw.get("resolution") or [960, 540],
        "shots": [shot],
    }
    return root


def _normalize_shot_start_times(raw: dict) -> dict:
    """Recompute shot start_sec from duration_sec when the LLM dropped them.

    Some models (notably gemini/gemini-3.1-pro-preview) copy the prompt's
    example `start_sec: 0.0` onto every shot instead of computing the
    running sum across durations. The Pydantic shot-contiguity check
    then rejects the storyboard and the whole provider chain may
    exhaust. We auto-recover only when the failure pattern is
    unambiguous: every shot has start_sec missing or 0.0 AND
    duration_sec values look usable. If any shot has a non-zero
    start_sec we trust the model and let validation catch real bugs.
    """
    shots = raw.get("shots", [])
    if len(shots) < 2:
        return raw
    starts = [float(s.get("start_sec", 0.0) or 0.0) for s in shots]
    durations = [float(s.get("duration_sec", 0.0) or 0.0) for s in shots]
    if any(st > 1e-9 for st in starts):
        return raw  # the model set some non-zero starts; trust it
    if any(d <= 0 for d in durations):
        return raw  # bad durations — let Pydantic surface that

    out = dict(raw)
    new_shots = []
    cursor = 0.0
    for shot, dur in zip(shots, durations):
        s = dict(shot)
        s["start_sec"] = round(cursor, 6)
        new_shots.append(s)
        cursor += dur
    out["shots"] = new_shots
    return out


def _normalize_relative_times(raw: dict) -> dict:
    """Accept storyboard times expressed either globally or per-shot.

    The prompt asks for global ``start_sec`` per shot, but LLMs often emit
    camera/object key times relative to the current shot, e.g. shot 2 starts
    at 4.5s while its camera keys are 0.0 and 5.0. The runtime and schema
    use global times, so convert values that clearly fit inside the shot's
    local [0, duration] window.

    Defensive: when the input doesn't match the expected shape (e.g. the
    LLM returned camera as a dict instead of a list of camera keys),
    pass it through unchanged. Pydantic will then emit a clear schema
    error rather than this normalizer raising an opaque ``dict(...)``
    failure.
    """
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    shots = []
    raw_shots = raw.get("shots", [])
    if not isinstance(raw_shots, list):
        return raw
    for shot in raw_shots:
        if not isinstance(shot, dict):
            shots.append(shot)
            continue
        shot_out = dict(shot)
        start = float(shot_out.get("start_sec", 0.0) or 0.0)
        duration = float(shot_out.get("duration_sec", 0.0) or 0.0)
        camera_in = shot.get("camera", [])
        objects_in = shot.get("objects", [])
        if isinstance(camera_in, list):
            shot_out["camera"] = _normalize_timed_items(
                camera_in, start, duration
            )
        if isinstance(objects_in, list):
            objects = []
            for obj in objects_in:
                if not isinstance(obj, dict):
                    objects.append(obj)
                    continue
                obj_out = _normalize_object(obj)
                kfs = obj.get("keyframes", [])
                if isinstance(kfs, list):
                    obj_out["keyframes"] = _normalize_timed_items(
                        kfs, start, duration
                    )
                objects.append(obj_out)
            shot_out["objects"] = objects
        shots.append(shot_out)
    out["shots"] = shots
    return out


def _normalize_object(obj: dict) -> dict:
    """Accept common near-schema object variants emitted by LLMs."""
    out = dict(obj)
    # The schema uses the more explicit helper name, but LLMs often emit
    # `primitive: curve` for a polyline-like mesh object.
    if out.get("primitive") == "curve":
        out["primitive"] = "curve_polyline"
    return out


def _normalize_timed_items(
    items: list[dict], shot_start: float, shot_duration: float
) -> list[dict]:
    times = [
        float(item["time_sec"])
        for item in items
        if isinstance(item, dict) and "time_sec" in item
    ]
    relative = (
        shot_start > 0
        and bool(times)
        and all(0 <= t <= shot_duration + 1e-6 for t in times)
        and any(t < shot_start - 1e-6 for t in times)
    )
    return [_normalize_timed_item(item, shot_start, relative) for item in items]


def _normalize_timed_item(item: dict, shot_start: float, relative: bool) -> dict:
    if not isinstance(item, dict):
        return item
    out = dict(item)
    if "time_sec" not in out or not relative:
        return out
    out["time_sec"] = float(out["time_sec"]) + shot_start
    return out
