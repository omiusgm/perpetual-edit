#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from booster.audio import SoundEngine
from booster.association import (
    AppleVisionResult,
    AssociationDetector,
    AssociationMatch,
    AssociationProfile,
    AsyncAppleVisionSemantics,
    OCRText,
)
from booster.archive import EditArchive
from booster.buffer import FrameRingBuffer, ObjectObservation
from booster.config import (
    AppConfig,
    choose_non_repeating_preset,
    choose_session_first_preset,
    choose_varied_preset,
    load_config,
)
from booster.edits import EditController
from booster.director import DirectorCoordinator, DirectorOptions
from booster.events import ActionEvent, ActionEventDetector
from booster.gesture_memes import (
    GestureMemeController,
    GestureMemeView,
    classify_meme_gesture,
)
from booster.hud import (
    EventLog,
    HudStatus,
    LiveMemeOverlay,
    compose_hud,
    edit_panel_size,
    error_screen,
    hit_test_director_control,
)
from booster.modes import KioskGate, next_mode
from booster.perception import AsyncSemanticPerception, SemanticPerceptionResult
from booster.qwen_director import (
    AsyncQwenMemeDirector,
    QwenContextMemory,
    QwenDirection,
    QwenMemePack,
)
from booster.story import SemanticStoryDetector
from booster.vision import VisionResult, create_vision_analyzer


PROJECT_ROOT = Path(__file__).resolve().parent


def build_association_profiles(config: Mapping[str, Any]) -> Tuple[AssociationProfile, ...]:
    """Adapt the human-readable config schema to the deterministic matcher."""

    hold = float(config.get("hold_seconds", 0.7))
    release = float(config.get("release_seconds", 0.9))
    cooldown = float(config.get("cooldown_seconds", 7.0))
    profiles = []
    for raw in config.get("profiles", ()):
        prohibited = {
            "face_labels",
            "race",
            "ethnicity",
            "nationality",
            "nationality_from_face",
            "appearance_nationality",
        }
        unsafe = prohibited.intersection(_text_key(key).replace(" ", "_") for key in raw)
        if unsafe:
            raise ValueError(
                "face/ethnicity association fields are not supported: "
                + ", ".join(sorted(unsafe))
            )
        text_terms = tuple(str(value) for value in raw.get("text_any", ()))
        object_labels = tuple(str(value) for value in raw.get("label_any", ()))
        pool = tuple(str(value) for value in raw.get("preset_pool", ()))
        if not pool:
            continue
        # Exact product text must win over broad labels such as "bottle".
        default_priority = 100 if text_terms else 25
        if str(raw.get("id", "")) == "beverage_unknown":
            default_priority = 5
        profiles.append(
            AssociationProfile.from_mapping(
                {
                    "id": raw.get("id", ""),
                    "preset_id": pool[0],
                    "title": raw.get("title", raw.get("id", "")),
                    "scene_labels": raw.get("scene_any", ()),
                    "object_labels": object_labels,
                    "ocr_terms": text_terms,
                    "minimum_confidence": raw.get("minimum_confidence", 0.50),
                    "fuzzy_ocr_threshold": raw.get(
                        "fuzzy_ocr_threshold",
                        0.58 if text_terms else 0.78,
                    ),
                    "hold_seconds": raw.get("hold_seconds", hold),
                    "release_seconds": raw.get("release_seconds", release),
                    "cooldown_seconds": raw.get("cooldown_seconds", cooldown),
                    "priority": raw.get("priority", default_priority),
                    "payload": dict(raw),
                }
            )
        )
    return tuple(profiles)


def _text_key(value: str) -> str:
    return " ".join(str(value).casefold().replace("_", " ").split())


def association_focus_box(
    match: AssociationMatch,
    result: Optional[AppleVisionResult],
    vision: VisionResult,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate the visible evidence without ever using the person's face box."""

    if result is not None and match.matched_ocr_terms:
        expected = tuple(_text_key(value) for value in match.matched_ocr_terms)
        ranked = []
        for item in result.texts:
            observed = _text_key(item.text)
            similarity = max(
                (SequenceMatcher(None, term, observed).ratio() for term in expected),
                default=0.0,
            )
            ranked.append((similarity * item.confidence, item.box))
        if ranked:
            return max(ranked, key=lambda value: value[0])[1]
    matched_objects = tuple(_text_key(value) for value in match.matched_object_labels)
    for item in vision.objects:
        label = _text_key(item.label)
        padded_label = f" {label} "
        if any(
            label == value
            or f" {value} " in padded_label
            or padded_label in f" {value} "
            for value in matched_objects
        ):
            return item.box
    if result is not None:
        regional = [
            item
            for item in result.labels
            if item.region != "full_frame" and item.region_kind not in {"face", "person"}
        ]
        if regional:
            return max(regional, key=lambda item: item.confidence).box
    return None


def association_event(
    match: AssociationMatch,
    focus_box: Optional[Tuple[int, int, int, int]],
) -> ActionEvent:
    payload = dict(match.payload)
    pool = tuple(str(value) for value in payload.get("preset_pool", ()))
    captions = tuple(str(value) for value in payload.get("captions", ()))
    focus_label = str(payload.get("focus_label", "object"))
    return ActionEvent(
        kind=match.profile_id,
        timestamp=match.timestamp,
        confidence=match.confidence,
        source="OBJECT_ASSOCIATION",
        preset_id=match.preset_id,
        story_id=f"context_{match.profile_id}",
        focus_label=focus_label,
        started_at=match.timestamp - 0.8,
        peak_at=match.timestamp,
        ended_at=match.timestamp,
        captions=captions,
        evidence=match.evidence,
        context_id=match.profile_id,
        context_label=str(payload.get("title", match.title)),
        preset_pool=pool,
    )


def qwen_direction_event(
    direction: QwenDirection,
    pack: QwenMemePack,
    evidence: Sequence[str],
) -> ActionEvent:
    """Convert a whitelisted text-only Qwen choice into a normal story event."""

    return ActionEvent(
        kind=f"qwen_{pack.pack_id}",
        timestamp=direction.anchor_timestamp,
        confidence=0.86,
        source="SEMANTIC_QWEN",
        preset_id=pack.preset_pool[0],
        story_id=f"qwen_{pack.pack_id}",
        focus_label=pack.focus_label,
        started_at=direction.anchor_timestamp - 2.6,
        peak_at=direction.anchor_timestamp,
        ended_at=direction.anchor_timestamp,
        captions=pack.captions,
        evidence=tuple(str(value) for value in evidence),
        context_id=f"qwen_{pack.pack_id}",
        context_label=pack.title,
        preset_pool=pack.preset_pool,
    )


def _demo_product_box(width: int, height: int) -> Tuple[int, int, int, int]:
    return (int(width * 0.72), int(height * 0.27), int(width * 0.14), int(height * 0.45))


def scripted_association_result(
    timestamp: float,
    epoch: int,
    width: int,
    height: int,
) -> AppleVisionResult:
    """Deterministic local fixture used only by --association-demo."""

    visible = 0.5 <= timestamp % 12.0 <= 3.8
    texts = (
        (OCRText("ASAHI SUPER DRY", 1.0, _demo_product_box(width, height)),)
        if visible
        else ()
    )
    return AppleVisionResult(
        epoch=epoch,
        timestamp=float(timestamp),
        frame_width=width,
        frame_height=height,
        texts=texts,
        backends=("SCRIPTED_APPLE_VISION",),
    )


HAND_CONFUSABLE_OBJECT_MIN_CONFIDENCE = {
    "cell phone": 0.62,
}

BARE_HAND_GESTURE_LABELS = {
    "iloveyou",
    "open_palm",
    "pointing_up",
    "thumb_down",
    "thumb_up",
    "victory",
}

BARE_HAND_MEME_GESTURES = {
    "fist",
    "open_palm",
    "point",
    "shaka",
    "victory",
}

# A recognized hand sign is the user's most deliberate signal.  Pose/FaceMesh
# runs in the fast analyzer, while the gesture task runs asynchronously and is
# usually better at classic signs such as Victory.  After pose mimic was added,
# blindly keeping any fast reactor signal meant a persistent SMUG/SIDE_EYE or
# POINTING result could hide the later, fresher hand classification completely.
# Keep the full set here (rather than only the MediaPipe canned labels) because
# custom landmark geometry also produces the two-hand and face-adjacent signs.
HAND_REACTOR_GESTURES = {
    "victory",
    "shaka",
    "point",
    "two_point",
    "thinking",
    "shhh",
    "cover_mouth",
    "fist",
    "hands_head",
    "dance",
    "open_palm",
}

# One frame can now contain three opinions about the same movement: the fast
# hand tracker, the async Gesture Recognizer and the optional body/face pass.
# Keep the arbitration in one small, explicit table.  Deliberate classic signs
# win, a specific full-body action wins over a generic palm/point, and facial
# expressions are used only when neither hands nor body provide stronger
# evidence.  This prevents the skeleton pass from randomly stealing old hand
# memes while still retaining the newer natural face reactions.
_REACTOR_SIGNAL_PRIORITY = {
    "victory": 40,
    "shaka": 40,
    "two_point": 40,
    "thinking": 40,
    "shhh": 40,
    "cover_mouth": 40,
    "hands_head": 40,
    "dance": 40,
    "pointing": 30,
    "reddit": 30,
    "arms_up": 30,
    "point": 20,
    "open_palm": 20,
    "fist": 20,
    "soy": 10,
    "smug": 10,
    "huh": 10,
    "side_eye": 10,
    "smile": 10,
}

# A cached async result may correct the fast tracker only while it is close
# enough to the current frame to represent the same movement.  This matches
# the controller's continuity budget and stays below the worker's broader TTL.
_MAX_DISAGREEING_SEMANTIC_AGE_SECONDS = 0.45


def _reactor_priority(signal: str, channel: str = "") -> int:
    normalized = str(signal or "").strip().casefold()
    if not normalized:
        return 0
    configured = _REACTOR_SIGNAL_PRIORITY.get(normalized)
    if configured is not None:
        return configured
    normalized_channel = str(channel or "").strip().upper()
    return {"HAND": 20, "POSE": 30, "FACE": 10}.get(normalized_channel, 5)


def _select_signal_candidate(
    candidates: Sequence[Tuple[Any, ...]],
    *,
    timestamp_index: int,
) -> Tuple[Any, ...]:
    """Resolve one priority bucket without making list order authoritative."""

    highest_priority = max(int(item[0]) for item in candidates)
    finalists = [item for item in candidates if int(item[0]) == highest_priority]
    if len(finalists) == 1:
        return finalists[0]
    if len({str(item[1]) for item in finalists}) == 1:
        # Both detectors agree; keep the freshest clock so a cached async
        # result cannot slow down an already-correct fast observation.
        return max(finalists, key=lambda item: float(item[timestamp_index]))
    # If equally deliberate detectors disagree, confidence is the correction
    # signal and recency is the deterministic tie-break.  Previously the first
    # list item always won, so the async recognizer could never correct a fast
    # Victory/Shaka/Thinking mix-up.
    return max(
        finalists,
        key=lambda item: (float(item[2]), float(item[timestamp_index])),
    )


def gesture_controller_sample(
    vision: VisionResult,
) -> Tuple[str, float, int, float, str]:
    """Return the one coherent sample consumed by the passive cat reactor."""

    # A fresh neutral observation releases the previous reaction.  Preserve
    # the reactor clock even when its signal is empty instead of falling
    # through to an unset meme clock and getting stuck in REFRACTORY.
    if vision.reactor_signal or vision.reactor_sample_timestamp >= 0.0:
        return (
            vision.reactor_signal,
            vision.reactor_confidence,
            vision.reactor_sample_epoch,
            vision.reactor_sample_timestamp,
            vision.reactor_channel or "HAND",
        )
    return (
        vision.meme_gesture,
        vision.meme_gesture_confidence,
        vision.meme_sample_epoch,
        vision.meme_sample_timestamp,
        "HAND" if vision.meme_gesture else "",
    )


def filter_hand_confused_objects(
    objects: Sequence[ObjectObservation],
    hand_landmarks: Sequence[Sequence[Tuple[float, float, float]]],
    hand_gestures: Sequence[Tuple[str, float]],
    frame_size: Tuple[int, int],
) -> Tuple[ObjectObservation, ...]:
    """Drop weak object guesses that are mostly an already-tracked hand.

    EfficientDet occasionally calls an open palm a phone around its global
    0.42 threshold.  A real high-confidence phone is retained, as is a weak
    phone away from every tracked hand.  Only landmarks produced by the same
    semantic job are compared; the faster current-frame hand box may belong to
    another timestamp (or be the union of two hands) and is intentionally not
    used. Filtering here keeps the false label out of both the HUD and the
    object-to-meme association engine.
    """

    frame_width, frame_height = frame_size
    kept = []
    for observation in objects:
        label = observation.label.strip().casefold()
        minimum = HAND_CONFUSABLE_OBJECT_MIN_CONFIDENCE.get(label)
        if minimum is None or observation.confidence >= minimum:
            kept.append(observation)
            continue

        ox, oy, ow, oh = observation.box
        hand_enveloped = False
        for hand_index, hand in enumerate(hand_landmarks):
            covered_points = 0
            total_points = 0
            for x, y, _ in hand:
                px = float(x) * frame_width
                py = float(y) * frame_height
                total_points += 1
                if ox <= px <= ox + ow and oy <= py <= oy + oh:
                    covered_points += 1
            landmark_coverage = covered_points / max(1.0, float(total_points))
            builtin_label, builtin_score = (
                hand_gestures[hand_index]
                if hand_index < len(hand_gestures)
                else ("", 0.0)
            )
            builtin_bare_hand = (
                str(builtin_label).strip().casefold() in BARE_HAND_GESTURE_LABELS
                and float(builtin_score) >= 0.36
            )
            custom = classify_meme_gesture(
                (tuple(hand),),
                ((builtin_label, builtin_score),),
            )
            custom_bare_hand = (
                custom.gesture_id in BARE_HAND_MEME_GESTURES
                and custom.confidence >= 0.82
            )
            required_points = 8 if builtin_bare_hand else 15
            required_coverage = 0.40 if builtin_bare_hand else 0.65
            if (
                covered_points >= required_points
                and total_points >= 15
                and landmark_coverage >= required_coverage
                and (builtin_bare_hand or custom_bare_hand)
            ):
                hand_enveloped = True
                break
        if not hand_enveloped:
            kept.append(observation)
    return tuple(kept)


def merge_semantic_vision(
    base: VisionResult,
    semantic: Optional[SemanticPerceptionResult],
    now: float,
) -> VisionResult:
    """Merge an async result into the current fast face/hand observation."""

    if semantic is None:
        return base
    raw_objects = tuple(
        ObjectObservation(
            label=item.label,
            confidence=float(item.score),
            box=item.box.to_pixels(semantic.frame_width, semantic.frame_height),
        )
        for item in semantic.objects
    )
    gesture = max(semantic.gestures, key=lambda item: item.score, default=None)
    raw_hands = semantic.hands or semantic.gestures
    landmark_hands = tuple(item for item in raw_hands if item.landmarks)
    hand_landmarks = tuple(item.landmarks for item in landmark_hands)
    hand_gestures = tuple(
        (item.label, float(item.score)) for item in landmark_hands
    )
    objects = filter_hand_confused_objects(
        raw_objects,
        hand_landmarks,
        hand_gestures,
        (semantic.frame_width, semantic.frame_height),
    )
    semantic_base = replace(base, objects=objects)
    meme_detection = classify_meme_gesture(
        hand_landmarks,
        hand_gestures,
        face_box=base.face_box,
        frame_size=(semantic.frame_width, semantic.frame_height),
    )
    semantic_meme_id = meme_detection.gesture_id
    object_backend = "MEDIAPIPE_OBJECT" in semantic.backends
    gesture_backend = "MEDIAPIPE_GESTURE" in semantic.backends
    hand_candidates = []
    if base.meme_gesture:
        hand_candidates.append(
            (
                _reactor_priority(base.meme_gesture, "HAND"),
                base.meme_gesture,
                base.meme_gesture_confidence,
                base.meme_sample_timestamp,
                base.meme_sample_epoch,
            )
        )
    fresh_base_observations = []
    for base_signal, base_timestamp in (
        (base.meme_gesture, base.meme_sample_timestamp),
        (base.reactor_signal, base.reactor_sample_timestamp),
    ):
        if (
            base_timestamp >= 0.0
            and max(0.0, float(now) - float(base_timestamp))
            <= _MAX_DISAGREEING_SEMANTIC_AGE_SECONDS
        ):
            # Empty is intentional: a fresh neutral hand/reactor frame is a
            # disagreement with an old cached positive gesture.
            fresh_base_observations.append(str(base_signal or ""))
    semantic_disagreement_is_stale = bool(
        fresh_base_observations
        and semantic_meme_id not in set(fresh_base_observations)
        and max(0.0, float(now) - float(semantic.timestamp))
        > _MAX_DISAGREEING_SEMANTIC_AGE_SECONDS
    )
    if semantic_meme_id and not semantic_disagreement_is_stale:
        hand_candidates.append(
            (
                _reactor_priority(semantic_meme_id, "HAND"),
                semantic_meme_id,
                meme_detection.confidence,
                float(semantic.timestamp),
                int(semantic.epoch),
            )
        )
    if hand_candidates:
        # max() is stable: an equally specific fast same-frame classification
        # keeps its lower latency, while a classic async sign such as Victory
        # can replace a weak generic fast guess such as Fist.
        (
            _,
            hand_meme_id,
            hand_meme_confidence,
            hand_sample_timestamp,
            hand_sample_epoch,
        ) = _select_signal_candidate(hand_candidates, timestamp_index=3)
    else:
        hand_meme_id = ""
        hand_meme_confidence = 0.0
        # Looking and finding no gesture is still a fresh sample.  Retain the
        # newest neutral clock so release dwell can advance after a meme.
        neutral_hand_clocks = []
        if base.meme_sample_timestamp >= 0.0:
            neutral_hand_clocks.append(
                (base.meme_sample_timestamp, base.meme_sample_epoch)
            )
        if gesture_backend:
            neutral_hand_clocks.append(
                (float(semantic.timestamp), int(semantic.epoch))
            )
        if neutral_hand_clocks:
            hand_sample_timestamp, hand_sample_epoch = max(
                neutral_hand_clocks,
                key=lambda item: item[0],
            )
        else:
            hand_sample_timestamp = -1.0
            hand_sample_epoch = 0

    reactor_candidates = []
    if base.reactor_signal:
        reactor_candidates.append(
            (
                _reactor_priority(base.reactor_signal, base.reactor_channel),
                base.reactor_signal,
                base.reactor_confidence,
                base.reactor_channel,
                base.reactor_sample_timestamp,
                base.reactor_sample_epoch,
            )
        )
    if hand_meme_id:
        reactor_candidates.append(
            (
                _reactor_priority(hand_meme_id, "HAND"),
                hand_meme_id,
                hand_meme_confidence,
                "HAND",
                hand_sample_timestamp,
                hand_sample_epoch,
            )
        )
    if reactor_candidates:
        (
            _,
            reactor_signal,
            reactor_confidence,
            reactor_channel,
            reactor_sample_timestamp,
            reactor_sample_epoch,
        ) = _select_signal_candidate(reactor_candidates, timestamp_index=4)
    else:
        reactor_signal = ""
        reactor_confidence = 0.0
        reactor_channel = ""
        neutral_reactor_clocks = []
        if base.reactor_sample_timestamp >= 0.0:
            neutral_reactor_clocks.append(
                (base.reactor_sample_timestamp, base.reactor_sample_epoch)
            )
        if hand_sample_timestamp >= 0.0:
            neutral_reactor_clocks.append(
                (hand_sample_timestamp, hand_sample_epoch)
            )
        if neutral_reactor_clocks:
            reactor_sample_timestamp, reactor_sample_epoch = max(
                neutral_reactor_clocks,
                key=lambda item: item[0],
            )
        else:
            reactor_sample_timestamp = -1.0
            reactor_sample_epoch = 0
    backend_label = "+".join(
        name.replace("MEDIAPIPE_", "") for name in semantic.backends
    )
    return replace(
        semantic_base,
        gesture=gesture.label if gesture is not None else "",
        gesture_confidence=float(gesture.score) if gesture is not None else 0.0,
        gesture_sample_timestamp=float(semantic.timestamp),
        gesture_sample_epoch=int(semantic.epoch),
        hand_landmarks=hand_landmarks,
        hand_gestures=hand_gestures,
        meme_gesture=hand_meme_id,
        meme_gesture_confidence=hand_meme_confidence,
        meme_sample_timestamp=hand_sample_timestamp,
        meme_sample_epoch=hand_sample_epoch,
        reactor_signal=reactor_signal,
        reactor_confidence=reactor_confidence,
        reactor_channel=reactor_channel,
        reactor_sample_timestamp=reactor_sample_timestamp,
        reactor_sample_epoch=reactor_sample_epoch,
        reactor_detector_available=bool(
            base.reactor_detector_available or gesture_backend
        ),
        semantic_age_seconds=max(0.0, float(now) - semantic.timestamp),
        semantic_available=bool(semantic.backends),
        gesture_detector_available=gesture_backend,
        object_detector_available=object_backend,
        drink_near_mouth=(
            drink_object_near_mouth(semantic_base)
            if object_backend
            else None
        ),
        engine=(
            f"{base.engine} + {backend_label}"
            if backend_label
            else base.engine
        ),
    )


DRINK_LABELS = {"bottle", "cup", "wine glass", "can", "mug"}


def has_drink_object(vision: VisionResult) -> bool:
    return any(item.label.strip().casefold() in DRINK_LABELS for item in vision.objects)


def drink_object_near_mouth(vision: VisionResult) -> bool:
    """Require object geometry as well as the existing hand-to-mouth signal."""

    if (
        not vision.hand_near_mouth
        or vision.face_box is None
        or not vision.objects
    ):
        return False
    face_x, face_y, face_w, face_h = vision.face_box
    mouth_x = face_x + face_w * 0.50
    mouth_y = face_y + face_h * 0.70
    normalizer = max(1.0, float(max(face_w, face_h)))
    for item in vision.objects:
        if item.label.strip().casefold() not in DRINK_LABELS:
            continue
        x, y, w, h = item.box
        nearest_x = float(np.clip(mouth_x, x, x + w))
        nearest_y = float(np.clip(mouth_y, y, y + h))
        distance = float(np.hypot(nearest_x - mouth_x, nearest_y - mouth_y))
        if distance / normalizer <= 0.72:
            return True
    return False


def gesture_playground_owns_hand_to_mouth(
    vision: VisionResult,
    playground_active: bool,
    controller_reserves_hand: bool = True,
) -> bool:
    """Keep deliberate face gestures out of the snack/sip story detector."""

    # Visible drink evidence is stronger than the deliberately broad
        # folded-hand-near-chin heuristic used by the thinking reaction.
    if vision.drink_near_mouth is True or drink_object_near_mouth(vision):
        return False
    return bool(
        playground_active
        and controller_reserves_hand
        and vision.meme_gesture in {"thinking", "shhh", "cover_mouth"}
        and vision.meme_gesture_confidence >= 0.52
    )


def reactor_subject_present(vision: VisionResult) -> bool:
    """Gate a reaction on the detector that actually produced its signal."""

    face_present = (
        vision.face_box is not None and vision.face_confidence >= 0.35
    )
    body_present = (
        len(vision.pose_points) >= 17
        and sum(
            1
            for index in (11, 12, 13, 14, 15, 16, 23, 24)
            if index < len(vision.pose_visibility)
            and vision.pose_visibility[index] >= 0.55
        )
        >= 3
    )
    mesh_present = len(vision.face_mesh_points) >= 363
    if not vision.reactor_signal:
        # Establish dwell from any reliable local subject tracker before the
        # short gesture itself begins.
        return face_present or body_present or mesh_present
    if vision.reactor_channel == "FACE":
        # OPEN_MOUTH/HEAD_TILT already require valid FaceMesh geometry.  Do not
        # throw that evidence away merely because the separate box detector
        # missed a profile or strongly tilted face.
        return face_present or mesh_present or body_present
    if vision.reactor_channel == "POSE":
        return face_present or body_present or mesh_present
    # A hand signal can belong to a distant full-body subject whose hand and
    # skeleton track correctly while the separate face-box detector misses.
    return face_present or body_present or mesh_present


def contextualize_hand_event(event: ActionEvent, vision: VisionResult) -> ActionEvent:
    """Turn the legacy hand gesture into a small five-beat story."""

    if event.source != "HAND_TO_MOUTH" or event.kind not in ("sip", "crunch"):
        return event
    peak = event.timestamp - min(0.22, max(0.04, event.contact_seconds * 0.25))
    if event.kind == "sip":
        story_id = "smart_sip"
        focus = "drink" if has_drink_object(vision) else "hand"
        captions = (
            "SETUP // BEVERAGE ACQUIRED",
            "INTENT // TRAJECTORY LOCKED",
            "ACTION // SMART SIP",
            "REACTION // ZERO HESITATION",
            "PAYOFF // SIGMA HYDRATED",
        )
    else:
        story_id = "smart_crunch"
        focus = "hand"
        captions = (
            "SETUP // SNACK PROTOCOL",
            "INTENT // CRUNCH INCOMING",
            "ACTION // BITE CONFIRMED",
            "REACTION // AURA INCREASED",
            "PAYOFF // CRUMBS OF POWER",
        )
    return replace(
        event,
        kind=story_id,
        timestamp=peak,
        preset_id=event.kind,
        story_id=story_id,
        focus_label=focus,
        started_at=event.timestamp - event.contact_seconds,
        peak_at=peak,
        ended_at=event.timestamp,
        captions=captions,
        evidence=(f"contact={event.contact_seconds:.3f}",),
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous live-buffer confidence-booster meme camera"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument("--source", type=Path, help="Use a video file instead of a camera")
    parser.add_argument("--demo", action="store_true", help="Use an animated camera simulator")
    parser.add_argument(
        "--association-demo",
        action="store_true",
        help="With --demo, show a deterministic ASAHI object-to-meme reaction",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open a GUI window")
    parser.add_argument("--output", type=Path, help="Write the rendered HUD to MP4")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--auto-script", action="store_true", help="Enable timed demo events")
    parser.add_argument(
        "--script-preset",
        choices=(
            "sip",
            "crunch",
            "velocity",
            "mente",
            "clima",
            "bootleg",
            "atencao",
            "ela",
            "fruta",
            "nunca",
            "portal",
            "glass",
            "aspect_orbit",
            "fluid_inception",
            "wave_surge",
            "pulse_chain",
            "glide_chain",
            "match_flow",
            "arch_chain",
            "subject_boomerang",
            "cart_drift",
            "gigachad",
            "final_form",
        ),
        help="Fire one deterministic preset (useful for rehearsals and tests)",
    )
    parser.add_argument("--script-at", type=float, default=3.0)
    parser.add_argument("--no-auto", action="store_true", help="Disable gesture detection")
    parser.add_argument(
        "--mode",
        choices=("classic", "smart", "kiosk"),
        help="Play mode; defaults to config semantic.default_mode",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable gesture/object Tasks; keep fast face+hands+pose analysis",
    )
    parser.add_argument(
        "--no-qwen",
        action="store_true",
        help="Disable the optional text-only Mac Studio meme director",
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--mute", action="store_true")
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-edits",
        action="store_true",
        help="Auto-save each 9:16 mini-edit even in headless mode",
    )
    save_group.add_argument(
        "--no-save-edits",
        action="store_true",
        help="Disable automatic 9:16 edit export",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--loop-source", action="store_true")
    return parser.parse_args()


class SyntheticSource:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        association_demo: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.association_demo = bool(association_demo)
        self.frame_index = 0

    def read(self) -> Tuple[bool, np.ndarray, VisionResult]:
        now = self.frame_index / self.fps
        self.frame_index += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        gradient = np.linspace(0, 32, self.width, dtype=np.uint8)
        frame[:, :, 0] = 18 + gradient[None, :] // 3
        frame[:, :, 1] = 20 + gradient[None, :] // 5
        frame[:, :, 2] = 22 + gradient[None, :] // 7

        face_w, face_h = int(self.width * 0.19), int(self.height * 0.43)
        face_x = int(self.width * 0.38 + np.sin(now * 0.5) * 8)
        face_y = int(self.height * 0.18)
        center = (face_x + face_w // 2, face_y + face_h // 2)
        cv2.ellipse(frame, center, (face_w // 2, face_h // 2), 0, 0, 360, (122, 156, 187), -1)
        cv2.ellipse(frame, (center[0], face_y + 15), (face_w // 2 + 5, 40), 0, 180, 360, (25, 29, 33), -1)
        cv2.circle(frame, (center[0] - 25, center[1] - 22), 6, (20, 24, 25), -1)
        cv2.circle(frame, (center[0] + 25, center[1] - 22), 6, (20, 24, 25), -1)
        cv2.line(frame, (center[0] - 18, center[1] + 45), (center[0] + 22, center[1] + 45), (50, 62, 66), 3)
        cv2.rectangle(
            frame,
            (center[0] - 74, face_y + face_h - 3),
            (center[0] + 74, self.height),
            (22, 31, 37),
            -1,
        )

        cycle = now % 12.0
        sip_contact = 3.0 <= cycle <= 4.2
        crunch_contact = 8.0 <= cycle <= 8.45
        near = sip_contact or crunch_contact
        victory = 1.0 <= cycle <= 2.2
        meme_gesture = (
            "victory"
            if victory
            else (
                "open_palm"
                if 5.0 <= cycle <= 6.1
                else ("point" if 9.0 <= cycle <= 10.1 else "")
            )
        )
        pet_visible = 5.0 <= cycle <= 6.5
        if near:
            hand_center = (center[0] + 18, center[1] + 42)
        else:
            hand_center = (int(self.width * 0.72), int(self.height * 0.72))
        cv2.circle(frame, hand_center, 24, (105, 145, 178), -1, cv2.LINE_AA)
        cv2.line(frame, hand_center, (int(self.width * 0.72), self.height), (105, 145, 178), 20, cv2.LINE_AA)
        if sip_contact:
            bottle_box = (hand_center[0] - 18, hand_center[1] + 12, 36, 78)
            cv2.rectangle(
                frame,
                (bottle_box[0], bottle_box[1]),
                (bottle_box[0] + bottle_box[2], bottle_box[1] + bottle_box[3]),
                (58, 104, 72),
                -1,
            )
        else:
            bottle_box = None
        if pet_visible:
            cat_box = (int(self.width * 0.12), int(self.height * 0.61), 94, 118)
            cat_center = (cat_box[0] + 47, cat_box[1] + 58)
            cv2.ellipse(frame, cat_center, (42, 54), 0, 0, 360, (78, 92, 115), -1)
            cv2.circle(frame, (cat_center[0] - 14, cat_center[1] - 8), 4, (12, 15, 18), -1)
            cv2.circle(frame, (cat_center[0] + 14, cat_center[1] - 8), 4, (12, 15, 18), -1)
        else:
            cat_box = None
        demo_product_box = None
        if self.association_demo and 0.5 <= cycle <= 3.8:
            demo_product_box = _demo_product_box(self.width, self.height)
            px, py, pw, ph = demo_product_box
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (32, 32, 36), -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (220, 220, 225), 2)
            cv2.putText(
                frame,
                "ASAHI",
                (px + 4, py + int(ph * 0.52)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.42,
                (235, 235, 238),
                1,
                cv2.LINE_AA,
            )
        points = tuple(
            (
                hand_center[0] + int(np.cos(index / 21 * np.pi * 2) * 18),
                hand_center[1] + int(np.sin(index / 21 * np.pi * 2) * 18),
            )
            for index in range(21)
        )
        vision = VisionResult(
            face_box=(face_x, face_y, face_w, face_h),
            face_confidence=0.97,
            hand_points=points,
            hand_near_mouth=near,
            hand_mouth_distance=0.24 if near else 2.4,
            motion_score=0.08 + (0.12 if near else 0.0),
            engine="SCRIPTED SMART-STORY SIMULATOR",
            hand_box=(hand_center[0] - 28, hand_center[1] - 28, 56, 56),
            objects=tuple(
                item
                for item in (
                    ObjectObservation("bottle", 0.91, bottle_box) if bottle_box else None,
                    ObjectObservation("cat", 0.94, cat_box) if cat_box else None,
                    (
                        ObjectObservation("bottle", 0.98, demo_product_box)
                        if demo_product_box
                        else None
                    ),
                )
                if item is not None
            ),
            gesture="Victory" if victory else "",
            gesture_confidence=0.95 if victory else 0.0,
            gesture_sample_timestamp=now,
            gesture_sample_epoch=0,
            meme_gesture=meme_gesture,
            meme_gesture_confidence=0.95 if meme_gesture else 0.0,
            meme_sample_timestamp=now,
            meme_sample_epoch=0,
            glasses_score=0.16,
            semantic_available=True,
            gesture_detector_available=True,
            object_detector_available=True,
            drink_near_mouth=sip_contact,
            face_mesh_sample_timestamp=now,
        )
        return True, frame, vision


def open_capture(config: AppConfig, args: argparse.Namespace):
    if args.source is not None:
        capture = cv2.VideoCapture(str(args.source))
    else:
        camera_index = config.camera["index"] if args.camera is None else args.camera
        if sys.platform == "darwin":
            capture = cv2.VideoCapture(int(camera_index), cv2.CAP_AVFOUNDATION)
        else:
            capture = cv2.VideoCapture(int(camera_index))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(config.camera["capture_width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(config.camera["capture_height"]))
        capture.set(cv2.CAP_PROP_FPS, float(config.camera["target_fps"]))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def semantic_model_paths(
    semantic_config: Mapping[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> Tuple[Path, Optional[Path]]:
    """Resolve the always-on gesture model and optional advanced object model."""

    gesture_model = Path(
        semantic_config.get(
            "gesture_model",
            "assets/models/gesture_recognizer.task",
        )
    )
    if not gesture_model.is_absolute():
        gesture_model = project_root / gesture_model

    object_model: Optional[Path] = None
    if bool(semantic_config.get("object_detection_enabled", False)):
        object_model = Path(
            semantic_config.get(
                "object_model",
                "assets/models/efficientdet_lite0.tflite",
            )
        )
        if not object_model.is_absolute():
            object_model = project_root / object_model
    return gesture_model, object_model


def make_writer(path: Path, fps: float, size: Tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def window_render_size(
    window_title: str,
    fallback: Tuple[int, int],
    *,
    minimum: Tuple[int, int] = (320, 240),
) -> Tuple[int, int]:
    """Read the live OpenCV viewport without changing headless/export size."""

    try:
        _, _, width, height = cv2.getWindowImageRect(window_title)
    except (AttributeError, cv2.error):
        return fallback
    if width <= 0 or height <= 0:
        return fallback
    return max(minimum[0], int(width)), max(minimum[1], int(height))


def fit_writer_frame(
    frame: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Letterbox a responsive GUI frame into a fixed-size diagnostic video."""

    output_width, output_height = output_size
    if frame.shape[1] == output_width and frame.shape[0] == output_height:
        return frame
    source_height, source_width = frame.shape[:2]
    scale = min(
        output_width / max(1, source_width),
        output_height / max(1, source_height),
    )
    resized = cv2.resize(
        frame,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((output_height, output_width, 3), (8, 11, 12), dtype=np.uint8)
    x = (output_width - resized.shape[1]) // 2
    y = (output_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    camera_cfg = config.camera
    window_cfg = config.window
    analysis_cfg = config.analysis
    semantic_cfg = config.semantic
    gesture_cfg = config.gesture_playground
    director_cfg = config.director
    qwen_cfg = config.qwen_director
    association_cfg = config.association
    mode = str(args.mode or semantic_cfg.get("default_mode", "smart")).lower()
    output_size = (int(window_cfg["width"]), int(window_cfg["height"]))
    processing_size = (
        int(camera_cfg["processing_width"]),
        int(camera_cfg["processing_height"]),
    )
    target_fps = float(camera_cfg["target_fps"])
    analysis_stride = max(1, int(analysis_cfg.get("analysis_stride", 1)))
    minimum_buffer_seconds = max(
        2.2,
        min(
            float(camera_cfg["buffer_seconds"]),
            float(analysis_cfg.get("minimum_buffer_seconds", 8.0)),
        ),
    )
    mirror = bool(camera_cfg["mirror"]) and not args.no_mirror
    source_is_file = args.source is not None
    synthetic = (
        SyntheticSource(
            *processing_size,
            target_fps,
            association_demo=args.association_demo,
        )
        if args.demo or args.association_demo
        else None
    )
    capture = None if synthetic is not None else open_capture(config, args)

    if capture is not None and not capture.isOpened():
        message = f"Could not open {'video ' + str(args.source) if source_is_file else 'camera'}"
        print(message, file=sys.stderr)
        if not args.headless:
            screen = error_screen(message, output_size)
            cv2.imshow(window_cfg["title"], screen)
            while True:
                if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                    break
            cv2.destroyAllWindows()
        return 2

    source_fps = target_fps
    if capture is not None and source_is_file:
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if 1.0 <= detected_fps <= 240.0:
            source_fps = detected_fps

    ring = FrameRingBuffer(float(camera_cfg["buffer_seconds"]))
    analyzer = None
    semantic_worker: Optional[AsyncSemanticPerception] = None
    association_worker: Optional[AsyncAppleVisionSemantics] = None
    qwen_worker: Optional[AsyncQwenMemeDirector] = None
    warning = None
    if synthetic is None:
        analyzer, warning = create_vision_analyzer(
            float(analysis_cfg["hand_to_mouth_threshold"]),
            pose_mimic_enabled=bool(
                analysis_cfg.get("pose_mimic_enabled", True)
            ),
            body_pose_enabled=bool(
                analysis_cfg.get("body_pose_enabled", False)
            ),
        )
    hands_available = synthetic is not None or bool(getattr(analyzer, "has_hands", False))
    if (
        synthetic is None
        and not args.no_semantic
        and bool(semantic_cfg.get("enabled", True))
    ):
        gesture_model, object_model = semantic_model_paths(
            semantic_cfg,
            PROJECT_ROOT,
        )
        semantic_worker = AsyncSemanticPerception(
            gesture_model_path=gesture_model,
            object_model_path=object_model,
            object_score_threshold=float(
                semantic_cfg.get("object_score_threshold", 0.42)
            ),
            gesture_score_threshold=float(
                semantic_cfg.get("gesture_score_threshold", 0.55)
            ),
            object_allowlist=semantic_cfg.get("object_allowlist", ()),
            input_max_side=480,
            startup_wait_seconds=0.45,
        )
    association_profiles = build_association_profiles(association_cfg)
    association_enabled = bool(association_cfg.get("enabled", False))
    association_detector = AssociationDetector(
        association_profiles,
        global_cooldown_seconds=float(
            association_cfg.get("global_cooldown_seconds", 0.75)
        ),
    )
    if (
        synthetic is None
        and association_enabled
        and association_profiles
    ):
        helper_source = Path(
            association_cfg.get("helper_source", "tools/vision_semantics.swift")
        )
        helper_binary = Path(
            association_cfg.get("helper_binary", "runtime/tools/vision_semantics")
        )
        if not helper_source.is_absolute():
            helper_source = PROJECT_ROOT / helper_source
        if not helper_binary.is_absolute():
            helper_binary = PROJECT_ROOT / helper_binary
        association_worker = AsyncAppleVisionSemantics(
            helper_source_path=helper_source,
            helper_binary_path=helper_binary,
            startup_wait_seconds=0.15,
            max_regions=5,
        )
    qwen_packs = tuple(
        QwenMemePack.from_mapping(item)
        for item in qwen_cfg.get("packs", ())
    )
    qwen_pack_by_id = {pack.pack_id: pack for pack in qwen_packs}
    qwen_relay = Path(
        qwen_cfg.get(
            "relay_command",
            "",
        )
    ).expanduser()
    if (
        bool(qwen_cfg.get("enabled", False))
        and not args.no_qwen
        and qwen_packs
    ):
        qwen_worker = AsyncQwenMemeDirector(
            relay_command=qwen_relay,
            enabled=True,
            timeout_seconds=float(qwen_cfg.get("timeout_seconds", 12.0)),
            minimum_interval_seconds=float(
                qwen_cfg.get("minimum_interval_seconds", 15.0)
            ),
            cache_ttl_seconds=float(qwen_cfg.get("cache_ttl_seconds", 45.0)),
        )
        if mode == "classic":
            qwen_worker.pause()
    qwen_context = QwenContextMemory(
        float(qwen_cfg.get("context_window_seconds", 9.0))
    )
    director_options = DirectorOptions.from_mapping(
        director_cfg.get("default_options", {})
    )
    if args.association_demo:
        # A focused deterministic rehearsal; the normal app keeps the user's
        # configured random mode and all five independent checkboxes.
        director_options = DirectorOptions(
            action_stories=False,
            object_memes=True,
            freestyle_edits=False,
            culture_themes=True,
            random_per_cycle=False,
        )
    configured_director_seed = int(director_cfg.get("seed", 90517))
    deterministic_session = synthetic is not None or source_is_file
    session_seed = (
        configured_director_seed
        if deterministic_session
        else random.SystemRandom().getrandbits(64)
    )
    first_preset_rng = random.Random(session_seed ^ 0x5045525045545541)
    randomize_first_live_preset = not deterministic_session
    director = DirectorCoordinator(
        director_options,
        route_timeout_seconds=float(
            director_cfg.get("route_timeout_seconds", 12.0)
        ),
        freestyle_delay_seconds=float(
            director_cfg.get("freestyle_delay_seconds", 4.0)
        ),
        seed=session_seed,
    )
    director.begin_cycle(
        0.0,
        object_available=bool(association_worker is not None or args.association_demo),
    )
    detector = ActionEventDetector(
        analysis_cfg, auto_enabled=bool(analysis_cfg["auto_enabled"]) and not args.no_auto
    )
    story_detector = SemanticStoryDetector(thresholds=semantic_cfg)
    gesture_controller = GestureMemeController.from_mapping(
        gesture_cfg,
        PROJECT_ROOT,
    )
    kiosk_gate = KioskGate(
        float(semantic_cfg.get("kiosk_face_dwell_seconds", 0.9)),
        float(semantic_cfg.get("kiosk_gesture_hold_seconds", 0.45)),
        float(semantic_cfg.get("kiosk_subject_gap_seconds", 0.35)),
    )
    editor = EditController(float(analysis_cfg["post_roll_seconds"]))
    logs = EventLog()
    runtime_dir = PROJECT_ROOT / "runtime"
    sound = SoundEngine(
        PROJECT_ROOT,
        runtime_dir,
        config.presets,
        analyzer_enabled=bool(config.audio["analyzer_enabled"]) and not args.mute,
        edits_enabled=bool(config.audio["edits_enabled"]) and not args.mute,
        analyzer_volume=float(config.audio.get("analyzer_volume", 0.08)),
        edit_volume=float(config.audio.get("edit_volume", 0.82)),
    )
    archive_enabled = (
        not args.no_save_edits
        and (not args.headless or args.save_edits)
    )
    archive = EditArchive(
        PROJECT_ROOT,
        runtime_dir,
        target_fps,
        enabled=archive_enabled,
        audio_volume=float(config.audio.get("edit_volume", 0.82)),
    )
    saved_edit_paths = []
    last_scheduled_track_id: Optional[str] = None
    recent_preset_ids = deque(maxlen=4)
    last_event_at: Optional[float] = None

    def schedule_event(event: ActionEvent) -> bool:
        nonlocal last_event_at, last_scheduled_track_id
        if not editor.accepts_trigger:
            logs.add("EVENT_DROPPED reason=capture_cycle_busy")
            return False
        eligible_presets = config.presets
        if event.preset_pool:
            eligible_presets = tuple(
                config.preset(preset_id) for preset_id in event.preset_pool
            )
        requested = config.preset(event.requested_preset_id)
        if requested not in eligible_presets:
            requested = eligible_presets[0]
            event = replace(event, preset_id=requested.id)
        if (
            randomize_first_live_preset
            and last_scheduled_track_id is None
            and event.source != "KEYBOARD"
        ):
            opening = choose_session_first_preset(
                eligible_presets,
                first_preset_rng,
            )
            if opening is not None:
                requested = opening
                event = replace(
                    event,
                    kind=event.kind if event.is_story else opening.id,
                    preset_id=opening.id,
                )
                logs.add(f"SESSION_SHUFFLE opening={opening.title}")
        if (
            event.is_story
            or event.source in ("HAND_TO_MOUTH", "TIMED_DEMO")
            or event.source.startswith("SEMANTIC_")
        ):
            selected = choose_varied_preset(
                requested,
                eligible_presets,
                last_scheduled_track_id,
                tuple(recent_preset_ids),
            )
        else:
            selected = choose_non_repeating_preset(
                requested,
                eligible_presets,
                last_scheduled_track_id,
            )
        if selected is None:
            logs.add("TRACK_GUARD rejected=no_distinct_track")
            return False
        if selected.id != requested.id:
            guard = (
                "TRACK_GUARD"
                if requested.track_id == last_scheduled_track_id
                else "VARIETY_GUARD"
            )
            logs.add(
                f"{guard} requested={requested.title} substituted={selected.title}"
            )
            event = replace(
                event,
                kind=event.kind if event.is_story else selected.id,
                preset_id=selected.id,
                source=f"{event.source}+{guard}",
            )
        if not editor.request(event, selected):
            logs.add("EVENT_DROPPED reason=queue_full")
            return False
        # A trigger owns the capture cycle immediately, including its clean
        # +1 s reaction tail. Stop passive reactions before the replay exists
        # so its viewer cannot become source material for another edit.
        gesture_controller.freeze()
        last_scheduled_track_id = selected.track_id
        recent_preset_ids.append(selected.id)
        last_event_at = event.timestamp
        logs.add(
            f"EVENT_LOCKED type={selected.title} source={event.source} confidence={event.confidence:.2f}"
        )
        if event.is_story:
            logs.add(
                f"STORY_LOCKED kind={event.story_id} focus={event.focus_label or 'subject'}"
            )
        sound.play_analyzer("locked")
        return True

    engine_name = (
        "SIMULATOR"
        if synthetic is not None
        else getattr(analyzer, "engine_name", "VISION")
    )
    if warning:
        print(warning, file=sys.stderr)
        if bool(getattr(analyzer, "has_hands", False)):
            logs.add(f"ANALYZER_DEGRADED engine={engine_name}")
        else:
            logs.add("VISION_FALLBACK active=face+timed-demo")
    else:
        logs.add(f"ANALYZER_ONLINE engine={engine_name}")
    semantic_warning = ""
    if synthetic is not None:
        logs.add("SEMANTIC_MODE source=scripted-local")
    elif semantic_worker is not None:
        semantic_status = semantic_worker.status
        semantic_warning = "; ".join(semantic_status.warnings)
        logs.add(
            "SEMANTIC_MODE gesture="
            + str(semantic_status.gesture_available).lower()
            + " objects="
            + str(semantic_status.object_available).lower()
        )
    else:
        semantic_warning = "semantic models disabled"
        logs.add("SEMANTIC_MODE fallback=face+hands+pose")
    logs.add(f"PLAY_MODE mode={mode}")
    if args.association_demo:
        association_state = "SCRIPTED LOCAL"
        logs.add("CONTEXT_AI source=scripted-asahi")
    elif association_worker is not None:
        association_state = "APPLE VISION STARTING"
        logs.add("CONTEXT_AI local=true backend=apple-vision")
    elif association_enabled and association_profiles:
        association_state = "OBJECT LABELS ONLY"
        logs.add("CONTEXT_AI backend=generic-objects")
    else:
        association_state = "OFF"
    if qwen_worker is None:
        qwen_state = "OFF"
        logs.add("QWEN_DIRECTOR enabled=false")
    elif qwen_worker.available:
        qwen_state = "READY"
        logs.add("QWEN_DIRECTOR text_tokens_only=true frames=false")
    else:
        qwen_state = "FALLBACK LOCAL"
        logs.add("QWEN_DIRECTOR relay=unavailable fallback=local")
    logs.add(f"DIRECTOR route={director.route}")
    logs.add("RING_BUFFER rolling=true local_only=true")
    logs.add("EDIT_SOURCE live_buffer_only=true")
    if archive_enabled:
        logs.add(
            "AUTO_EXPORT 9x16=true audio_mux="
            + str(archive.audio_mux_available).lower()
        )

    logical_now = 0.0
    if args.auto_script or not hands_available:
        detector.set_timer_mode(True, logical_now)
        logs.add("TIMED_DEMO armed=true")

    writer = make_writer(args.output, source_fps, output_size) if args.output else None
    manual_writer = None
    manual_writer_path: Optional[Path] = None
    window_title = str(window_cfg["title"])
    fullscreen = bool(window_cfg["fullscreen"]) or args.fullscreen
    hud_enabled = True
    motion_history = deque(maxlen=90)
    fps_samples = deque(maxlen=45)
    wall_started = time.monotonic()
    prior_wall = wall_started
    frame_index = 0
    previous_detector_state = detector.state
    last_vision: Optional[VisionResult] = None
    last_frame_wall = wall_started
    next_file_deadline = wall_started
    scripted_event_sent = False
    semantic_interval = 1.0 / max(
        0.5,
        float(semantic_cfg.get("semantic_fps", 5.0)),
    )
    next_semantic_submit = 0.0
    semantic_ttl = float(semantic_cfg.get("result_ttl_seconds", 0.75))
    association_interval = 1.0 / max(
        0.25,
        float(association_cfg.get("analysis_fps", 1.25)),
    )
    association_ttl = float(association_cfg.get("result_ttl_seconds", 2.5))
    association_popup_seconds = float(
        association_cfg.get("popup_duration_seconds", 3.4)
    )
    next_association_submit = 0.0
    association_epoch = (
        association_worker.current_epoch if association_worker is not None else 0
    )
    association_overlay: Optional[LiveMemeOverlay] = None
    association_result: Optional[AppleVisionResult] = None
    qwen_epoch = qwen_worker.current_epoch if qwen_worker is not None else 0
    qwen_result_ttl = float(qwen_cfg.get("result_ttl_seconds", 12.0))
    qwen_minimum_signals = int(qwen_cfg.get("minimum_distinct_signals", 2))
    qwen_object_min_confidence = float(
        qwen_cfg.get("object_min_confidence", 0.62)
    )
    qwen_object_sample_interval = max(
        0.25,
        float(qwen_cfg.get("object_sample_interval_seconds", 0.9)),
    )
    next_qwen_object_sample = 0.0
    last_qwen_direction_fingerprint: Optional[str] = None
    ui_actions = deque()
    story_state = "CLASSIC"
    gesture_meme_view: Optional[GestureMemeView] = None
    kiosk_state = kiosk_gate.state if mode == "kiosk" else "OFF"
    display_size = output_size

    if not args.headless:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, output_size[0], output_size[1])

        def on_mouse(event: int, x: int, y: int, flags: int, data: Any) -> None:
            del flags, data
            if event != cv2.EVENT_LBUTTONUP:
                return
            option = hit_test_director_control(x, y, display_size)
            if option is not None:
                ui_actions.append(option)

        cv2.setMouseCallback(window_title, on_mouse)
        if fullscreen:
            cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            current_wall = time.monotonic()
            if synthetic is not None:
                ok, frame, source_vision = synthetic.read()
                logical_now = frame_index / source_fps
                vision = (
                    VisionResult(engine="ANALYSIS PAUSED")
                    if editor.capture_paused
                    else source_vision
                )
            else:
                ok, raw = capture.read()
                if not ok:
                    if source_is_file and args.loop_source:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame = cv2.resize(raw, processing_size, interpolation=cv2.INTER_AREA)
                if mirror:
                    frame = cv2.flip(frame, 1)
                if source_is_file:
                    logical_now = frame_index / source_fps
                else:
                    logical_now = current_wall - wall_started
                if editor.capture_paused:
                    vision = VisionResult(engine="ANALYSIS PAUSED")
                else:
                    if last_vision is None or frame_index % analysis_stride == 0:
                        analyzed_vision = analyzer.process(frame)
                        last_vision = replace(
                            analyzed_vision,
                            meme_sample_timestamp=logical_now,
                            reactor_sample_timestamp=logical_now,
                            face_mesh_sample_timestamp=(
                                logical_now
                                if len(analyzed_vision.face_mesh_points) >= 468
                                else -1.0
                            ),
                        )
                    vision = last_vision
                    if semantic_worker is not None and mode in ("smart", "kiosk"):
                        if logical_now + 1e-9 >= next_semantic_submit:
                            semantic_worker.submit(frame, logical_now)
                            next_semantic_submit = logical_now + semantic_interval
                        vision = merge_semantic_vision(
                            vision,
                            semantic_worker.latest(
                                now_timestamp=logical_now,
                                max_age_seconds=semantic_ttl,
                            ),
                            logical_now,
                        )

            # Runtime circuit breakers can disable Hands without stopping the
            # camera. Keep detector fallback state current instead of relying
            # on the startup snapshot.
            hands_available = synthetic is not None or bool(
                getattr(analyzer, "has_hands", False)
            )

            if association_overlay is not None and logical_now >= association_overlay.expires_at:
                association_overlay = None
                sound.stop_association()
                association_state = (
                    "SCRIPTED LOCAL"
                    if args.association_demo
                    else (
                        "OCR + OBJECTS LOCAL"
                        if association_worker is not None and association_worker.available
                        else "OBJECT LABELS ONLY"
                    )
                )

            if (
                not editor.capture_paused
                and director_options.object_memes
                and logical_now + 1e-9 >= next_association_submit
            ):
                if args.association_demo:
                    association_result = scripted_association_result(
                        logical_now,
                        association_epoch,
                        frame.shape[1],
                        frame.shape[0],
                    )
                elif association_worker is not None:
                    object_regions = tuple(
                        {
                            "box": item.box,
                            "label": item.label,
                            "kind": "object",
                        }
                        for item in vision.objects
                        if _text_key(item.label) not in {"person", "human", "face"}
                    )
                    hand_regions = (
                        ({"box": vision.hand_box, "kind": "hand", "label": "hand"},)
                        if vision.hand_box is not None
                        else ()
                    )
                    association_worker.submit(
                        frame,
                        logical_now,
                        object_regions=object_regions,
                        hand_regions=hand_regions,
                        epoch=association_epoch,
                    )
                next_association_submit = logical_now + association_interval
            if association_worker is not None:
                association_result = association_worker.latest(
                    epoch=association_epoch,
                    now_timestamp=logical_now,
                    max_age_seconds=association_ttl,
                )
                apple_status = association_worker.status
                if apple_status.available:
                    association_state = "OCR + OBJECTS LOCAL"
                elif apple_status.ready:
                    association_state = "OBJECT LABELS ONLY"
                else:
                    association_state = "APPLE VISION STARTING"

            event = None
            # The live preview remains visible, but frames are admitted to the
            # rolling memory only before an edit starts.  COLLECTING_TAIL still
            # records the configured +1 s reaction; ASSEMBLING and PLAYING do
            # not record or run action detection.
            admit_frames = not editor.capture_paused
            if mode == "kiosk":
                kiosk_update = kiosk_gate.update(
                    logical_now,
                    vision,
                    replaying=editor.capture_paused,
                )
                kiosk_state = kiosk_update.state
                admit_frames = admit_frames and kiosk_update.should_record
                if (
                    kiosk_update.state == "ACTIVE"
                    and not kiosk_update.should_record
                    and not kiosk_update.deactivated
                ):
                    # A tracker gap pauses recording and must also break every
                    # temporal hold; otherwise the blind interval would count
                    # as an observed hand/pose/story duration.
                    detector.cancel_candidate()
                    previous_detector_state = detector.state
                    story_detector.cancel_candidates()
                    gesture_controller.cancel_candidate()
                if kiosk_update.activated:
                    ring.clear()
                    motion_history.clear()
                    detector.cancel_candidate()
                    previous_detector_state = detector.state
                    story_detector.reset()
                    gesture_controller.reset(logical_now, require_release=True)
                    gesture_meme_view = None
                    last_event_at = None
                    logs.add("KIOSK_OPT_IN accepted=true capture=started")
                    sound.play_analyzer("done")
                elif kiosk_update.deactivated:
                    # A confirmed break ends the visible kiosk session. Discard
                    # its unfinished buffer and require a new opt-in.
                    ring.clear()
                    motion_history.clear()
                    aborted_edit = editor.abort_capture_cycle()
                    detector.cancel_candidate()
                    previous_detector_state = detector.state
                    story_detector.reset()
                    gesture_controller.reset(logical_now, require_release=True)
                    gesture_meme_view = None
                    last_event_at = None
                    last_vision = None
                    association_overlay = None
                    sound.stop_association()
                    qwen_context.reset()
                    if qwen_worker is not None:
                        qwen_epoch = qwen_worker.reset_epoch()
                        if mode == "classic":
                            qwen_worker.pause()
                        else:
                            qwen_worker.resume()
                    last_qwen_direction_fingerprint = None
                    next_qwen_object_sample = logical_now
                    if semantic_worker is not None:
                        semantic_worker.reset_epoch()
                        next_semantic_submit = logical_now
                    if association_worker is not None:
                        association_epoch = association_worker.reset_epoch()
                    else:
                        association_epoch += 1
                    association_result = None
                    association_detector.reset()
                    next_association_submit = logical_now
                    director.begin_cycle(
                        logical_now,
                        object_available=bool(
                            args.association_demo
                            or association_worker is not None
                            or vision.object_detector_available
                            or vision.objects
                        ),
                    )
                    logs.add("KIOSK_SESSION ended=subject_left buffer=discarded")
                    if aborted_edit:
                        logs.add("EDIT_ABORTED reason=kiosk_subject_left")
                    logs.add(f"DIRECTOR route={director.route}")
            else:
                kiosk_state = "OFF"

            if not admit_frames:
                gesture_meme_view = None

            if admit_frames:
                object_available = bool(
                    args.association_demo
                    or association_worker is not None
                    or vision.object_detector_available
                    or vision.objects
                )
                if (
                    not ring.ready(minimum_buffer_seconds)
                    and director.route
                    in (
                        "ACTION_EDIT",
                        "OBJECT_EDIT",
                        "FREESTYLE_EDIT",
                        "MIXED",
                    )
                ):
                    # A route that needs recorded history must not expire while
                    # the fresh replay-safe buffer is still warming up.
                    director.route_started_at = logical_now
                else:
                    director.tick(logical_now, object_available=object_available)
                association_match = None
                association_response = director.association_response()
                if association_response is not None:
                    association_match = association_detector.update(
                        logical_now,
                        association_result,
                        object_labels=vision.objects,
                    )
                if association_match is not None:
                    qwen_context.observe(
                        logical_now,
                        "context",
                        association_match.profile_id,
                        association_match.confidence,
                    )
                    payload = dict(association_match.payload)
                    is_culture = bool(payload.get("culture_theme", False))
                    if is_culture and not director_options.culture_themes:
                        logs.add(
                            f"CONTEXT_SKIPPED profile={association_match.profile_id} reason=culture_off"
                        )
                        director.complete_popup(logical_now, object_available)
                    else:
                        focus_box = association_focus_box(
                            association_match,
                            association_result,
                            vision,
                        )
                        focus_label = str(payload.get("focus_label", "object"))
                        if focus_box is not None:
                            vision = replace(
                                vision,
                                objects=vision.objects
                                + (
                                    ObjectObservation(
                                        focus_label,
                                        association_match.confidence,
                                        focus_box,
                                    ),
                                ),
                            )
                        evidence_text = (
                            association_match.matched_ocr_terms[0]
                            if association_match.matched_ocr_terms
                            else (
                                association_match.matched_object_labels[0]
                                if association_match.matched_object_labels
                                else "VISIBLE OBJECT"
                            )
                        )
                        if (
                            association_response == "edit"
                            and ring.ready(minimum_buffer_seconds)
                        ):
                            event = association_event(association_match, focus_box)
                            director.lock_for_edit()
                            association_state = "EDIT " + association_match.profile_id.upper()
                            logs.add(
                                f"CONTEXT_EDIT profile={association_match.profile_id} evidence={evidence_text}"
                            )
                        else:
                            association_overlay = LiveMemeOverlay(
                                profile_id=association_match.profile_id,
                                title=str(payload.get("title", association_match.title)),
                                subtitle=str(payload.get("subtitle", "VISIBLE MATCH")),
                                evidence=str(evidence_text),
                                confidence=association_match.confidence,
                                accent_bgr=tuple(
                                    int(value)
                                    for value in payload.get("accent_bgr", (244, 223, 65))
                                ),
                                sound_style=str(payload.get("sound_style", "beverage")),
                                started_at=logical_now,
                                expires_at=logical_now + association_popup_seconds,
                                focus_box=focus_box,
                            )
                            sound.play_association(
                                str(payload.get("sound_style", "beverage"))
                            )
                            association_state = "POPUP " + association_match.profile_id.upper()
                            logs.add(
                                f"CONTEXT_POPUP profile={association_match.profile_id} evidence={evidence_text}"
                            )
                            director.complete_popup(logical_now, object_available)
                playground_active = (
                    gesture_controller.enabled
                    and event is None
                    and editor.accepts_trigger
                    and mode in ("smart", "kiosk")
                    and (mode != "kiosk" or kiosk_state == "ACTIVE")
                )
                playground_backend_available = bool(
                    vision.reactor_detector_available
                    or vision.gesture_detector_available
                    or hands_available
                    or synthetic is not None
                )
                (
                    reactor_gesture_id,
                    reactor_gesture_confidence,
                    reactor_sample_epoch,
                    reactor_sample_timestamp,
                    reactor_sample_channel,
                ) = gesture_controller_sample(vision)
                playground_update = gesture_controller.observe(
                    logical_now,
                    gesture_id=reactor_gesture_id,
                    confidence=reactor_gesture_confidence,
                    # For pose reactions this argument means "subject present";
                    # distant full-body references often have reliable torso
                    # landmarks even when the face detector has no face box.
                    face_present=reactor_subject_present(vision),
                    sample_epoch=reactor_sample_epoch,
                    sample_timestamp=reactor_sample_timestamp,
                    active=playground_active,
                    backend_available=playground_backend_available,
                    blocked=association_overlay is not None,
                )
                gesture_meme_view = playground_update.view
                accepted_meme_gesture = ""
                accepted_meme_confidence = 0.0
                if playground_update.hit is not None:
                    hit = playground_update.hit
                    accepted_meme_gesture = hit.gesture_id
                    accepted_meme_confidence = hit.confidence
                    qwen_context.observe(
                        logical_now,
                        "gesture",
                        hit.gesture_id,
                        hit.confidence,
                    )
                    logs.add(
                        "GESTURE_MEME "
                        f"signal={hit.gesture_id} "
                        f"channel={reactor_sample_channel or 'HAND'} "
                        f"moments={gesture_controller.unique_count}/{gesture_controller.unique_goal}"
                    )
                if logical_now + 1e-9 >= next_qwen_object_sample:
                    for item in vision.objects:
                        normalized_object = _text_key(item.label)
                        if (
                            item.confidence >= qwen_object_min_confidence
                            and normalized_object
                            not in {"person", "human", "face", "hand"}
                        ):
                            qwen_context.observe(
                                logical_now,
                                "object",
                                normalized_object,
                                item.confidence,
                            )
                    next_qwen_object_sample = (
                        logical_now + qwen_object_sample_interval
                    )
                face_mesh_age = (
                    logical_now - vision.face_mesh_sample_timestamp
                    if vision.face_mesh_sample_timestamp >= 0.0
                    else float("inf")
                )
                # Store a mesh only beside the exact frame that was analyzed.
                # GigaChad sessions later prefer this coherent 12 FPS subset;
                # cached geometry is never glued onto a newer moving JPEG.
                mesh_is_current = (
                    len(vision.face_mesh_points) >= 468
                    and -1e-6 <= face_mesh_age <= 1.0 / max(2.0, source_fps * 2.0)
                )
                ring.append(
                    logical_now,
                    frame,
                    vision.face_box,
                    motion_score=vision.motion_score,
                    hand_near_mouth=vision.hand_near_mouth,
                    face_confidence=vision.face_confidence,
                    hand_box=vision.hand_box,
                    objects=vision.objects,
                    gesture=vision.gesture,
                    gesture_confidence=vision.gesture_confidence,
                    # Async hit metadata is attached below to the historical
                    # sample frame rather than this later delivery frame.
                    meme_gesture="",
                    meme_gesture_confidence=0.0,
                    hand_near_eyes=vision.hand_near_eyes,
                    glasses_score=vision.glasses_score,
                    semantic_available=vision.semantic_available,
                    gesture_detector_available=vision.gesture_detector_available,
                    object_detector_available=vision.object_detector_available,
                    drink_near_mouth=vision.drink_near_mouth,
                    face_mesh_points=(
                        vision.face_mesh_points if mesh_is_current else ()
                    ),
                    face_mesh_sample_timestamp=(
                        vision.face_mesh_sample_timestamp
                        if mesh_is_current
                        else -1.0
                    ),
                )
                if playground_update.hit is not None:
                    # Only a stabilized hit becomes montage metadata. Raw
                    # one-frame guesses remain tracker-only and cannot steal
                    # a key-moment slot. The JPEG itself is never changed.
                    ring.annotate_meme_gesture(
                        playground_update.hit.timestamp,
                        accepted_meme_gesture,
                        accepted_meme_confidence,
                    )
                if (
                    qwen_worker is not None
                    and qwen_worker.available
                    and mode in ("smart", "kiosk")
                ):
                    qwen_snapshot = qwen_context.snapshot(
                        logical_now,
                        qwen_epoch,
                        qwen_packs,
                        minimum_distinct_signals=qwen_minimum_signals,
                    )
                    if qwen_snapshot is not None and qwen_worker.submit(qwen_snapshot):
                        qwen_state = "QUEUED"
                        logs.add(
                            "QWEN_CONTEXT submitted="
                            f"{len(qwen_snapshot.signals)} frames=false"
                        )
                motion_history.append(vision.motion_score)

                # Once an action owns the current capture cycle, do not allow
                # another gesture or timer event to queue during its tail.
                if editor.accepts_trigger:
                    buffer_ready = ring.ready(minimum_buffer_seconds)
                    if (
                        event is None
                        and director.action_enabled()
                        and mode in ("smart", "kiosk")
                        and detector.auto_enabled
                    ):
                        # Do not arm/latch a semantic story until the rolling
                        # memory can actually service it.  Otherwise an early
                        # gesture can be consumed during warm-up and never
                        # become an edit when the buffer reaches 20 seconds.
                        story_vision = (
                            vision
                            if (
                                not gesture_controller.enabled
                                or not playground_active
                                or not playground_backend_available
                                or gesture_controller.ready_for_edit
                            )
                            else replace(
                                vision,
                                gesture="",
                                gesture_confidence=0.0,
                            )
                        )
                        thinking_owns_hand = gesture_playground_owns_hand_to_mouth(
                            vision,
                            playground_active,
                            gesture_controller.reserves_hand_to_mouth,
                        )
                        if thinking_owns_hand:
                            story_vision = replace(
                                story_vision,
                                hand_near_mouth=False,
                                drink_near_mouth=False,
                            )
                        semantic_event = (
                            story_detector.update(logical_now, story_vision)
                            if buffer_ready
                            else None
                        )
                        if semantic_event is not None:
                            story_state = "LOCK " + semantic_event.semantic_kind.upper()
                            event = semantic_event.to_action_event()
                        elif not buffer_ready:
                            story_state = (
                                "GESTURE MOMENTS "
                                f"{gesture_controller.unique_count}/{gesture_controller.unique_goal}"
                                if gesture_meme_view is not None
                                else "SEMANTIC WARMUP"
                            )
                        elif gesture_controller.ready_for_edit:
                            story_state = "GESTURE EDIT ARMED"
                        elif vision.gesture and vision.gesture != "None":
                            story_state = "GESTURE " + vision.gesture.upper()
                        elif vision.objects:
                            story_state = "TRACK " + vision.objects[0].label.upper()
                        else:
                            story_state = "SEMANTIC SCAN"
                    elif event is not None:
                        story_state = "CONTEXT EDIT"
                    elif not director.action_enabled():
                        story_state = "DIRECTOR " + director.route
                    elif mode == "classic":
                        story_state = "CLASSIC"
                    else:
                        story_state = "PAUSED"

                    if event is None and director.action_enabled():
                        thinking_owns_hand = gesture_playground_owns_hand_to_mouth(
                            vision,
                            playground_active,
                            gesture_controller.reserves_hand_to_mouth,
                        )
                        semantic_drink_owns_signal = (
                            mode in ("smart", "kiosk")
                            and vision.object_detector_available
                            and vision.drink_near_mouth is True
                        )
                        if thinking_owns_hand:
                            detector.cancel_candidate()
                        elif not semantic_drink_owns_signal:
                            event = detector.update(
                                logical_now,
                                vision,
                                buffer_ready,
                                hands_available,
                            )
                            if event is not None and mode in ("smart", "kiosk"):
                                event = contextualize_hand_event(event, vision)
                    if qwen_worker is not None and mode in ("smart", "kiosk"):
                        qwen_direction = qwen_worker.latest(
                            epoch=qwen_epoch,
                            now_timestamp=logical_now,
                            max_age_seconds=qwen_result_ttl,
                        )
                        if (
                            qwen_direction is not None
                            and qwen_direction.context_fingerprint
                            != last_qwen_direction_fingerprint
                        ):
                            qwen_pack = qwen_pack_by_id.get(qwen_direction.pack_id)
                            if event is not None:
                                # Fast deterministic stories always win the
                                # frame.  A late model answer cannot replace a
                                # locally observed action.
                                last_qwen_direction_fingerprint = (
                                    qwen_direction.context_fingerprint
                                )
                                logs.add("QWEN_DIRECTION skipped=local_event_priority")
                            elif qwen_pack is None:
                                last_qwen_direction_fingerprint = (
                                    qwen_direction.context_fingerprint
                                )
                                logs.add("QWEN_DIRECTION rejected=unknown_pack")
                            elif association_overlay is None:
                                evidence_label = " + ".join(
                                    qwen_direction.evidence[:2]
                                )
                                association_overlay = LiveMemeOverlay(
                                    profile_id=f"qwen_{qwen_pack.pack_id}",
                                    title=qwen_pack.title,
                                    subtitle=qwen_pack.subtitle,
                                    evidence=evidence_label or "LOCAL TOKENS",
                                    confidence=0.86,
                                    accent_bgr=qwen_pack.accent_bgr,
                                    sound_style=qwen_pack.sound_style,
                                    started_at=logical_now,
                                    expires_at=(
                                        logical_now + association_popup_seconds
                                    ),
                                )
                                sound.play_association(qwen_pack.sound_style)
                                last_qwen_direction_fingerprint = (
                                    qwen_direction.context_fingerprint
                                )
                                if director.action_enabled() and buffer_ready:
                                    event = qwen_direction_event(
                                        qwen_direction,
                                        qwen_pack,
                                        qwen_direction.evidence,
                                    )
                                    story_state = (
                                        "QWEN LOCK " + qwen_pack.pack_id.upper()
                                    )
                                    qwen_state = "EDIT " + qwen_pack.pack_id.upper()
                                    logs.add(
                                        "QWEN_DIRECTION edit="
                                        f"{qwen_pack.pack_id} frames=false"
                                    )
                                else:
                                    qwen_state = "MEME " + qwen_pack.pack_id.upper()
                                    logs.add(
                                        "QWEN_DIRECTION popup="
                                        f"{qwen_pack.pack_id} frames=false"
                                    )
                    if (
                        event is None
                        and args.script_preset is not None
                        and not scripted_event_sent
                        and logical_now >= max(minimum_buffer_seconds, args.script_at)
                    ):
                        event = detector.manual(
                            args.script_preset,
                            logical_now,
                            buffer_ready,
                        )
                        scripted_event_sent = event is not None
                    if (
                        event is None
                        and director.freestyle_due(logical_now, buffer_ready)
                    ):
                        requested = config.presets[
                            (editor.completed_count + len(recent_preset_ids) + 3)
                            % len(config.presets)
                        ]
                        event = ActionEvent(
                            kind="freestyle_montage",
                            timestamp=logical_now,
                            confidence=0.82,
                            source="SEMANTIC_FREESTYLE",
                            preset_id=requested.id,
                            story_id="freestyle_montage",
                            focus_label="subject",
                            started_at=logical_now - minimum_buffer_seconds,
                            peak_at=logical_now,
                            ended_at=logical_now,
                            captions=(
                                "DIRECTOR // RANDOM MEMORY",
                                "SETUP // MOMENT SELECTED",
                                "DROP // FREESTYLE CUT",
                                "REACTION // NO CONTEXT NEEDED",
                                "PAYOFF // CAMERA MADE A MEME",
                            ),
                            evidence=("director=freestyle",),
                        )
                        story_state = "FREESTYLE LOCK"
                    if detector.state != previous_detector_state:
                        if detector.state == "CANDIDATE":
                            logs.add("ACTION_CANDIDATE type=hand_to_mouth")
                            sound.play_analyzer("scan")
                        previous_detector_state = detector.state
            elif mode == "kiosk" and not editor.capture_paused:
                story_state = "WAIT OPT-IN"

            if event is not None and schedule_event(event):
                director.lock_for_edit()

            transitions = editor.update(logical_now, ring)
            for transition in transitions:
                if transition == "CAPTURED":
                    logs.add("BUFFER_SLICE captured=true")
                elif transition == "ASSEMBLING":
                    detector.pause_for_edit()
                    if semantic_worker is not None:
                        semantic_worker.pause(clear_cache=True)
                    if association_worker is not None:
                        association_worker.pause(clear_cache=True)
                    if qwen_worker is not None:
                        qwen_worker.pause()
                        qwen_state = "PAUSED EDIT"
                    association_overlay = None
                    sound.stop_association()
                    previous_detector_state = detector.state
                    logs.add("CAPTURE_PAUSED replay_safe=true")
                    logs.add("EDIT_COMPILE indexing=true")
                    sound.play_analyzer("assemble")
                elif transition == "PLAYING" and editor.active is not None:
                    logs.add(f"EDIT_PLAYBACK preset={editor.active.preset.title}")
                    sound.play_edit(editor.active.preset)
                    output_path = archive.start(
                        editor.active,
                        sound.edit_path(editor.active.preset),
                    )
                    if output_path is not None:
                        logs.add(f"EDIT_EXPORT recording={output_path.name}")
                elif transition == "DONE":
                    archive.finish()
                    sound.stop_edit()
                    ring.clear()
                    motion_history.clear()
                    last_event_at = None
                    last_vision = None
                    story_detector.reset()
                    gesture_controller.reset(logical_now)
                    gesture_meme_view = None
                    story_state = "WAIT OPT-IN" if mode == "kiosk" else "SEMANTIC SCAN"
                    if semantic_worker is not None:
                        semantic_worker.reset_epoch()
                        semantic_worker.resume()
                        next_semantic_submit = logical_now
                    if association_worker is not None:
                        association_epoch = association_worker.reset_epoch()
                        association_worker.resume()
                    else:
                        association_epoch += 1
                    association_result = None
                    association_detector.reset()
                    association_overlay = None
                    qwen_context.reset()
                    if qwen_worker is not None:
                        qwen_epoch = qwen_worker.reset_epoch()
                        if mode == "classic":
                            qwen_worker.pause()
                        else:
                            qwen_worker.resume()
                    last_qwen_direction_fingerprint = None
                    next_qwen_object_sample = logical_now
                    next_association_submit = logical_now
                    director.reset_after_edit(
                        logical_now,
                        object_available=bool(
                            association_worker is not None or args.association_demo
                        ),
                    )
                    if mode == "kiosk":
                        kiosk_gate.reset()
                        kiosk_state = kiosk_gate.state
                    detector.arm_post_playback_cooldown(
                        logical_now,
                        timer_delay_seconds=minimum_buffer_seconds,
                    )
                    previous_detector_state = detector.state
                    logs.add(
                        "CAPTURE_RESUMED fresh_buffer=true"
                        + (" kiosk=awaiting_opt_in" if mode == "kiosk" else "")
                    )
                    logs.add(f"DIRECTOR route={director.route}")
                    sound.play_analyzer("done")

            archive.record_until(logical_now)
            for notice in archive.poll():
                if notice.status == "SAVED":
                    saved_edit_paths.append(notice.path)
                    logs.add(f"EDIT_SAVED audio=true file={notice.path.name}")
                else:
                    logs.add(
                        f"EDIT_EXPORT {notice.status.lower()} file={notice.path.name}"
                    )

            if qwen_worker is not None:
                qwen_status = qwen_worker.status
                qwen_overlay_visible = bool(
                    association_overlay is not None
                    and association_overlay.profile_id.startswith("qwen_")
                    and logical_now < association_overlay.expires_at
                )
                if editor.capture_paused or qwen_status.paused:
                    qwen_state = "PAUSED"
                elif qwen_overlay_visible:
                    qwen_state = "MEME " + association_overlay.profile_id[5:].upper()
                elif qwen_status.inflight:
                    qwen_state = "THINKING"
                elif qwen_status.last_error:
                    qwen_state = "FALLBACK LOCAL"
                elif qwen_status.available:
                    qwen_state = "READY TEXT ONLY"
                else:
                    qwen_state = "OFFLINE LOCAL"

            frame_wall = time.monotonic()
            delta = max(1e-6, frame_wall - last_frame_wall)
            last_frame_wall = frame_wall
            fps_samples.append(1.0 / delta)
            fps = float(np.median(fps_samples)) if fps_samples else 0.0
            status = HudStatus(
                elapsed=logical_now,
                fps=fps,
                buffer_seconds=ring.coverage_seconds,
                buffer_max_seconds=ring.max_seconds,
                buffer_latest_at=ring.latest_timestamp,
                detector_state=(
                    "EDIT_PAUSED" if editor.capture_paused else detector.state
                ),
                candidate_progress=detector.candidate_progress(logical_now),
                cooldown_remaining=detector.cooldown_remaining(logical_now),
                timer_remaining=detector.timer_remaining(logical_now),
                auto_enabled=detector.auto_enabled,
                timer_enabled=detector.timer_enabled,
                edit_phase=editor.phase,
                queue_size=editor.queue_size,
                completed_count=editor.completed_count,
                engine=vision.engine,
                edit_audio=sound.edits_enabled,
                analyzer_audio=sound.analyzer_enabled,
                last_event_at=last_event_at,
                capture_paused=editor.capture_paused,
                logs=logs.recent(3),
                mode=mode,
                story_state=story_state,
                kiosk_state=kiosk_state,
                semantic_warning=semantic_warning,
                director_route=director.route,
                director_options=tuple(
                    (name, bool(getattr(director_options, name)))
                    for name in (
                        "action_stories",
                        "object_memes",
                        "freestyle_edits",
                        "culture_themes",
                        "random_per_cycle",
                    )
                ),
                association_state=association_state,
                qwen_state=qwen_state,
                association_overlay=association_overlay,
                gesture_meme_view=gesture_meme_view,
                gesture_reactor_state=gesture_controller.reactor_state,
                gesture_reactor_progress=gesture_controller.reactor_progress,
                gesture_distinct_moments=gesture_controller.unique_count,
                gesture_distinct_goal=gesture_controller.unique_goal,
                neutral_meme_path=gesture_controller.idle_image_path,
            )
            # Product invariant: the live camera never moves and every replay
            # is rendered as the same dedicated portrait panel on the right.
            # Presets differ inside that panel, not by rearranging the screen.
            edit_layout = "right_portrait"
            if not args.headless:
                display_size = window_render_size(window_title, display_size)
            render_size = display_size if not args.headless else output_size
            panel_width, panel_height = edit_panel_size(render_size, edit_layout)
            panel_render_width = max(378, panel_width)
            panel_render_height = max(672, panel_height)
            panel = editor.render_panel(
                panel_render_width,
                panel_render_height,
                logical_now,
            )
            canvas = compose_hud(
                frame,
                vision,
                status,
                render_size,
                panel,
                motion_history,
                hud_enabled,
                edit_layout,
            )
            if writer is not None:
                writer.write(fit_writer_frame(canvas, output_size))
            if manual_writer is not None:
                manual_writer.write(fit_writer_frame(canvas, output_size))

            key = -1
            if not args.headless:
                cv2.imshow(window_title, canvas)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            option_hotkeys = {
                ord("g"): "action_stories",
                ord("o"): "object_memes",
                ord("u"): "freestyle_edits",
                ord("y"): "culture_themes",
                ord("d"): "random_per_cycle",
            }
            if key in option_hotkeys:
                ui_actions.append(option_hotkeys[key])
            while ui_actions:
                option = ui_actions.popleft()
                if not editor.accepts_trigger:
                    logs.add(f"DIRECTOR_TOGGLE rejected={option} reason=capture_cycle_busy")
                    continue
                enabled = director_options.toggle(option)
                if option == "object_memes" and not enabled:
                    association_overlay = None
                    association_result = None
                    association_detector.reset()
                    sound.stop_association()
                elif option == "object_memes" and enabled:
                    next_association_submit = logical_now
                object_available = bool(
                    args.association_demo
                    or association_worker is not None
                    or vision.object_detector_available
                    or vision.objects
                )
                director.begin_cycle(logical_now, object_available=object_available)
                logs.add(
                    f"DIRECTOR_TOGGLE option={option} enabled={str(enabled).lower()} route={director.route}"
                )
            if key in (
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
                ord("5"),
                ord("6"),
                ord("7"),
                ord("8"),
                ord("9"),
                ord("0"),
                ord("z"),
                ord("x"),
                ord("c"),
                ord("b"),
                ord("l"),
                ord("j"),
                ord("k"),
                ord("p"),
                ord("i"),
                ord("s"),
                ord("w"),
                ord("e"),
                ord("["),
            ):
                kind = {
                    ord("1"): "sip",
                    ord("2"): "crunch",
                    ord("3"): "final_form",
                    ord("4"): "velocity",
                    ord("5"): "mente",
                    ord("6"): "clima",
                    ord("7"): "bootleg",
                    ord("8"): "atencao",
                    ord("9"): "ela",
                    ord("0"): "fruta",
                    ord("z"): "nunca",
                    ord("x"): "portal",
                    ord("c"): "glass",
                    ord("b"): "aspect_orbit",
                    ord("l"): "fluid_inception",
                    ord("j"): "wave_surge",
                    ord("k"): "pulse_chain",
                    ord("p"): "glide_chain",
                    ord("i"): "match_flow",
                    ord("s"): "arch_chain",
                    ord("w"): "subject_boomerang",
                    ord("e"): "gigachad",
                    ord("["): "cart_drift",
                }[key]
                if not editor.accepts_trigger:
                    logs.add("MANUAL_TRIGGER rejected=capture_cycle_busy")
                else:
                    manual_event = detector.manual(
                        kind,
                        logical_now,
                        ring.ready(minimum_buffer_seconds),
                    )
                    if manual_event is None:
                        logs.add("MANUAL_TRIGGER rejected=buffer_warming")
                    else:
                        schedule_event(manual_event)
                        logs.add(f"MANUAL_TRIGGER requested={kind}")
            elif key == ord("v"):
                if not editor.accepts_trigger:
                    logs.add("MODE_SWITCH rejected=capture_cycle_busy")
                else:
                    mode = next_mode(mode)
                    ring.clear()
                    motion_history.clear()
                    story_detector.reset()
                    gesture_controller.reset(logical_now)
                    gesture_meme_view = None
                    kiosk_gate.reset()
                    kiosk_state = kiosk_gate.state if mode == "kiosk" else "OFF"
                    story_state = "WAIT OPT-IN" if mode == "kiosk" else mode.upper()
                    last_event_at = None
                    if semantic_worker is not None:
                        semantic_worker.reset_epoch()
                        semantic_worker.resume()
                        next_semantic_submit = logical_now
                    if association_worker is not None:
                        association_epoch = association_worker.reset_epoch()
                        association_worker.resume()
                    else:
                        association_epoch += 1
                    association_result = None
                    association_detector.reset()
                    association_overlay = None
                    sound.stop_association()
                    qwen_context.reset()
                    if qwen_worker is not None:
                        qwen_epoch = qwen_worker.reset_epoch()
                        if mode == "classic":
                            qwen_worker.pause()
                        else:
                            qwen_worker.resume()
                    last_qwen_direction_fingerprint = None
                    next_qwen_object_sample = logical_now
                    next_association_submit = logical_now
                    director.begin_cycle(
                        logical_now,
                        object_available=bool(
                            association_worker is not None or args.association_demo
                        ),
                    )
                    detector.arm_post_playback_cooldown(
                        logical_now,
                        timer_delay_seconds=minimum_buffer_seconds,
                    )
                    logs.add(f"MODE_SWITCH mode={mode} fresh_buffer=true")
            elif key == ord("a"):
                logs.add(f"AUTO_GESTURE enabled={detector.toggle_auto()}")
            elif key == ord("t"):
                logs.add(f"TIMED_DEMO enabled={detector.toggle_timer_mode(logical_now)}")
            elif key in (10, 13):
                prior = editor.last_completed
                requested = prior.preset if prior is not None else None
                selected = (
                    choose_non_repeating_preset(
                        requested,
                        config.presets,
                        last_scheduled_track_id,
                    )
                    if requested is not None
                    else None
                )
                if selected is not None and editor.replay_last(logical_now, selected):
                    gesture_controller.freeze()
                    gesture_meme_view = None
                    detector.pause_for_edit()
                    if semantic_worker is not None:
                        semantic_worker.pause(clear_cache=True)
                    if association_worker is not None:
                        association_worker.pause(clear_cache=True)
                    if qwen_worker is not None:
                        qwen_worker.pause()
                        qwen_state = "PAUSED REMIX"
                    association_overlay = None
                    sound.stop_association()
                    previous_detector_state = detector.state
                    if requested is not None and selected.id != requested.id:
                        logs.add(
                            f"TRACK_GUARD remix={requested.title}->{selected.title}"
                        )
                    last_scheduled_track_id = selected.track_id
                    recent_preset_ids.append(selected.id)
                    logs.add("CAPTURE_PAUSED reason=remix")
                    logs.add(f"REMIX_LAST accepted=true preset={selected.title}")
                else:
                    logs.add("REMIX_LAST accepted=false")
            elif key == ord("m"):
                logs.add(f"EDIT_AUDIO enabled={sound.toggle_edits()}")
            elif key == ord("n"):
                logs.add(f"ANALYZER_AUDIO enabled={sound.toggle_analyzer()}")
            elif key == ord("h"):
                hud_enabled = not hud_enabled
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_title,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord("r"):
                if manual_writer is None:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    manual_writer_path = PROJECT_ROOT / "captures" / f"session-{stamp}.mp4"
                    manual_writer = make_writer(manual_writer_path, target_fps, output_size)
                    logs.add(f"FILE_RECORDING started={manual_writer_path.name}")
                else:
                    manual_writer.release()
                    manual_writer = None
                    logs.add("FILE_RECORDING stopped=true")

            frame_index += 1
            if args.max_seconds is not None and logical_now >= args.max_seconds:
                break

            if source_is_file and not args.headless:
                next_file_deadline += 1.0 / source_fps
                delay = next_file_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 0.05))
            elif synthetic is not None and not args.headless:
                next_file_deadline += 1.0 / source_fps
                delay = next_file_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 0.05))
    finally:
        if capture is not None:
            capture.release()
        if analyzer is not None:
            analyzer.close()
        if semantic_worker is not None:
            semantic_worker.close()
        if association_worker is not None:
            association_worker.close()
        if qwen_worker is not None:
            qwen_worker.close()
        if writer is not None:
            writer.release()
        if manual_writer is not None:
            manual_writer.release()
        for notice in archive.close():
            if notice.status == "SAVED":
                saved_edit_paths.append(notice.path)
            else:
                print(
                    f"Edit export {notice.status}: {notice.path} {notice.message}",
                    file=sys.stderr,
                )
        sound.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(
        f"Finished: frames={frame_index}, edits={editor.completed_count}, "
        f"buffer={ring.coverage_seconds:.2f}s"
    )
    if manual_writer_path is not None:
        print(f"Last recording: {manual_writer_path}")
    if saved_edit_paths:
        print(f"Last ready edit: {saved_edit_paths[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
