"""Host-owned plan/act/verify/recover state for the PRISM Android agent.

The language model proposes plans and actions.  This module owns the parts that
must not depend on the model being honest or remembering correctly:

* bounded plan and replan state;
* action/result/observation correlation;
* evidence-grounded step transitions and terminal completion;
* failed-action admission and recovery directives;
* an append-only per-run event journal.

It deliberately has no device, network, or PRISM imports.  ``agent_prism.py``
remains the harness that gathers defended observations, calls a model, applies
PROVE, and executes through ``DefendedDevice``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


SCHEMA_VERSION = 1
MAX_PLAN_STEPS = 8
MAX_PLAN_CRITERIA = 8
MAX_TEXT_CHARS = 320


class Phase(str, Enum):
    PLANNING = "planning"
    ACTING = "acting"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class Outcome(str, Enum):
    PROGRESS = "progress"
    NO_PROGRESS = "no_progress"
    FAILED = "failed"
    BLOCKED = "blocked"


def _bounded_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _bounded_text(item)
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove volatile coordinates while keeping the model-visible intent."""
    return {key: value for key, value in params.items() if key != "xy"}


def _action_signature(action: str, params: dict[str, Any]) -> str:
    identity = _safe_params(params)
    if action in {"tap", "web_tap"} and any(
        identity.get(key) not in (None, "")
        for key in ("rid", "text", "desc", "class", "selector")
    ):
        identity.pop("idx", None)
    canonical = json.dumps(
        {"action": action, "params": identity},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Observation:
    """Compact, deterministic representation of one defended device view."""

    screen_sig: str
    screen_changed: bool
    current_package: str | None
    labels: tuple[str, ...]
    element_count: int

    @classmethod
    def from_context(cls, ctx: Any, current_package: str | None = None) -> "Observation":
        canonical: list[dict[str, Any]] = []
        labels: list[str] = []
        for raw in getattr(ctx, "ui_elements", []) or []:
            element = {
                key: raw.get(key)
                for key in (
                    "class", "text", "desc", "hint", "rid", "input_field",
                    "selected", "focused", "disabled",
                )
                if raw.get(key) not in (None, "", False)
            }
            canonical.append(element)
            for key in ("text", "desc", "hint", "rid"):
                value = raw.get(key)
                if value not in (None, ""):
                    labels.append(_bounded_text(value, 180))

        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
        return cls(
            screen_sig=signature,
            screen_changed=bool(getattr(ctx, "screen_changed", True)),
            current_package=_bounded_text(current_package, 180) or None,
            labels=tuple(labels[:80]),
            element_count=len(canonical),
        )

    def event_view(self) -> dict[str, Any]:
        return {
            "screen_sig": self.screen_sig,
            "screen_changed": self.screen_changed,
            "current_package": self.current_package,
            "element_count": self.element_count,
            "labels": list(self.labels[:30]),
        }


@dataclass
class PlanStep:
    id: str
    objective: str
    success_evidence: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    failures: int = 0

    def prompt_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "success_evidence": list(self.success_evidence),
            "status": self.status.value,
            "attempts": self.attempts,
            "failures": self.failures,
        }


@dataclass
class AgentPlan:
    goal: str
    success_criteria: list[str]
    steps: list[PlanStep]
    revision: int = 0
    source: str = "model"

    def prompt_view(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "goal": self.goal,
            "success_criteria": list(self.success_criteria),
            "steps": [step.prompt_view() for step in self.steps],
        }


@dataclass
class Verification:
    attempt_id: str
    outcome: Outcome
    reason: str
    evidence: list[str] = field(default_factory=list)

    def prompt_view(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass
class ActionAttempt:
    id: str
    step_id: str | None
    action: str
    params: dict[str, Any]
    before: Observation
    result: str | None = None
    policy: dict[str, Any] | None = None
    verification: Verification | None = None

    def evidence_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_id": self.step_id,
            "action": self.action,
            "params": _safe_params(self.params),
            "result": self.result,
            "policy_decision": (
                self.policy.get("decision") if isinstance(self.policy, dict) else None
            ),
            "verification": (
                self.verification.prompt_view() if self.verification else None
            ),
        }


class EventJournal:
    """Small append-only JSONL journal; failures never stop the agent run."""

    def __init__(self, session_id: str, run_id: str, path: str | None = None):
        if path is None:
            safe_run = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
            root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "agent_runs")
            )
            path = os.path.join(root, f"{safe_run}.jsonl")
        self.path = os.path.abspath(path)
        self.session_id = session_id
        self.run_id = run_id
        self._seq = 0

    def append(self, event_type: str, phase: Phase, data: dict[str, Any]) -> None:
        self._seq += 1
        event = {
            "schema_version": SCHEMA_VERSION,
            "seq": self._seq,
            "ts": time.time(),
            "session_id": self.session_id,
            "run_id": self.run_id,
            "type": event_type,
            "phase": phase.value,
            "data": data,
        }
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                handle.flush()
        except Exception:
            # The journal is observability, not an execution authority.  PRISM
            # and PROVE must remain available even on a read-only filesystem.
            pass


class PlanValidationError(ValueError):
    pass


def parse_plan(
    payload: Any,
    *,
    task: str,
    revision: int = 0,
    source: str = "model",
    id_prefix: str = "s",
) -> AgentPlan:
    """Validate model-authored plan content while keeping status host-owned."""
    if not isinstance(payload, dict):
        raise PlanValidationError("plan must be a JSON object")

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanValidationError("plan requires a non-empty steps list")

    steps: list[PlanStep] = []
    for index, raw in enumerate(raw_steps[:MAX_PLAN_STEPS], start=1):
        if not isinstance(raw, dict):
            continue
        objective = _bounded_text(raw.get("objective") or raw.get("description"))
        if not objective:
            continue
        evidence = _string_list(
            raw.get("success_evidence", raw.get("evidence", [])),
            limit=5,
        )
        steps.append(
            PlanStep(
                id=f"{id_prefix}{index}",
                objective=objective,
                success_evidence=evidence,
            )
        )
    if not steps:
        raise PlanValidationError("plan has no valid steps")

    goal = _bounded_text(payload.get("goal") or task)
    criteria = _string_list(payload.get("success_criteria", []), limit=MAX_PLAN_CRITERIA)
    if not criteria:
        criteria = list(steps[-1].success_evidence)

    return AgentPlan(
        goal=goal or _bounded_text(task),
        success_criteria=criteria,
        steps=steps,
        revision=revision,
        source=source,
    )


def fallback_plan(task: str, procedure: str | None = None) -> AgentPlan:
    """Construct a conservative plan if the isolated planner is unavailable."""
    objectives: list[str] = []
    if procedure:
        for line in procedure.splitlines():
            is_list_item = bool(re.match(r"^\s*(?:[-*•]|\d+[.)])", line))
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if cleaned and is_list_item:
                objectives.append(_bounded_text(cleaned))
            if len(objectives) >= MAX_PLAN_STEPS:
                break
    if not objectives:
        objectives = [_bounded_text(task)]

    steps = [
        PlanStep(id=f"s{index}", objective=objective)
        for index, objective in enumerate(objectives, start=1)
        if objective
    ]
    return AgentPlan(
        goal=_bounded_text(task),
        success_criteria=[],
        steps=steps or [PlanStep(id="s1", objective="Complete the user task")],
        source="fallback",
    )


_EVIDENCE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "screen", "shows",
    "showing", "the", "to", "visible", "with",
})


def _normal(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_.@:+-]+", value.casefold()))


def _evidence_target(value: str) -> tuple[str, str]:
    raw = value.strip()
    match = re.match(r"^(text|package|rid)\s*:\s*(.+)$", raw, re.I)
    if match:
        return match.group(1).casefold(), _normal(match.group(2))
    quoted = re.findall(r"['\"]([^'\"]{2,})['\"]", raw)
    if quoted:
        return "text", _normal(quoted[0])
    return "text", _normal(raw)


def evidence_matches(value: str, observation: Observation) -> bool:
    """Ground a literal evidence claim in the current package/UI labels."""
    kind, target = _evidence_target(value)
    if not target:
        return False
    if kind == "package":
        package = _normal(observation.current_package or "")
        return target == package

    normalized_labels = [_normal(label) for label in observation.labels]
    if kind == "rid":
        return any(target == label or target in label.split() for label in normalized_labels)
    if any(target == label or target in label for label in normalized_labels):
        return True

    tokens = [
        token for token in target.split()
        if token not in _EVIDENCE_STOPWORDS and len(token) >= 3
    ]
    if len(tokens) >= 2:
        return any(all(token in label.split() for token in tokens) for label in normalized_labels)
    if len(tokens) == 1 and len(tokens[0]) >= 5:
        return any(tokens[0] in label.split() for label in normalized_labels)
    return False


class AgentController:
    """Deterministic orchestration state surrounding the model/device loop."""

    def __init__(
        self,
        task: str,
        session_id: str,
        *,
        max_decisions: int,
        max_replans: int = 2,
        journal_path: str | None = None,
    ):
        self.task = task
        self.session_id = session_id
        self.run_id = f"{session_id}-{uuid.uuid4().hex[:8]}"
        self.max_decisions = max(1, int(max_decisions))
        self.max_replans = max(0, int(max_replans))
        self.phase = Phase.PLANNING
        self.plan: AgentPlan | None = None
        self.pending_attempt: ActionAttempt | None = None
        self.attempts: list[ActionAttempt] = []
        self.last_verification: Verification | None = None
        self.last_feedback: str | None = None
        self.decision_count = 0
        self.action_count = 0
        self.replan_count = 0
        self.completion_rejections = 0
        self.consecutive_failures = 0
        self.consecutive_no_progress = 0
        self._replan_reason: str | None = None
        self._forbidden: list[dict[str, Any]] = []
        self.journal = EventJournal(session_id, self.run_id, journal_path)
        self.journal.append(
            "run.started",
            self.phase,
            {
                "task": _bounded_text(task, 1000),
                "max_decisions": self.max_decisions,
                "max_replans": self.max_replans,
            },
        )

    @property
    def journal_path(self) -> str:
        return self.journal.path

    @property
    def current_step(self) -> PlanStep | None:
        if not self.plan:
            return None
        return next(
            (step for step in self.plan.steps if step.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}),
            None,
        )

    @property
    def can_replan(self) -> bool:
        return self.replan_count < self.max_replans

    @property
    def needs_replan(self) -> bool:
        return self._replan_reason is not None

    @property
    def replan_reason(self) -> str | None:
        return self._replan_reason

    def install_initial_plan(self, payload: Any, procedure: str | None = None) -> AgentPlan:
        try:
            plan = parse_plan(payload, task=self.task)
        except PlanValidationError as exc:
            plan = fallback_plan(self.task, procedure)
            self.last_feedback = f"Planner unavailable or invalid; using fallback plan: {exc}"
        self.plan = plan
        self.phase = Phase.ACTING
        self.journal.append("plan.created", self.phase, plan.prompt_view())
        return plan

    def revise_plan(self, payload: Any, reason: str) -> bool:
        """Replace only unfinished work; completed facts remain host-owned."""
        self.replan_count += 1
        next_revision = (self.plan.revision + 1) if self.plan else self.replan_count
        try:
            replacement = parse_plan(
                payload,
                task=self.task,
                revision=next_revision,
                source="replan",
                id_prefix=f"r{next_revision}s",
            )
        except PlanValidationError as exc:
            self._replan_reason = None
            self.phase = Phase.RECOVERING
            self.last_feedback = f"Replan attempt failed validation: {exc}"
            self.journal.append(
                "plan.revision_failed",
                self.phase,
                {"reason": reason, "error": str(exc), "attempt": self.replan_count},
            )
            return False

        completed = []
        if self.plan:
            completed = [
                step for step in self.plan.steps
                if step.status in {StepStatus.COMPLETED, StepStatus.SKIPPED}
            ]
            if not replacement.success_criteria:
                replacement.success_criteria = list(self.plan.success_criteria)
        remaining_slots = max(0, MAX_PLAN_STEPS - len(completed))
        replacement.steps = completed + replacement.steps[:remaining_slots]
        self.plan = replacement
        self._replan_reason = None
        self.consecutive_failures = 0
        self.consecutive_no_progress = 0
        self.last_feedback = f"Plan revised after: {reason}"
        self.phase = Phase.ACTING
        self.journal.append(
            "plan.revised",
            self.phase,
            {"reason": reason, "plan": replacement.prompt_view()},
        )
        return True

    def record_replan_unavailable(self, reason: str) -> None:
        self.replan_count += 1
        self._replan_reason = None
        self.phase = Phase.RECOVERING
        self.last_feedback = f"Replanning was unavailable: {reason}. Use a different grounded route."
        self.journal.append(
            "plan.revision_failed",
            self.phase,
            {"reason": reason, "attempt": self.replan_count},
        )

    def acknowledge_replan_exhausted(self) -> None:
        reason = self._replan_reason or "recovery budget exhausted"
        self._replan_reason = None
        self.phase = Phase.RECOVERING
        self.last_feedback = (
            f"Replan budget exhausted after: {reason}. Make one new grounded attempt "
            "or report the dead end; never bypass a security block."
        )
        self.journal.append(
            "recovery.exhausted",
            self.phase,
            {"reason": reason, "replans": self.replan_count},
        )

    def request_replan(self, reason: str) -> None:
        reason = _bounded_text(reason, 600)
        if not self._replan_reason:
            self._replan_reason = reason
        self.phase = Phase.RECOVERING
        self.journal.append("recovery.requested", self.phase, {"reason": reason})

    def begin_decision(self, outer_step: int) -> None:
        self.decision_count += 1
        self.journal.append(
            "decision.started",
            self.phase,
            {
                "decision": self.decision_count,
                "outer_step": outer_step,
                "remaining": max(0, self.max_decisions - self.decision_count),
            },
        )

    def record_proposal(self, action: str, params: dict[str, Any], thought: str) -> None:
        self.journal.append(
            "action.proposed",
            self.phase,
            {
                "action": action,
                "params": _safe_params(params),
                "thought": _bounded_text(thought, 600),
                "step_id": self.current_step.id if self.current_step else None,
            },
        )

    def note_loop_signal(self, signal: str, action: str, params: dict[str, Any]) -> None:
        reason = f"loop detector reported {signal} severity for action {action}"
        self.last_feedback = (
            "The proposed action is part of a repeated no-progress pattern. "
            "Do not execute it again; revise the route from the current observation."
        )
        self.request_replan(reason)

    def admit_action(
        self,
        action: str,
        params: dict[str, Any],
        observation: Observation,
    ) -> tuple[bool, str | None]:
        signature = _action_signature(action, params)
        for blocked in reversed(self._forbidden[-8:]):
            same_action = blocked["signature"] == signature
            same_screen = blocked.get("screen_sig") == observation.screen_sig
            if same_action and (same_screen or blocked.get("outcome") == Outcome.BLOCKED.value):
                reason = (
                    f"exact retry rejected: prior {blocked['outcome']} on "
                    f"attempt {blocked['attempt_id']} ({blocked['reason']})"
                )
                self.last_feedback = reason
                self.journal.append(
                    "action.rejected",
                    Phase.RECOVERING,
                    {"action": action, "params": _safe_params(params), "reason": reason},
                )
                self.phase = Phase.RECOVERING
                if self.can_replan:
                    self.request_replan(
                        f"exact failed action {action} was proposed again on the same state"
                    )
                return False, reason
        return True, None

    def begin_action(
        self,
        action: str,
        params: dict[str, Any],
        observation: Observation,
    ) -> ActionAttempt:
        if self.pending_attempt is not None:
            raise RuntimeError("cannot begin an action before verifying the previous attempt")
        self.action_count += 1
        current = self.current_step
        if current:
            current.status = StepStatus.IN_PROGRESS
            current.attempts += 1
        attempt = ActionAttempt(
            id=f"a{self.action_count}",
            step_id=current.id if current else None,
            action=action,
            params=dict(params),
            before=observation,
        )
        self.pending_attempt = attempt
        self.phase = Phase.ACTING
        self.journal.append(
            "action.started",
            self.phase,
            attempt.evidence_view(),
        )
        return attempt

    def settle_action(self, result: str, policy: dict[str, Any] | None = None) -> None:
        if self.pending_attempt is None:
            raise RuntimeError("cannot settle an action that was not started")
        self.pending_attempt.result = _bounded_text(result, 1000)
        self.pending_attempt.policy = policy
        self.phase = Phase.VERIFYING
        self.journal.append(
            "action.settled",
            self.phase,
            {
                "attempt_id": self.pending_attempt.id,
                "result": self.pending_attempt.result,
                "policy": policy,
            },
        )

    def observe(self, observation: Observation) -> tuple[Verification | None, PlanStep | None]:
        self.journal.append("observation.recorded", self.phase, observation.event_view())
        verification = None
        if self.pending_attempt is not None and self.pending_attempt.result is not None:
            verification = self._verify_pending(observation)
        completed = self.auto_advance(observation)
        return verification, completed

    def _verify_pending(self, after: Observation) -> Verification:
        attempt = self.pending_attempt
        if attempt is None or attempt.result is None:
            raise RuntimeError("pending attempt has no result")

        result = attempt.result.strip()
        lowered = result.casefold()
        evidence: list[str] = []

        if lowered.startswith("blocked_by_"):
            outcome = Outcome.BLOCKED
            reason = result
        elif (
            lowered.startswith("error:")
            or lowered.startswith("not found:")
            or lowered.startswith("bad xy:")
            or lowered == "unknown"
        ):
            outcome = Outcome.FAILED
            reason = result
        elif lowered != "ok":
            outcome = Outcome.FAILED
            reason = f"unexpected executor result: {result}"
        else:
            if attempt.before.screen_sig != after.screen_sig:
                evidence.append("screen signature changed")
            if attempt.before.current_package != after.current_package and after.current_package:
                evidence.append(f"package:{after.current_package}")
            if attempt.action == "open_app":
                expected = str(attempt.params.get("package", ""))
                if expected and expected == (after.current_package or ""):
                    evidence.append(f"package:{expected}")
            if attempt.action in {"type", "web_type"}:
                typed = str(attempt.params.get("text", "")).strip()
                if typed and evidence_matches(f"text:{typed}", after):
                    evidence.append(f"text:{typed}")

            if evidence:
                outcome = Outcome.PROGRESS
                reason = "; ".join(evidence)
            else:
                outcome = Outcome.NO_PROGRESS
                reason = "executor returned ok but the defended observation did not change"

        verification = Verification(
            attempt_id=attempt.id,
            outcome=outcome,
            reason=reason,
            evidence=evidence,
        )
        attempt.verification = verification
        self.attempts.append(attempt)
        self.last_verification = verification
        self.pending_attempt = None

        step = next(
            (item for item in (self.plan.steps if self.plan else []) if item.id == attempt.step_id),
            None,
        )
        if outcome == Outcome.PROGRESS:
            self.consecutive_failures = 0
            self.consecutive_no_progress = 0
            self.last_feedback = f"Attempt {attempt.id} made observable progress: {reason}"
            self.phase = Phase.ACTING
        else:
            self.consecutive_failures += 1
            if outcome == Outcome.NO_PROGRESS:
                self.consecutive_no_progress += 1
            else:
                self.consecutive_no_progress = 0
            if step:
                step.failures += 1
            self.last_feedback = f"Attempt {attempt.id} {outcome.value}: {reason}"
            self._forbidden.append({
                "signature": _action_signature(attempt.action, attempt.params),
                "screen_sig": attempt.before.screen_sig,
                "outcome": outcome.value,
                "attempt_id": attempt.id,
                "reason": reason,
                "action": attempt.action,
                "params": _safe_params(attempt.params),
            })
            if outcome == Outcome.BLOCKED:
                self.request_replan(
                    f"{attempt.id} was blocked by a security or policy boundary; "
                    "the blocked action must not be bypassed"
                )
            elif (step and step.failures >= 2) or self.consecutive_no_progress >= 2:
                self.request_replan(
                    f"{attempt.id} produced {outcome.value} twice on plan step "
                    f"{attempt.step_id}; choose a different high-level route"
                )
            else:
                self.phase = Phase.RECOVERING

        self.journal.append(
            "action.verified",
            self.phase,
            {
                "attempt": attempt.evidence_view(),
                "after": after.event_view(),
            },
        )
        return verification

    def _complete_step(self, step: PlanStep, evidence: list[str], source: str) -> PlanStep:
        step.status = StepStatus.COMPLETED
        self.last_feedback = f"Plan step {step.id} verified complete via {source}."
        self.phase = Phase.ACTING if self.current_step else Phase.VERIFYING
        self.journal.append(
            "plan.step_completed",
            self.phase,
            {
                "step": step.prompt_view(),
                "evidence": evidence,
                "source": source,
            },
        )
        return step

    def auto_advance(self, observation: Observation) -> PlanStep | None:
        if self.phase == Phase.RECOVERING or self.needs_replan:
            return None
        step = self.current_step
        if not step or not step.success_evidence:
            return None
        matched = [
            item for item in step.success_evidence
            if evidence_matches(item, observation)
        ]
        if not matched:
            return None
        return self._complete_step(step, matched, "planned observation evidence")

    def claim_step_complete(
        self,
        claimed_evidence: Any,
        observation: Observation,
    ) -> tuple[bool, str]:
        step = self.current_step
        if not step:
            return False, "there is no unfinished plan step"
        claims = _string_list(
            claimed_evidence if isinstance(claimed_evidence, list) else [claimed_evidence],
            limit=6,
        )
        grounded_claims = [item for item in claims if evidence_matches(item, observation)]
        expected = [
            item for item in step.success_evidence
            if evidence_matches(item, observation)
        ]
        has_verified_progress = any(
            attempt.step_id == step.id
            and attempt.verification is not None
            and attempt.verification.outcome == Outcome.PROGRESS
            for attempt in self.attempts
        )
        substitute_evidence = bool(grounded_claims and has_verified_progress)
        if step.success_evidence and not expected and not substitute_evidence:
            reason = "current screen does not satisfy the step's planned success evidence"
        elif not grounded_claims and not expected:
            reason = "advance requires at least one literal package/UI evidence claim"
        elif not step.success_evidence and not has_verified_progress:
            reason = "advance without planned evidence requires a verified progressing attempt on this step"
        else:
            evidence = list(dict.fromkeys(expected + grounded_claims))
            self._complete_step(step, evidence, "model claim checked by host")
            return True, f"step {step.id} completed"

        self.last_feedback = reason
        self.phase = Phase.RECOVERING
        self.journal.append(
            "plan.step_rejected",
            self.phase,
            {"step_id": step.id, "claims": claims, "reason": reason},
        )
        return False, reason

    def completion_payload(self, observation: Observation, summary: str, evidence: Any) -> dict[str, Any]:
        return {
            "task": self.task,
            "plan": self.plan.prompt_view() if self.plan else None,
            "completion_proposal": {
                "summary": _bounded_text(summary, 1000),
                "claimed_evidence": _string_list(
                    evidence if isinstance(evidence, list) else [evidence], limit=8
                ),
            },
            "current_observation": observation.event_view(),
            "trusted_action_outcomes": [
                attempt.evidence_view() for attempt in self.attempts[-8:]
            ],
        }

    def _verifier_evidence_grounded(self, claim: str, observation: Observation) -> bool:
        match = re.fullmatch(r"action\s*:\s*([A-Za-z0-9_.-]+)", claim.strip(), re.I)
        if match:
            attempt = next((item for item in self.attempts if item.id == match.group(1)), None)
            return bool(
                attempt
                and attempt.result == "ok"
                and attempt.verification
                and attempt.verification.outcome == Outcome.PROGRESS
            )
        return evidence_matches(claim, observation)

    def accept_completion(
        self,
        verdict: Any,
        observation: Observation,
        summary: str,
    ) -> tuple[bool, str]:
        """Accept terminal success only with a valid, grounded verifier result."""
        if not isinstance(verdict, dict):
            return self.reject_completion("completion verifier returned no structured verdict")
        state = str(verdict.get("verdict", "uncertain")).casefold()
        satisfied = _string_list(
            verdict.get("satisfied_criteria", []), limit=MAX_PLAN_CRITERIA
        )
        missing = _string_list(verdict.get("missing_criteria", []), limit=MAX_PLAN_CRITERIA)
        claims = _string_list(verdict.get("grounded_evidence", []), limit=12)
        grounded = [claim for claim in claims if self._verifier_evidence_grounded(claim, observation)]
        screen_grounded = [
            claim for claim in grounded
            if not re.match(r"^action\s*:", claim.strip(), re.I)
        ]
        criteria = list(self.plan.success_criteria) if self.plan else []
        satisfied_keys = {item.casefold() for item in satisfied}
        unaccounted = [item for item in criteria if item.casefold() not in satisfied_keys]
        unresolved = [
            step for step in (self.plan.steps if self.plan else [])
            if step.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}
        ]
        deterministic = False
        if criteria:
            deterministic = all(
                evidence_matches(item, observation) for item in criteria
            )

        if state != "complete":
            reason = _bounded_text(verdict.get("reason") or f"verdict={state}", 800)
            return self.reject_completion(reason, missing)
        if missing:
            return self.reject_completion("verifier reported missing success criteria", missing)
        if unaccounted:
            return self.reject_completion(
                "verifier did not account for every host success criterion", unaccounted
            )
        if not grounded and not deterministic:
            return self.reject_completion(
                "verifier supplied no evidence grounded in the current screen/package or a verified action"
            )
        if not screen_grounded and not deterministic:
            return self.reject_completion(
                "completion requires current screen or package evidence; action progress alone is insufficient"
            )
        if unresolved and criteria and not deterministic:
            required_claims = min(2, len(criteria))
            if len(set(grounded)) < required_claims:
                return self.reject_completion(
                    "unfinished plan steps require independently grounded evidence for the goal"
                )

        if self.plan:
            for step in self.plan.steps:
                if step.status in {StepStatus.PENDING, StepStatus.IN_PROGRESS}:
                    step.status = StepStatus.SKIPPED
        self.phase = Phase.COMPLETED
        reason = _bounded_text(verdict.get("reason") or "completion verified", 800)
        self.journal.append(
            "run.completed",
            self.phase,
            {
                "summary": _bounded_text(summary, 1200),
                "reason": reason,
                "grounded_evidence": grounded,
                "deterministic_criteria_match": deterministic,
                "decisions": self.decision_count,
                "actions": self.action_count,
                "replans": self.replan_count,
            },
        )
        return True, reason

    def reject_completion(
        self,
        reason: str,
        missing: Iterable[str] | None = None,
    ) -> tuple[bool, str]:
        self.completion_rejections += 1
        missing_list = list(missing or [])
        detail = _bounded_text(reason, 800)
        if missing_list:
            detail = f"{detail}; missing: {', '.join(missing_list)}"
        self.last_feedback = f"Completion rejected: {detail}"
        self.phase = Phase.RECOVERING
        self.journal.append(
            "run.completion_rejected",
            self.phase,
            {"reason": detail, "count": self.completion_rejections},
        )
        if self.completion_rejections >= 2 and self.can_replan:
            self.request_replan("completion verification failed twice with missing or ungrounded evidence")
        return False, detail

    def record_agent_failure(self, reason: str) -> bool:
        """Treat model ``fail`` as a recovery request before terminal failure."""
        reason = _bounded_text(reason, 1000)
        if self.can_replan:
            self.last_feedback = f"Agent reported a dead end: {reason}"
            step_id = self.current_step.id if self.current_step else "none"
            self.request_replan(f"acting model reported a dead end on plan step {step_id}")
            return False
        self.fail(reason)
        return True

    def fail(self, reason: str) -> None:
        if self.phase == Phase.FAILED:
            return
        self.phase = Phase.FAILED
        self.journal.append(
            "run.failed",
            self.phase,
            {
                "reason": _bounded_text(reason, 1200),
                "decisions": self.decision_count,
                "actions": self.action_count,
                "replans": self.replan_count,
            },
        )

    def replan_context(self, observation: Observation) -> dict[str, Any]:
        return {
            "task": self.task,
            "reason_for_replan": self._replan_reason,
            "current_plan": self.plan.prompt_view() if self.plan else None,
            # Replanning is privileged control flow.  Keep untrusted UI labels
            # out of it; the acting model may inspect them later under the
            # ordinary PRISM/PROVE action boundary.
            "current_observation": {
                "screen_sig": observation.screen_sig,
                "screen_changed": observation.screen_changed,
                "current_package": observation.current_package,
                "element_count": observation.element_count,
            },
            "completed_steps": [
                step.prompt_view() for step in (self.plan.steps if self.plan else [])
                if step.status == StepStatus.COMPLETED
            ],
            "recent_outcomes": [
                {
                    "id": attempt.id,
                    "step_id": attempt.step_id,
                    "action": attempt.action,
                    "outcome": (
                        attempt.verification.outcome.value
                        if attempt.verification else "unverified"
                    ),
                }
                for attempt in self.attempts[-6:]
            ],
            "blocked_or_failed_actions": [
                {
                    "action": item["action"],
                    "outcome": item["outcome"],
                }
                for item in self._forbidden[-6:]
            ],
        }

    def prompt_state(self) -> dict[str, Any]:
        active = self.current_step
        recovery = None
        if self.phase == Phase.RECOVERING or self._replan_reason:
            recovery = {
                "reason": self._replan_reason or self.last_feedback,
                "instruction": (
                    "Choose a different grounded route. Never bypass PRISM/PROVE or retry "
                    "an exact action listed under do_not_repeat."
                ),
            }
        return {
            "run_id": self.run_id,
            "phase": self.phase.value,
            "plan": self.plan.prompt_view() if self.plan else None,
            "active_step": active.prompt_view() if active else None,
            "all_plan_steps_resolved": bool(self.plan and active is None),
            "last_verification": (
                self.last_verification.prompt_view() if self.last_verification else None
            ),
            "last_feedback": self.last_feedback,
            "recovery": recovery,
            "do_not_repeat": [
                {
                    "action": item["action"],
                    "params": item["params"],
                    "outcome": item["outcome"],
                    "reason": item["reason"],
                }
                for item in self._forbidden[-5:]
            ],
            "budget": {
                "decisions_used": self.decision_count,
                "decisions_remaining": max(0, self.max_decisions - self.decision_count),
                "final_decision": self.decision_count >= self.max_decisions,
                "actions_executed": self.action_count,
                "replans_used": self.replan_count,
                "replans_remaining": max(0, self.max_replans - self.replan_count),
            },
        }


__all__ = [
    "AgentController",
    "AgentPlan",
    "Observation",
    "Outcome",
    "Phase",
    "PlanStep",
    "StepStatus",
    "evidence_matches",
]
