"""Independent permission manager for the Hybrid Agent.

Every agent action (a shell command or a file operation) is turned into an
:class:`ActionRequest`, evaluated here, and only then executed by the tool
executor. The decision is one of:

* ``allow`` – run without asking
* ``ask``   – pause and show the approval dialog
* ``deny``  – refuse outright

Stored "Always Allow" grants are consulted *after* classification. They can only
relax decisions the classifier marked as grantable (file writes/deletes and
scripts inside an approved directory). Installation, elevation, registry,
service, firewall and other system changes are never grantable, so a stored
grant never replaces an individual approval for those.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import urlparse

from app.agent import command_classifier as classifier
from app.agent.command_classifier import ALLOW, ASK, DENY, Classification, is_within

#: Categories that a stored grant may cover. Everything else is per-request only.
GRANTABLE_CATEGORIES = frozenset({"write", "delete", "execution"})

#: Actions that always need a fresh, individual approval.
ABSOLUTE_ASK_CATEGORIES = frozenset(
    {
        "install",
        "system",
        "elevated",
        "network",
        "opaque",
        "credentials",
        "destructive",
        "self",
        "computer",
    }
)

_FILE_KINDS = {"read", "write", "create_dir", "move", "delete"}


class PermissionError_(RuntimeError):
    """Raised when permission state would be widened by a non-user origin."""


class PermissionEscalationError(PermissionError_):
    """Raised when a sub-agent tries to grant itself rights."""


@dataclass(frozen=True)
class ActionRequest:
    kind: str
    description: str = ""
    command: str = ""
    shell: str = "cmd"
    cwd: str = ""
    paths: tuple[str, ...] = ()
    purpose: str = ""
    target: str = ""
    requires_admin: bool = False
    task_id: str = ""
    source: str = "agent"
    mode: str = ""
    recursive: bool = False

    def fingerprint(self) -> str:
        payload = "\u001f".join(
            [
                self.kind,
                self.command,
                self.shell,
                os.path.normcase(os.path.normpath(self.cwd or "")),
                "|".join(os.path.normcase(os.path.normpath(p)) for p in self.paths),
                self.description,
                self.target,
                str(self.recursive),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


@dataclass(frozen=True)
class PermissionDecision:
    decision: str
    category: str
    reason: str
    request: ActionRequest
    requires_admin: bool = False
    grantable: bool = False
    scope: str = ""
    grant_id: str = ""
    classification: Classification | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == ALLOW

    @property
    def fingerprint(self) -> str:
        return self.request.fingerprint()


@dataclass(frozen=True)
class ApprovalRequest:
    """What the approval dialog needs to display."""

    decision: PermissionDecision
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", self.decision.request.fingerprint())

    @property
    def command(self) -> str:
        return self.decision.request.command

    @property
    def shell(self) -> str:
        return self.decision.request.shell

    @property
    def cwd(self) -> str:
        return self.decision.request.cwd

    @property
    def purpose(self) -> str:
        request = self.decision.request
        return request.purpose or request.description

    @property
    def target(self) -> str:
        request = self.decision.request
        if request.target:
            return request.target
        if request.paths:
            return ", ".join(request.paths)
        return request.command

    @property
    def risk(self) -> str:
        if self.decision.classification is not None:
            return classifier.describe_risk(self.decision.classification)
        if self.decision.category == "write":
            return "ファイル書き込み"
        if self.decision.category == "delete":
            return "削除"
        if self.decision.category == "system":
            return "システム情報の取得"
        return self.decision.category

    @property
    def reason(self) -> str:
        return self.decision.reason

    @property
    def requires_admin(self) -> bool:
        return self.decision.requires_admin

    @property
    def grantable(self) -> bool:
        return self.decision.grantable


@dataclass(frozen=True)
class ApprovalOutcome:
    """Result of the approval dialog."""

    choice: str  # allow_once | always_allow | reject | stop
    fingerprint: str = ""
    grant_id: str = ""

    @property
    def allowed(self) -> bool:
        return self.choice in {"allow_once", "always_allow"}

    @property
    def stop_task(self) -> bool:
        return self.choice == "stop"


class RejectingApprovalBroker:
    """Default broker used when no UI is attached (tests, headless runs)."""

    def __init__(self, choice: str = "reject"):
        self.choice = choice
        self.requests: list[ApprovalRequest] = []

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        self.requests.append(request)
        return ApprovalOutcome(self.choice, request.fingerprint)


class PermissionManager:
    def __init__(self, store: Any, settings: Any):
        self.store = store
        self.settings = settings

    # -- configuration -----------------------------------------------------
    @property
    def full_access(self) -> bool:
        return bool(self.settings.get("agent.full_access", False))

    def protected_paths(self) -> tuple[str, ...]:
        paths = []
        try:
            from app.core import paths as app_paths

            paths.extend(
                [
                    str(app_paths.config_dir()),
                    str(app_paths.data_dir() / "agent.db"),
                    str(app_paths.config_dir() / "secrets.dat"),
                    str(app_paths.config_dir() / "settings.json"),
                ]
            )
        except Exception:
            pass
        return tuple(paths)

    def allowed_roots(self, workspace: str = "") -> tuple[str, ...]:
        roots: list[str] = []
        for value in (workspace, self.settings.get("agent.workspace", "")):
            text = str(value or "").strip()
            if text:
                roots.append(text)
        extra = self.settings.get("agent.extra_directories", []) or []
        if isinstance(extra, (list, tuple)):
            roots.extend(str(item) for item in extra if str(item or "").strip())
        seen: list[str] = []
        for root in roots:
            normalized = classifier.normalize_path(root)
            if normalized and normalized not in seen:
                seen.append(normalized)
        return tuple(roots)

    # -- evaluation --------------------------------------------------------
    def evaluate(self, request: ActionRequest) -> PermissionDecision:
        if request.kind == "command":
            decision = self._evaluate_command(request)
        elif request.kind in _FILE_KINDS:
            decision = self._evaluate_path(request)
        elif request.kind in {"system", "memory"}:
            decision = PermissionDecision(
                ALLOW,
                "read_only",
                "Local, read-only agent operation.",
                request,
            )
        elif request.kind == "computer":
            decision = PermissionDecision(
                ASK,
                "computer",
                "Computer Use always requires an individual approval.",
                request,
                requires_admin=False,
                grantable=False,
            )
        elif request.kind == "performance":
            decision = PermissionDecision(
                ALLOW,
                "read_only",
                "Read-only local performance sampling.",
                request,
            )
        elif request.kind == "web_search":
            decision = self._evaluate_web_search(request)
        else:
            decision = self._deny(request, "unknown", f"Unknown action kind: {request.kind}")
        return self._apply_grant(decision)

    def _evaluate_command(self, request: ActionRequest) -> PermissionDecision:
        if request.source == "agent" and not bool(
            self.settings.get("agent.command_execution", True)
        ):
            return self._deny(
                request, "disabled", "Command execution is disabled in Agent settings."
            )
        classification = classifier.classify_command(
            request.command,
            shell=request.shell,
            cwd=request.cwd,
            allowed_roots=self.allowed_roots(request.cwd),
            protected_paths=self.protected_paths(),
        )
        category = classification.category
        requires_admin = bool(
            request.requires_admin or classification.requires_admin
        )
        grantable = (
            classification.grantable
            and category in GRANTABLE_CATEGORIES
            and not requires_admin
        )
        return PermissionDecision(
            decision=classification.decision,
            category=category,
            reason=classification.reason,
            request=request,
            requires_admin=requires_admin,
            grantable=grantable,
            scope=classification.scope or request.cwd,
            classification=classification,
        )

    def _evaluate_path(self, request: ActionRequest) -> PermissionDecision:
        if request.source == "agent" and request.kind != "read" and not bool(
            self.settings.get("agent.file_access", True)
        ):
            return self._deny(request, "disabled", "File access is disabled in Agent settings.")
        paths = tuple(str(path) for path in request.paths if str(path or "").strip())
        if not paths:
            return self._deny(request, "unknown", "The target path is missing.")
        for path in paths:
            if classifier.is_protected_path(path, self.protected_paths()):
                return self._deny(
                    request,
                    "credentials",
                    "That path holds credentials, application settings or agent state.",
                )
        roots = self.allowed_roots(request.cwd if request.kind != "read" else "")
        inside = all(is_within(path, roots) for path in paths) if roots else False

        if request.kind == "read":
            return PermissionDecision(
                ALLOW, "read_only", "Reading a file or directory.", request, scope=request.cwd
            )

        if inside:
            return PermissionDecision(
                ALLOW,
                "read_only" if request.kind == "write" else request.kind,
                "Inside the working area.",
                request,
                scope=request.cwd,
            )

        if self.full_access and request.kind != "delete":
            return PermissionDecision(
                ALLOW,
                request.kind,
                "Full Access Mode allows file changes outside the working area.",
                request,
                scope=request.cwd,
            )

        if request.kind == "delete":
            grantable = not request.recursive and len(paths) == 1
            return PermissionDecision(
                ASK,
                "delete",
                "Deleting outside the working area always needs approval.",
                request,
                grantable=grantable,
                scope=os.path.dirname(paths[0]),
            )

        scope = os.path.dirname(paths[0]) if paths else request.cwd
        return PermissionDecision(
            ASK,
            request.kind,
            "Writing outside the working area needs approval.",
            request,
            grantable=not request.recursive,
            scope=scope,
        )

    def _evaluate_web_search(self, request: ActionRequest) -> PermissionDecision:
        from app.agent.web_search import ALLOWED_SEARCH_HOSTS

        host = (urlparse(request.target).hostname or "").casefold()
        if host not in ALLOWED_SEARCH_HOSTS:
            return self._deny(
                request, "network", "Only the configured public search endpoint is allowed."
            )
        return PermissionDecision(
            ASK,
            "network",
            f"Read-only web search to {host} requires network approval.",
            request,
            grantable=True,
            scope=host,
        )

    @staticmethod
    def _deny(request: ActionRequest, category: str, reason: str) -> PermissionDecision:
        return PermissionDecision(DENY, category, reason, request)

    # -- stored grants -----------------------------------------------------
    def _grant_target(self, decision: PermissionDecision) -> str:
        request = decision.request
        if request.kind == "command":
            program = ""
            if decision.classification is not None:
                program = decision.classification.program
            return program.casefold()
        if request.kind == "web_search":
            return (urlparse(request.target).hostname or "").casefold()
        return ""

    def _grant_scope(self, decision: PermissionDecision) -> str:
        request = decision.request
        if request.kind == "command":
            return classifier.normalize_path(request.cwd)
        if request.paths:
            return classifier.normalize_path(os.path.dirname(request.paths[0]))
        return classifier.normalize_path(decision.scope)

    def _apply_grant(self, decision: PermissionDecision) -> PermissionDecision:
        if decision.decision != ASK or not decision.grantable:
            return decision
        if (
            decision.requires_admin
            or (
                decision.category in ABSOLUTE_ASK_CATEGORIES
                and decision.request.kind != "web_search"
            )
        ):
            return decision
        scope = self._grant_scope(decision)
        target = self._grant_target(decision)
        try:
            grants = self.store.list_grants()
        except Exception:
            return decision
        for grant in grants:
            if grant.kind != decision.request.kind:
                continue
            if grant.target and grant.target.casefold() != target:
                continue
            if grant.scope:
                if not scope or not is_within(scope, [grant.scope]):
                    continue
            elif grant.target:
                continue
            try:
                self.store.touch_grant(grant.id)
            except Exception:
                pass
            return PermissionDecision(
                ALLOW,
                decision.category,
                f"Allowed by a stored permission for {grant.label}.",
                decision.request,
                requires_admin=decision.requires_admin,
                grantable=decision.grantable,
                scope=decision.scope,
                grant_id=grant.id,
                classification=decision.classification,
            )
        return decision

    def add_grant(self, decision: PermissionDecision, *, origin: str = "agent") -> Any:
        """Persist an "Always Allow" grant.

        Only user-originated flows (the approval dialog and Agent settings) may
        call this; the agent itself never has access to a grant-writing API.
        """
        if origin != "user":
            raise PermissionEscalationError(
                "Stored permissions can only be created from the approval dialog."
            )
        if decision.decision != ASK or not decision.grantable:
            raise PermissionEscalationError("This action cannot be granted persistently.")
        if decision.requires_admin or (
            decision.category in ABSOLUTE_ASK_CATEGORIES
            and decision.request.kind != "web_search"
        ):
            raise PermissionEscalationError("High-risk actions always need approval.")
        return self.store.add_grant(
            kind=decision.request.kind,
            scope=self._grant_scope(decision),
            target=self._grant_target(decision),
            category=decision.category,
        )

    def list_grants(self) -> list[Any]:
        try:
            return self.store.list_grants()
        except Exception:
            return []

    def revoke_grant(self, grant_id: str, *, origin: str = "user") -> None:
        if origin != "user":
            raise PermissionEscalationError("The agent cannot revoke stored permissions.")
        self.store.delete_grant(grant_id)

    def record_approval(
        self, decision: PermissionDecision, approval: ApprovalOutcome
    ) -> None:
        try:
            self.store.add_approval(
                decision=approval.choice,
                task_id=decision.request.task_id,
                kind=decision.request.kind,
                category=decision.category,
                target=ApprovalRequest(decision).target,
                command=decision.request.command,
                risk=ApprovalRequest(decision).risk,
                fingerprint=decision.fingerprint,
            )
        except Exception:
            pass

    def view(self) -> "PermissionView":
        return PermissionView(self)


class PermissionView:
    """Read-only permission façade handed to sub-agents.

    Sub-agents can evaluate but can never add, widen or remove grants, and they
    cannot construct a new manager around the same store.
    """

    __slots__ = ("_manager",)

    def __init__(self, manager: PermissionManager):
        object.__setattr__(self, "_manager", manager)

    def __setattr__(self, name: str, value: Any) -> None:
        raise PermissionEscalationError("Sub-agents cannot modify permission objects.")

    def evaluate(self, request: ActionRequest) -> PermissionDecision:
        return self._manager.evaluate(request)

    def allowed_roots(self, workspace: str = "") -> tuple[str, ...]:
        return self._manager.allowed_roots(workspace)

    def escalate(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionEscalationError(
            "Sub-agents inherit the main agent's permissions and may not escalate them."
        )

    def add_grant(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionEscalationError("Sub-agents cannot create stored permissions.")

    def revoke_grant(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionEscalationError("Sub-agents cannot change stored permissions.")


__all__ = [
    "ABSOLUTE_ASK_CATEGORIES",
    "ActionRequest",
    "ApprovalOutcome",
    "ApprovalRequest",
    "GRANTABLE_CATEGORIES",
    "PermissionDecision",
    "PermissionEscalationError",
    "PermissionManager",
    "PermissionView",
    "RejectingApprovalBroker",
]
