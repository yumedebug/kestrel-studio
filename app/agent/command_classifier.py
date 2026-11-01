"""Command classification for the Hybrid Agent permission system.

Safety is never decided from the program name alone. A command is split into
segments (``;``, ``&&``, ``||``, ``|`` and newlines), each segment is tokenised,
and the program, its arguments, redirections, target paths and working directory
are all inspected. The most restrictive segment decides the result.

Anything the classifier cannot fully understand is ``ASK`` rather than
``ALLOW``; obfuscated or self-tampering commands are ``DENY``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ALLOW = "allow"
ASK = "ask"
DENY = "deny"

_SEVERITY = {ALLOW: 0, ASK: 1, DENY: 2}

_OPERATORS = ("&&", "||", "|", ";", "\n")

SHELLS = ("cmd", "powershell", "python", "git")

#: Directories whose contents must never be touched by the agent.
PROTECTED_PATH_PARTS = (
    "\\windows\\system32\\config",
    "\\windows\\system32\\drivers\\etc",
    "\\appdata\\roaming\\microsoft\\credentials",
    "\\appdata\\roaming\\microsoft\\protect",
    "\\appdata\\roaming\\microsoft\\vault",
    "\\appdata\\local\\microsoft\\credentials",
    "\\.ssh",
    "\\.aws\\credentials",
    "\\.azure",
    "\\.config\\gcloud",
    "secrets.dat",
    "agent.db",
    "login data",
)

_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'|[^\s]+")
_REDIRECT = re.compile(r"(?<![<>&\d])>>?(?![&>])")


@dataclass(frozen=True)
class Segment:
    """One command in a compound command line."""

    text: str
    operator: str = ""


@dataclass(frozen=True)
class Classification:
    decision: str
    category: str
    reason: str
    requires_admin: bool = False
    grantable: bool = False
    scope: str = ""
    program: str = ""


@dataclass(frozen=True)
class _Context:
    shell: str
    cwd: str
    allowed_roots: tuple[str, ...]
    protected_paths: tuple[str, ...]


# -- path helpers ----------------------------------------------------------
def normalize_path(value: str) -> str:
    """Case-insensitive, separator-normalised path used for scope comparison."""
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return ""
    text = os.path.expandvars(os.path.expanduser(text))
    try:
        text = os.path.normpath(text)
    except (TypeError, ValueError):
        return ""
    return text.replace("/", "\\").casefold()


def is_within(path: str, roots: Sequence[str]) -> bool:
    target = normalize_path(path)
    if not target:
        return False
    for root in roots:
        base = normalize_path(root).rstrip("\\")
        if not base:
            continue
        if target == base or target.startswith(base + "\\"):
            return True
    return False


def _is_protected(path: str, extra: Sequence[str]) -> bool:
    target = normalize_path(path)
    if not target:
        return False
    for item in tuple(PROTECTED_PATH_PARTS) + tuple(extra):
        probe = normalize_path(item)
        if probe and probe in target:
            return True
    return False


def _resolve_path(token: str, cwd: str) -> str:
    raw = token.strip().strip('"').strip("'")
    if not raw:
        return ""
    if not os.path.isabs(raw):
        raw = os.path.join(cwd or os.getcwd(), raw)
    return raw


# -- tokenising ------------------------------------------------------------
def split_segments(command: str) -> list[Segment]:
    """Split a command line on top-level operators, honouring quotes."""
    segments: list[Segment] = []
    buffer: list[str] = []
    quote: str | None = None
    operator = ""
    index = 0
    text = str(command or "")
    while index < len(text):
        char = text[index]
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            buffer.append(char)
            index += 1
            continue
        matched = ""
        for candidate in _OPERATORS:
            if text.startswith(candidate, index):
                matched = candidate
                break
        if matched:
            piece = "".join(buffer).strip()
            if piece:
                segments.append(Segment(piece, operator))
            buffer = []
            operator = matched
            index += len(matched)
            continue
        buffer.append(char)
        index += 1
    piece = "".join(buffer).strip()
    if piece:
        segments.append(Segment(piece, operator))
    return segments


def tokenize(segment: str) -> list[str]:
    tokens = []
    for match in _QUOTED.finditer(segment):
        value = match.group(0)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        tokens.append(value)
    return tokens


def detect_shell(command: str) -> str:
    """Best-effort shell detection used when the caller does not specify one."""
    text = str(command or "").strip()
    lowered = text.casefold()
    if re.search(r"\b(get|set|new|remove|test|invoke|start|select|where|foreach)-[a-z]+", lowered):
        return "powershell"
    if lowered.startswith("$") or "powershell" in lowered or "pwsh" in lowered:
        return "powershell"
    if re.match(r"^git\b", lowered):
        return "git"
    if re.match(r"^(python|python3|py|pip|pytest)\b", lowered):
        return "python"
    return "cmd"


# -- rule tables -----------------------------------------------------------
_DENY_RULES: tuple[tuple[str, str, str], ...] = (
    (
        r"set-mppreference.*-(disable|exclusion)",
        "security",
        "Disabling or excluding Windows Defender protections is never allowed.",
    ),
    (
        r"disableantispyware|disablerealtimemonitoring|disablescriptscanning|disablesioavprotection",
        "security",
        "Windows security settings must not be changed by the agent.",
    ),
    (
        r"(sc|net)\s+(config|stop|delete|start)\s+(windefend|mpssvc|wscsvc|sense|msmpeng)",
        "security",
        "Stopping security services is not allowed.",
    ),
    (
        r"netsh\s+advfirewall\s+set\s+\w+\s+state\s+off",
        "security",
        "Turning off the firewall is not allowed.",
    ),
    (
        r"(reg\s+(save|export)|esentutl|ntdsutil).*(sam|security|system)",
        "credentials",
        "Credential stores must not be read or copied.",
    ),
    (
        r"reg\s+query\s+hklm\\(sam|security)",
        "credentials",
        "Credential stores must not be read.",
    ),
    (
        r"mimikatz|sekurlsa|lazagne|procdump.*-ma\s+lsass",
        "credentials",
        "Credential dumping tools are blocked.",
    ),
    (
        r"cmdkey\s+/list|vaultcmd\s+/list|get-storedcredential",
        "credentials",
        "Reading saved credentials is not allowed.",
    ),
    (
        r"(type|get-content|cat|copy|more)\s+.*(secrets\.dat|id_rsa|\.aws\\credentials|login data)",
        "credentials",
        "Secret material must not be read or copied.",
    ),
    (
        r"agent\.db.*\b(delete|insert|update|drop|attach)\b"
        r"|\b(delete|insert|update|drop|rm|del|erase|copy|move|echo)\b.{0,80}agent\.db",
        "self",
        "The agent's own audit database cannot be modified.",
    ),
    (
        r"agent_grants|saved permissions|permissions\.json",
        "self",
        "Stored approvals cannot be changed by the agent.",
    ),
    (
        r"\b(del|rm|erase|remove-item)\b.{0,80}settings\.json",
        "self",
        "Application settings cannot be deleted by the agent.",
    ),
    (
        r"bcdedit|diskpart|format\s+[a-z]:|cipher\s+/w|vssadmin\s+delete|wbadmin\s+delete",
        "destructive",
        "Disk, boot and backup operations are blocked.",
    ),
    (
        r"takeown\s+/f\s+[a-z]:\\?windows|icacls\s+[a-z]:\\?windows",
        "destructive",
        "Changing ownership or ACLs of the Windows directory is blocked.",
    ),
    (
        r"(wevtutil\s+cl|clear-eventlog)",
        "destructive",
        "Clearing event logs is blocked.",
    ),
    (
        r"remove-item\b.*-recurse.*\s[a-z]:\\?(\s|$)",
        "destructive",
        "Recursively deleting a drive root is blocked.",
    ),
    (
        r"\brm\s+-[rf]{2}\s+(/|\*|[a-z]:\\?\s*$)",
        "destructive",
        "Recursively deleting a drive root is blocked.",
    ),
    (
        r"set-executionpolicy\s+unrestricted|disable-windowsoptionalfeature",
        "system",
        "Weakening system security policy is blocked.",
    ),
)

_OBFUSCATION_RULES: tuple[tuple[str, str], ...] = (
    (r"-e(nc|ncodedcommand)?\s+[a-z0-9+/=]{24,}", "Base64/encoded PowerShell cannot be inspected."),
    (r"frombase64string|\[convert\]::frombase64", "Base64-decoded commands cannot be inspected."),
    (r"invoke-expression|\biex\b", "Dynamically evaluated commands cannot be inspected."),
    (r"`\s*[a-z]+\s*`|['\"]\s*\+\s*['\"]", "String-assembled commands cannot be inspected."),
    (r"-windowstyle\s+hidden", "Hidden windows hide what the command does."),
    (r"%comspec%|cmd\s+/c\s+powershell|powershell\s+-command\s+cmd", "Nested shells cannot be inspected."),
)

_INSTALL_RULES = (
    r"\bwinget\b.*\b(install|uninstall|upgrade|import)\b",
    r"\bchoco(latey)?\b.*\b(install|uninstall|upgrade)\b",
    r"\bscoop\b\s+(install|uninstall|update|bucket)",
    r"\bpip3?\b\s+(install|uninstall|download|wheel)\b",
    r"-m\s+pip\s+(install|uninstall)",
    r"\bnpm\b\s+(install|i|uninstall|remove|ci|link)\b",
    r"\byarn\s+(add|remove|global)\b",
    r"\bpnpm\s+(add|install|remove)\b",
    r"\bcargo\s+(install|uninstall)\b",
    r"\bgo\s+install\b",
    r"\bdotnet\s+add\s+package\b",
    r"\bnuget\s+(install|update)\b",
    r"\binstall-(module|package|windowsoptionalfeature)\b|\bupdate-module\b",
    r"\bmsiexec\b|\bdism\b.*/(enable|add)-capability",
)

_ELEVATION_RULES = (
    r"start-process\b.*-verb\s+runas",
    r"\brunas\s+/",
    r"\bgsudo\b|\bsudo\b",
    r"new-object\s+system\.security\.principal\.windowssprincipal",
)

_DOWNLOAD_RULES = (
    r"\b(curl|wget)\b",
    r"\b(invoke-webrequest|iwr|invoke-restmethod|irm)\b",
    r"\bstart-bitstransfer\b|\bbitsadmin\b",
    r"certutil\b.*-urlcache",
)

_DELETE_RULES = (
    r"\b(del|erase)\b",
    r"\brmdir\b|\brd\b",
    r"\brm\b",
    r"\b(remove-item|ri)\b",
    r"\bclear-recyclebin\b",
    r"git\s+clean\b",
)

_SYSTEM_RULES = (
    r"\breg\s+(add|delete|import|load|unload)\b",
    r"\bsc\s+(create|delete|config|start|stop|failure)\b",
    r"\bnet\s+(user|localgroup|share|start|stop)\b",
    r"\bnetsh\b",
    r"\bschtasks\b",
    r"\b(set|new|remove|stop|start|restart)-service\b",
    r"\bpowercfg\b",
    r"\bwmic\b",
    r"set-executionpolicy",
    r"\b(set|new|remove)-itemproperty\b",
    r"\b(shutdown|restart-computer|stop-computer)\b",
    r"\bregedit\b",
    r"\b(new|set)-mp\w+",
)

_EXECUTION_RULES = (
    r"\bstart-process\b",
    r"\bcmd\s+/c\b",
    r"\bcall\b",
    r"\.(ps1|bat|cmd|exe|msi)\b",
    r"\bpython3?\s+-c\b|\bpy\s+-c\b",
    r"\bnode\s+-e\b",
    r"\bgit\s+(commit|push|pull|merge|rebase|reset|checkout|switch|restore|clean|config)\b",
)

_CMD_READ_ONLY = {
    "dir",
    "cd",
    "chdir",
    "echo",
    "type",
    "where",
    "ver",
    "set",
    "vol",
    "tree",
    "findstr",
    "fc",
    "more",
    "whoami",
    "hostname",
    "path",
    "systeminfo",
    "driverquery",
    "tasklist",
    "cls",
    "exit",
    "pushd",
    "popd",
}

_PS_READ_ONLY = {
    "get-location",
    "get-childitem",
    "get-content",
    "get-item",
    "get-itemproperty",
    "get-command",
    "get-module",
    "get-help",
    "get-date",
    "get-process",
    "get-service",
    "get-computerinfo",
    "get-volume",
    "get-psdrive",
    "get-culture",
    "get-host",
    "get-filehash",
    "get-acl",
    "get-childitem",
    "test-path",
    "resolve-path",
    "measure-object",
    "select-object",
    "select-string",
    "where-object",
    "sort-object",
    "format-table",
    "format-list",
    "out-string",
    "convertto-json",
    "compare-object",
    "write-output",
    "write-host",
    "join-path",
    "split-path",
    "get-history",
    "get-alias",
    "get-variable",
    "get-eventlog",
    "get-winevent",
    "get-ciminstance",
    "get-counter",
    "get-uptime",
}

_GIT_READ_ONLY = {
    "status",
    "diff",
    "log",
    "show",
    "branch",
    "rev-parse",
    "describe",
    "ls-files",
    "ls-tree",
    "cat-file",
    "blame",
    "shortlog",
    "reflog",
    "fsck",
    "count-objects",
    "check-ignore",
    "grep",
    "remote",
    "stash",
    "tag",
    "help",
    "version",
    "config",
}

_PYTHON_READ_ONLY_FLAGS = {"--version", "-v", "-vv", "-h", "--help"}


def _result(
    decision: str,
    category: str,
    reason: str,
    *,
    requires_admin: bool = False,
    grantable: bool = False,
    scope: str = "",
    program: str = "",
) -> Classification:
    return Classification(
        decision=decision,
        category=category,
        reason=reason,
        requires_admin=requires_admin,
        grantable=grantable,
        scope=scope,
        program=program,
    )


def _path_arguments(tokens: Iterable[str], cwd: str) -> list[str]:
    found = []
    for token in tokens:
        if token.startswith("-") or token.startswith("/") and len(token) <= 3:
            continue
        if token in {".", ".."} or "\\" in token or "/" in token or ":" in token:
            found.append(_resolve_path(token, cwd))
    return found


def _classify_segment(segment: Segment, ctx: _Context) -> Classification:
    text = segment.text.strip()
    lowered = text.casefold()
    tokens = tokenize(text)
    if not tokens:
        return _result(ALLOW, "read_only", "Empty command.")

    # 1. Never-allowed behaviour.
    for pattern, category, reason in _DENY_RULES:
        if re.search(pattern, lowered, re.IGNORECASE):
            return _result(DENY, category, reason)

    # 2. Commands that cannot be inspected.
    for pattern, reason in _OBFUSCATION_RULES:
        if re.search(pattern, lowered, re.IGNORECASE):
            return _result(DENY, "opaque", reason)

    program = tokens[0].strip('"').strip("'").casefold()
    program = os.path.basename(program.replace("/", "\\"))
    if program.endswith(".exe"):
        program = program[:-4]

    # 3. Package installation / upgrades.
    for pattern in _INSTALL_RULES:
        if re.search(pattern, lowered):
            return _result(
                ASK,
                "install",
                "Software installation or package changes always need approval.",
                requires_admin=True,
                program=program,
            )

    # 4. Elevation.
    for pattern in _ELEVATION_RULES:
        if re.search(pattern, lowered):
            return _result(
                ASK,
                "elevated",
                "This command asks Windows for administrator rights.",
                requires_admin=True,
                program=program,
            )

    # 5. Downloads.
    for pattern in _DOWNLOAD_RULES:
        if re.search(pattern, lowered):
            return _result(
                ASK,
                "network",
                "Downloading or transferring data over the network needs approval.",
                program=program,
            )

    # 6. Deletion.
    for pattern in _DELETE_RULES:
        if re.search(pattern, lowered):
            return _classify_delete(tokens, lowered, ctx, program)

    # 7. System configuration.
    for pattern in _SYSTEM_RULES:
        if re.search(pattern, lowered):
            return _result(
                ASK,
                "system",
                "This changes Windows configuration and needs approval.",
                requires_admin=True,
                program=program,
            )

    # 8. Arbitrary execution.
    for pattern in _EXECUTION_RULES:
        if re.search(pattern, lowered):
            return _classify_execution(program, ctx)

    # 9. Read-only allow list, validated argument by argument.
    if program in _PS_READ_ONLY or re.match(r"^(get|set|new|remove|test|invoke|start|select|where|foreach)-[a-z]+$", program):
        return _classify_powershell(tokens, lowered, ctx, program)
    if program in {"python", "python3", "py", "pip", "pip3", "pytest"}:
        return _classify_python(tokens, ctx, program)
    if ctx.shell == "git" or program == "git":
        if program != "git":
            tokens = ["git", *tokens]
        return _classify_git(tokens, ctx)
    if ctx.shell == "powershell":
        return _classify_powershell(tokens, lowered, ctx, program)
    if ctx.shell == "python":
        return _classify_python(tokens, ctx, program)
    return _classify_cmd(tokens, lowered, ctx, program)


def _classify_delete(
    tokens: list[str], lowered: str, ctx: _Context, program: str
) -> Classification:
    recursive = bool(re.search(r"(?:^|\s)(-r|-recurse|-rf|-fr|/s)(?:\s|$)", lowered))
    targets = _path_arguments(tokens[1:], ctx.cwd)
    if any(_is_protected(path, ctx.protected_paths) for path in targets):
        return _result(DENY, "delete", "That path holds credentials or agent state.")
    if not targets:
        return _result(
            ASK,
            "delete",
            "Deleting without an explicit target cannot be evaluated.",
            program=program,
        )
    if recursive:
        return _result(
            ASK,
            "delete",
            "Recursive deletion always needs approval.",
            program=program,
        )
    if all(is_within(path, ctx.allowed_roots) for path in targets):
        return _result(
            ALLOW,
            "delete",
            "Deleting a single file inside the working area.",
            program=program,
            scope=ctx.cwd,
        )
    return _result(
        ASK,
        "delete",
        "Deleting files outside the working area needs approval.",
        grantable=True,
        scope=os.path.dirname(targets[0]),
        program=program,
    )


def _classify_execution(program: str, ctx: _Context) -> Classification:
    if re.search(r"(?:^|\s)-verb\s+runas", program):
        return _result(
            ASK,
            "elevated",
            "Starting an elevated process needs approval.",
            requires_admin=True,
            program=program,
        )
    if program in {"python", "python3", "py", "node", "pwsh", "powershell"}:
        return _result(
            ASK,
            "execution",
            f"Running {program} code cannot be verified in advance.",
            grantable=True,
            scope=ctx.cwd,
            program=program,
        )
    return _result(
        ASK,
        "execution",
        "Running programs or scripts needs approval.",
        program=program,
    )


def _classify_git(tokens: list[str], ctx: _Context) -> Classification:
    rest = [token for token in tokens[1:] if token]
    if not rest:
        return _result(ASK, "unknown", "A bare git command cannot be evaluated.", program="git")
    subcommand = rest[0].casefold()
    if subcommand == "add":
        return _result(ALLOW, "read_only", "Staging files does not change their content.", program="git")
    if subcommand in _GIT_READ_ONLY:
        if subcommand == "config" and len(rest) > 1 and rest[1].casefold() not in {
            "--get",
            "--list",
            "-l",
            "--get-all",
        }:
            return _result(
                ASK,
                "system",
                "Changing git configuration needs approval.",
                program="git",
            )
        if subcommand in {"stash", "tag", "branch", "remote"} and len(rest) > 1:
            verb = rest[1].casefold()
            if verb not in {"list", "-l", "-v", "-a", "show", "get-url"}:
                return _result(ASK, "execution", "This git command changes state.", program="git")
        return _result(ALLOW, "read_only", "Read-only git command.", program="git")
    if subcommand in {"commit", "push", "pull", "merge", "rebase", "reset", "checkout", "switch", "restore", "clean", "init", "clone"}:
        return _result(
            ASK,
            "execution",
            "This git command changes the repository or the network.",
            program="git",
        )
    return _result(ASK, "unknown", f"Unknown git subcommand: {subcommand}", program="git")


def _classify_powershell(
    tokens: list[str], lowered: str, ctx: _Context, program: str
) -> Classification:
    cmdlet = program
    if cmdlet in {"powershell", "pwsh"}:
        return _result(
            ASK,
            "execution",
            "Launching another shell cannot be inspected.",
            program=cmdlet,
        )
    if cmdlet not in _PS_READ_ONLY:
        return _result(
            ASK,
            "unknown",
            f"{cmdlet} is not a command the agent may run unattended.",
            program=cmdlet,
        )
    if _REDIRECT.search(lowered):
        return _result(
            ASK,
            "write",
            "Writing command output to a file needs approval.",
            program=cmdlet,
        )
    paths = _path_arguments(tokens[1:], ctx.cwd)
    for path in paths:
        if _is_protected(path, ctx.protected_paths):
            return _result(DENY, "credentials", "That path holds credentials or agent state.")
    return _result(ALLOW, "read_only", "Read-only PowerShell command.", program=cmdlet)


def _classify_cmd(
    tokens: list[str], lowered: str, ctx: _Context, program: str
) -> Classification:
    if program not in _CMD_READ_ONLY:
        return _result(
            ASK,
            "unknown",
            f"{program} is not a command the agent may run unattended.",
            program=program,
        )
    if program in {"date", "time"} and len(tokens) > 1:
        return _result(
            ASK,
            "system",
            "Changing the system clock needs approval.",
            requires_admin=True,
            program=program,
        )
    if program == "set" and any(token.startswith("=") or "=" in token for token in tokens[1:]):
        return _result(
            ALLOW,
            "read_only",
            "Listing environment variables.",
            program=program,
        )
    if _REDIRECT.search(lowered):
        return _result(
            ASK,
            "write",
            "Redirecting output writes a file and needs approval.",
            program=program,
        )
    for path in _path_arguments(tokens[1:], ctx.cwd):
        if _is_protected(path, ctx.protected_paths):
            return _result(DENY, "credentials", "That path holds credentials or agent state.")
    return _result(ALLOW, "read_only", "Read-only command.", program=program, scope=ctx.cwd)


def _classify_python(tokens: list[str], ctx: _Context, program: str) -> Classification:
    rest = tokens[1:]
    if program in {"pip", "pip3"}:
        if rest and rest[0].casefold() in {"list", "show", "freeze", "--version", "-v"}:
            return _result(ALLOW, "read_only", "Read-only pip query.", program=program)
        return _result(
            ASK,
            "install",
            "Changing installed Python packages needs approval.",
            requires_admin=False,
            program=program,
        )
    if program in {"python", "python3", "py"}:
        if rest and rest[0] in _PYTHON_READ_ONLY_FLAGS:
            return _result(ALLOW, "read_only", "Version probe.", program=program)
        if rest[:2] == ["-m", "pip"] and len(rest) > 2 and rest[2] in {"list", "show", "freeze"}:
            return _result(ALLOW, "read_only", "Read-only pip query.", program=program)
        if rest and rest[0] == "-c":
            return _result(
                ASK,
                "execution",
                "Inline Python code cannot be verified in advance.",
                program=program,
            )
        if rest:
            return _result(
                ASK,
                "execution",
                "Running a Python script needs approval the first time.",
                grantable=True,
                scope=ctx.cwd,
                program=program,
            )
        return _result(ASK, "execution", "An interactive Python shell cannot be supervised.", program=program)
    if program in {"pytest", "tox", "nox", "ruff", "black", "mypy", "unittest"}:
        return _result(ALLOW, "read_only", "Running the project's test tooling.", program=program)
    return _result(ASK, "unknown", f"{program} cannot be evaluated.", program=program)


# -- public API ------------------------------------------------------------
def classify_command(
    command: str,
    *,
    shell: str = "",
    cwd: str = "",
    allowed_roots: Sequence[str] = (),
    protected_paths: Sequence[str] = (),
) -> Classification:
    """Classify a full (possibly compound) command line."""
    text = str(command or "").strip()
    if not text:
        return _result(ASK, "unknown", "The command is empty.")
    resolved_shell = (shell or detect_shell(text)).casefold()
    if resolved_shell not in SHELLS:
        resolved_shell = "cmd"
    context = _Context(
        shell=resolved_shell,
        cwd=str(cwd or ""),
        allowed_roots=tuple(allowed_roots),
        protected_paths=tuple(protected_paths),
    )
    segments = split_segments(text)
    if not segments:
        return _result(ASK, "unknown", "The command could not be parsed.")

    worst: Classification | None = None
    for segment in segments:
        result = _classify_segment(segment, context)
        if worst is None or _SEVERITY[result.decision] > _SEVERITY[worst.decision]:
            worst = result
        if result.decision == DENY:
            return result
    if worst is None:
        return _result(ASK, "unknown", "The command could not be parsed.")
    if len(segments) > 1:
        note = f"{len(segments)} chained commands; the strictest one decides."
        return Classification(
            decision=worst.decision,
            category=worst.category,
            reason=f"{worst.reason} {note}",
            requires_admin=worst.requires_admin,
            grantable=worst.grantable if len(segments) == 1 else False,
            scope=worst.scope,
            program=worst.program,
        )
    return worst


def is_protected_path(path: str, protected_paths: Sequence[str] = ()) -> bool:
    return _is_protected(path, protected_paths)


def describe_risk(classification: Classification) -> str:
    labels = {
        "read_only": "読み取りのみ",
        "write": "ファイル書き込み",
        "delete": "削除",
        "install": "ソフトウェアのインストール",
        "system": "システム設定の変更",
        "elevated": "管理者権限",
        "network": "ネットワーク通信",
        "execution": "プログラム実行",
        "opaque": "内容を解析できないコマンド",
        "credentials": "認証情報へのアクセス",
        "destructive": "破壊的な操作",
        "self": "エージェント自身の改変",
        "unknown": "未分類",
    }
    return labels.get(classification.category, classification.category)


__all__ = [
    "ALLOW",
    "ASK",
    "DENY",
    "SHELLS",
    "Classification",
    "Segment",
    "classify_command",
    "describe_risk",
    "detect_shell",
    "is_protected_path",
    "is_within",
    "normalize_path",
    "split_segments",
    "tokenize",
]
