# Hermes Agent — Customizations

Small, intentional changes to Hermes Agent to better suit my local setup and preferred agent behaviour.

This document records what I've changed, why, and where, so changes can be tracked and maintained across upstream updates.

## 1. Remove misleading profile paths from the system prompt

- **File:** `agent/system_prompt.py`
- **Function:** `_active_profile_line()`
- **Change:** Removed the host-side profile directory paths and cross-profile modification warnings. The function now identifies only the active profile.
- **Reason:** Hermes terminal and file tools run inside Docker and cannot necessarily access the host's `~/.hermes/` directories. The original instructions encouraged the agent to waste tokens trying to access or modify host files it could not reach.
- **Scope:** Global — applies to all profiles.

From:
```
def _active_profile_line(agent: Any) -> str:
    """Name the running profile so the agent doesn't conflate ``~/.hermes/skills``
    (default) with ``~/.hermes/profiles/<active>/skills``.  Resolved from the
    agent's OWN home first (a build thread that lost the ContextVar would
    otherwise print "default" for a bot profile)."""
    _agent_home_path = _agent_home(agent)
    active_profile = _active_profile_name(agent, _ambient_file_safety_profile_name)
    if active_profile == "default":
        # With an explicit agent home, the default profile's data lives at the
        # ROOT (get_hermes_home() on a bound profile session is the PROFILE dir).
        # Without one, keep the ambient (patchable) resolution byte-identical.
        _root_str = str(get_default_hermes_root() if _agent_home_path is not None else get_hermes_home())
        return (
            "Active Hermes profile: default. Other profiles (if any) live "
            "under " + _root_str + "/profiles/<name>/. Each profile has its own "
            "skills/, plugins/, cron/, and memories/ that affect a different "
            "session than this one. Do not modify another profile's "
            "skills/plugins/cron/memories unless the user explicitly directs "
            "you to."
        )

```

To:

```def _active_profile_line(agent: Any) -> str:
    """Name the running profile so the agent doesn't conflate profile contexts."""
    active_profile = _active_profile_name(agent, _ambient_file_safety_profile_name)
    if active_profile == "default":
        return "Active Hermes profile: default."
    return f"Active Hermes profile: {active_profile}."
```

## 2. Soften task-completion wording for the confidant profile

- **File:** `agent/prompt_builder.py`
- **Constant:** `TASK_COMPLETION_GUIDANCE`
- **Function:** `_guidance_parts()`
- **Change:** For the `confidant` profile only, substitute **"If the user asks you to build,"** for **"When the user asks you to build,"** when assembling the guidance.
- **Reason:** Slightly soften the agent's default drive to perform work, without weakening task-completion expectations when work is explicitly requested.
- **Scope:** `confidant` profile only. All other profiles retain the original wording.

From:
```
def _guidance_parts(agent: Any) -> list[str]:
    """Universal + tool-aware + model-gated guidance blocks, each gated by its config.yaml key."""
    parts: list[str] = []
    if agent.valid_tool_names:
        parts += [
            text for flag, text in (
                ("_task_completion_guidance", TASK_COMPLETION_GUIDANCE),
                ("_parallel_tool_call_guidance", PARALLEL_TOOL_CALL_GUIDANCE),
            ) if getattr(agent, flag, True)
        ]
    parts.append(_tool_guidance_block(agent))  # None/empty entries are dropped by _join_tier
```
To:
```
def _guidance_parts(agent: Any) -> list[str]:
    """Universal + tool-aware + model-gated guidance blocks, each gated by its config.yaml key."""
    parts: list[str] = []
    if agent.valid_tool_names:
        active_profile = _active_profile_name(
            agent, _ambient_file_safety_profile_name
        )
        task_guidance = TASK_COMPLETION_GUIDANCE
        if active_profile == "confidant":
            task_guidance = task_guidance.replace(
                "When the user asks you to build,",
                "If the user asks you to build,",
                1,
            )
        parts += [
            text for flag, text in (
                ("_task_completion_guidance", task_guidance),
                ("_parallel_tool_call_guidance", PARALLEL_TOOL_CALL_GUIDANCE),
            ) if getattr(agent, flag, True)
        ]
    parts.append(_tool_guidance_block(agent))  # None/empty entries are dropped by _join_tier
```

## 3. Pin local Whisper transcription to GPU 1

- **File:** `tools/transcription_local.py`
- **Function:** `_create_whisper_model()`
- **Change:** When the requested device is `cuda`, set `device_index=1` when creating the faster-whisper model.
- **Reason:** Run local Whisper transcription on the GTX 1650 SUPER rather than the primary RTX 4070, preserving the main GPU's VRAM for LLM inference.
- **Scope:** Global — applies to CUDA-based local transcription.


## Future changes

Record each additional customization here, including its location, exact change, reason, and whether it applies globally or only to a particular profile.

