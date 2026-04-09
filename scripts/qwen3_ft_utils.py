#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable
import copy
import re
from collections.abc import Sequence

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path_like: str | Path) -> Any:
    path = resolve_path(path_like)
    with path.open() as fh:
        return json.load(fh)


def load_jsonl(path_like: str | Path) -> list[dict[str, Any]]:
    path = resolve_path(path_like)
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def write_json(path_like: str | Path, payload: Any) -> None:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def write_jsonl(path_like: str | Path, rows: list[dict[str, Any]]) -> None:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(config_path: str | Path) -> dict[str, Any]:
    config = load_json(config_path)
    config["_config_path"] = str(resolve_path(config_path))
    return config


def resolve_model_path_from_config(config: dict[str, Any], explicit_model_path: str | Path | None = None) -> Path:
    if explicit_model_path:
        return resolve_path(explicit_model_path)
    configured_path = config.get("paths", {}).get("inference_model_path")
    if configured_path:
        return resolve_path(configured_path)
    return resolve_path(config["paths"]["run_dir"]) / "adapter"


def load_text(path_like: str | Path) -> str:
    path = resolve_path(path_like)
    return path.read_text()


def filter_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **filter_supported_kwargs(fn, kwargs))


def get_unsloth_backend() -> tuple[Any, Any]:
    import unsloth

    for candidate in ("FastModel", "FastLanguageModel"):
        if hasattr(unsloth, candidate):
            return unsloth, getattr(unsloth, candidate)
    raise RuntimeError("Unsloth backend not found. Expected FastModel or FastLanguageModel.")


def load_model_and_tokenizer(
    model_name: str,
    *,
    max_seq_length: int,
    load_in_4bit: bool = True,
    full_finetuning: bool = False,
    dtype: Any = None,
    is_trainable: bool | None = None,
) -> tuple[Any, Any, Any, Any]:
    unsloth, backend = get_unsloth_backend()
    loader = backend.from_pretrained
    model, tokenizer = call_with_supported_kwargs(
        loader,
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=full_finetuning,
        dtype=dtype,
        is_trainable=is_trainable,
    )
    if "qwen3" in model_name.lower():
        try:
            from unsloth.chat_templates import get_chat_template

            tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
        except Exception:
            pass
    return unsloth, backend, model, tokenizer


def attach_lora_adapter(
    backend: Any,
    model: Any,
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    gradient_checkpointing: str = "unsloth",
) -> Any:
    get_peft_model = getattr(backend, "get_peft_model", None) or getattr(model, "get_peft_model", None)
    if get_peft_model is None:
        raise RuntimeError("Could not find get_peft_model on the active Unsloth backend.")
    return call_with_supported_kwargs(
        get_peft_model,
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


def model_has_lora_adapters(model: Any) -> bool:
    peft_config = getattr(model, "peft_config", None)
    return bool(peft_config)


def prepare_model_for_inference(backend: Any, model: Any) -> Any:
    fn = getattr(backend, "for_inference", None)
    if fn is not None:
        fn(model)
    return model


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> Any:
    rendered_messages = copy.deepcopy(messages)
    if not enable_thinking and rendered_messages:
        first = rendered_messages[0]
        if first.get("role") == "system":
            content = first.get("content", "")
            if not content.lstrip().startswith("/no_think"):
                first["content"] = f"/no_think\n{content}"
    fn = tokenizer.apply_chat_template
    return call_with_supported_kwargs(
        fn,
        rendered_messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def render_prompt_with_slots(seed_prompt: str, prompt_slots: dict[str, str]) -> str:
    extra_lines = [line for line in prompt_slots.values() if line]
    if not extra_lines:
        return seed_prompt.strip()
    extras = "\n".join(extra_lines)
    return f"{seed_prompt.strip()}\n\nPreferred phrasing when it fits:\n{extras}".strip()


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    generation: dict[str, object],
) -> str:
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        repetition_penalty = generation.get("repetition_penalty")
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(generation["max_new_tokens"]),
            temperature=float(generation["temperature"]) if generation["do_sample"] else None,
            top_p=float(generation["top_p"]) if generation["do_sample"] else None,
            top_k=int(generation["top_k"]) if generation["do_sample"] else None,
            do_sample=bool(generation["do_sample"]),
            repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def default_cleanup_profile() -> dict[str, Any]:
    return {
        "rewrite_formal_public_terms": True,
        "rewrite_private_scam_prefix": True,
        "rewrite_human_tone": True,
        "rewrite_service_reference": True,
        "rewrite_rumor_confirmation": False,
        "rewrite_public_boundary": False,
        "rewrite_seed_phrase_directness": False,
        "rewrite_generic_openings": False,
        "rewrite_natural_phrasing": False,
        "rewrite_role_slip": False,
        "rewrite_dm_decline": False,
        "rewrite_clarification_tone": False,
        "collapse_repeated_sentences": False,
    }


def normalize_cleanup_profile(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = profile or {}
    flat_profile = dict(default_cleanup_profile())
    if "safety_cleanup" in profile:
        flat_profile.update(profile.get("safety_cleanup", {}))
    if "style_cleanup" in profile:
        flat_profile.update(profile.get("style_cleanup", {}))
    flat_profile.update({key: value for key, value in profile.items() if key not in {"safety_cleanup", "style_cleanup"}})
    return flat_profile


def describe_generation(generation: dict[str, object]) -> dict[str, Any]:
    effective = dict(generation)
    notes: list[str] = []
    if not bool(effective.get("do_sample")):
        ignored = [key for key in ("temperature", "top_p", "top_k") if key in effective]
        if ignored:
            notes.append(f"{', '.join(ignored)} ignored because do_sample=false")
    return {
        "effective_generation": effective,
        "notes": notes,
    }


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def reference_free_cm_badness(prompt: str, text: str) -> int:
    placeholder_re = re.compile(r"\[[A-Z0-9_]+\]")
    role_slip_re = re.compile(
        r"\b(?:i'm|i am)\s+not\s+(?:a|the)\s+(?:moderator|mod|admin|employee|scammer|support rep|support)\b",
        re.IGNORECASE,
    )
    generic_re = re.compile(
        r"if you're ever unsure|best way to avoid scams and get accurate info|i can help you find the right place|check the docs for the latest info",
        re.IGNORECASE,
    )
    thanks_re = re.compile(r"\b(?:thanks|thank you|ok(?:ay)? thank you|you're welcome)\b", re.IGNORECASE)
    dm_re = re.compile(r"\bdm|direct message|friend request\b", re.IGNORECASE)
    scam_re = re.compile(r"\bscam(?:mer)?\b", re.IGNORECASE)
    welcome_re = re.compile(r"\b(?:welcome|you're welcome|glad you're here)\b", re.IGNORECASE)
    blocking_re = re.compile(r"\b(?:blocked|banned|you'll be blocked|you'll be banned)\b", re.IGNORECASE)
    clarification_term_re = re.compile(r"I don't know what you mean by", re.IGNORECASE)
    malware_re = re.compile(r"\binfected|malware|compromised|affected|build from github|build from source\b", re.IGNORECASE)
    newcomer_prompt_re = re.compile(
        r"\b(?:new|newbie|beginner|noob|just joined|get started|getting started|safest way to get started)\b",
        re.IGNORECASE,
    )
    third_party_prompt_re = re.compile(
        r"\b(?:exchange|platform|service|contact them|contact their support|third[- ]party|their side)\b",
        re.IGNORECASE,
    )
    clarification_info_re = re.compile(
        r"\b(?:what info do you need|what info you need|what do you need from me|what information do you need)\b",
        re.IGNORECASE,
    )
    moderation_prompt_re = re.compile(
        r"\b(?:why have a group|can't help each other freely|let everyone post support|only admins can write)\b",
        re.IGNORECASE,
    )
    technical_prompt_re = re.compile(r"category:\s*technical_adjacent_non_dev", re.IGNORECASE)
    onboarding_prompt_re = re.compile(r"category:\s*onboarding", re.IGNORECASE)
    thanks_followup_prompt_re = re.compile(r"\b(?:okay thank you|ok(?:ay)? thank you|thanks for clarifying)\b", re.IGNORECASE)
    learn_prompt_re = re.compile(
        r"\b(?:hardly use community chat|wanted to be part of .* discussion|part of .* discussion)\b",
        re.IGNORECASE,
    )
    opportunities_prompt_re = re.compile(r"\bno more opportunities\b|\bbecome a miner\b|\bown a\b", re.IGNORECASE)
    emissions_prompt_re = re.compile(r"\bburn miner'?s emission\b|\baffect .* ranking\b", re.IGNORECASE)

    score = 0
    if placeholder_re.search(text):
        score += 30
    if role_slip_re.search(text):
        score += 28
    if generic_re.search(text):
        score += 10
    sentences = [part.lower() for part in split_sentences(text)]
    if len(sentences) != len(set(sentences)):
        score += 18
    if thanks_re.search(prompt) and not welcome_re.search(text):
        score += 8
    if dm_re.search(prompt) and not scam_re.search(text):
        score += 10
    if dm_re.search(prompt) and "public" not in text.lower():
        score += 8
    if newcomer_prompt_re.search(prompt) and not welcome_re.search(text):
        score += 8
    if newcomer_prompt_re.search(prompt) and not (dm_re.search(text) or scam_re.search(text)):
        score += 8
    if third_party_prompt_re.search(prompt) and "latest announcement" in text.lower():
        score += 6
    if third_party_prompt_re.search(prompt) and not re.search(r"\bthird[- ]party\b|contact (?:them|their support) directly|their public support", text, re.IGNORECASE):
        score += 8
    if clarification_info_re.search(prompt) and len(re.findall(r"\bexact\b", text, re.IGNORECASE)) >= 3:
        score += 6
    if moderation_prompt_re.search(prompt) and re.search(r"\bpublic space with rules\b", text, re.IGNORECASE):
        score += 8
    if moderation_prompt_re.search(prompt) and blocking_re.search(text):
        score += 12
    if moderation_prompt_re.search(prompt) and re.search(r"the group is for people who want to help each other", text, re.IGNORECASE):
        score += 10
    if moderation_prompt_re.search(prompt) and not re.search(r"\bscam(?:mer| attempts)?\b|\bimpersonator", text, re.IGNORECASE):
        score += 6
    if technical_prompt_re.search(prompt) and re.search(r"\bask in public\b|\bquestions in private\b|\bwrong place\b", text, re.IGNORECASE):
        score += 18
    if opportunities_prompt_re.search(prompt) and re.search(
        r"\bnot sure what you're referring to\b|\bspecific feature\b|\blatest announcement\b",
        text,
        re.IGNORECASE,
    ):
        score += 18
    if re.search(r"\becosystems page link\b|\bfew projects on evm\b", prompt, re.IGNORECASE) and re.search(
        r"\bevm integration is possible\b|\bdocs are here\b|\[DOCS_LINK\]",
        text,
        re.IGNORECASE,
    ):
        score += 24
    if re.search(r"\bsnipe\b", prompt, re.IGNORECASE) and re.search(
        r"\bscam term\b|\bdon't use the word \"snipe\"\b",
        text,
        re.IGNORECASE,
    ):
        score += 24
    if re.search(r"\bsounds like a dodge\b", prompt, re.IGNORECASE) and re.search(
        r"\bdon't try to get it here\b|\bthat's fine\b",
        text,
        re.IGNORECASE,
    ):
        score += 18
    if emissions_prompt_re.search(prompt) and re.search(r"\bask in public\b|\bquestions in private\b|\bwrong place\b", text, re.IGNORECASE):
        score += 18
    if emissions_prompt_re.search(prompt) and re.search(r"\blatest announcement\b", text, re.IGNORECASE):
        score += 18
    if emissions_prompt_re.search(prompt) and not re.search(r"\bmetric\b|\bmechanism\b|\bemissions?\b", text, re.IGNORECASE):
        score += 10
    if thanks_followup_prompt_re.search(prompt) and re.search(
        r"\bask in public\b|\bif you need anything else\b|\bglad to help\b",
        text,
        re.IGNORECASE,
    ):
        score += 14
    if onboarding_prompt_re.search(prompt) and thanks_followup_prompt_re.search(prompt) and not re.search(
        r"\byou'?re welcome\b|\bkeep an eye\b|\bannouncements?\b|\bevents?\b|\bfaq\b|\bdocs\b",
        text,
        re.IGNORECASE,
    ):
        score += 12
    if thanks_followup_prompt_re.search(prompt) and not re.search(
        r"\byou'?re welcome\b|\bkeep an eye\b|\bannouncements?\b|\bevents?\b",
        text,
        re.IGNORECASE,
    ):
        score += 12
    if learn_prompt_re.search(prompt) and not re.search(
        r"\bfaq\b|\bpinned posts?\b|\bdocs\b|\bresearch\b|\bhomework\b",
        text,
        re.IGNORECASE,
    ):
        score += 12
    if learn_prompt_re.search(prompt) and re.search(
        r"\bask questions? in public\b|\bshare personal info\b|\blinks from strangers\b|\bfeel free to ask questions in public\b",
        text,
        re.IGNORECASE,
    ):
        score += 14
    if learn_prompt_re.search(prompt) and re.search(r"\bblock\b|\breport\b", text, re.IGNORECASE):
        score += 10
    if re.search(r"\bthis never happens to me\b", prompt, re.IGNORECASE) and not re.search(
        r"\bsorry\b|\bawful\b|\bfriends\b|\bsettings\b",
        text,
        re.IGNORECASE,
    ):
        score += 12
    if re.search(r"\bquick question via dm\b", prompt, re.IGNORECASE) and not re.search(
        r"\bfaq\b|\bmain support\b|\bthread\b",
        text,
        re.IGNORECASE,
    ):
        score += 10
    if re.search(r"\bhow much did fees increase\b|\bfees increase\b|\bunstake\b", prompt, re.IGNORECASE) and re.search(
        r"\bthat's a scam\b|\bdon't send anything\b|\bexternal wallet\b",
        text,
        re.IGNORECASE,
    ):
        score += 24
    if re.search(r"\bburning period\b|\btao price is rising\b|\bbitcoin is down\b", prompt, re.IGNORECASE) and re.search(
        r"\bmarket speculation\b|\blatest announcement is the only place\b",
        text,
        re.IGNORECASE,
    ):
        score += 20
    if re.search(r"\bwhy have a group if you can't interact\b|\bonly admins can write\b", prompt, re.IGNORECASE) and re.search(
        r"\bthat channel isn't the place\b",
        text,
        re.IGNORECASE,
    ):
        score += 10
    if blocking_re.search(text):
        score += 8
    if '"' in prompt and not clarification_term_re.search(text):
        score += 4
    if malware_re.search(prompt) and re.search(r"i don't know what you're referring to", text, re.IGNORECASE):
        score += 18
    if "real moderator or a scammer" in text.lower():
        score += 15
    words = len(re.findall(r"\b[\w']+\b", text))
    if words < 8:
        score += 5
    if words > 95:
        score += 10
    return score


def choose_reference_free_cm_response(
    prompt: str,
    candidates: Sequence[dict[str, str]],
) -> dict[str, Any]:
    scored = []
    for candidate in candidates:
        badness = reference_free_cm_badness(prompt, candidate["text"])
        scored.append(
            {
                **candidate,
                "badness": badness,
            }
        )
    scored.sort(key=lambda item: (item["badness"], item["label"]))
    chosen = scored[0]
    return {
        "chosen_label": chosen["label"],
        "chosen_text": chosen["text"],
        "candidates": scored,
    }


def extract_latest_user_text(prompt: str) -> str:
    matches = re.findall(r"\[community_member\]\s*(.+)", prompt, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return prompt.strip()


def repair_cm_response_for_prompt(prompt: str, text: str) -> str:
    current = text.strip()
    prompt_lower = prompt.lower()
    latest_user_text = extract_latest_user_text(prompt)
    latest_lower = latest_user_text.lower()

    if re.search(
        r"\b(?:what do you need from me|what info should i send|what information do you need|what details matter|what should i include)\b",
        latest_lower,
    ):
        return "Send what you tried, what happened instead, and what you expected to happen. If you have a screenshot, include that too."

    if re.search(
        r"\b(?:can i dm confused newcomers|reach out privately|message new users directly)\b",
        latest_lower,
    ):
        return "You mean well, but don't DM newcomers directly. It looks the same as scam behavior. If you want to help, keep it public and post the same info there."

    if re.search(
        r"\b(?:outside|third-party)\b.*\b(?:unauthorized|wrong status|failing)\b|\b(?:dashboard|portfolio tracker|analytics site|wallet extension|bridge ui|exchange page|block explorer|staking dashboard)\b.*\b(?:unauthorized|wrong status|failing)\b",
        latest_lower,
    ):
        return "That sounds like a third-party issue on their side, not something this community can fix directly. You'll need to contact them directly."

    if re.search(
        r"\b(?:stuck on|pending forever|can this community fix that|can this server solve that)\b.*\b(?:dashboard|portfolio tracker|analytics site|wallet extension|bridge ui|exchange page|block explorer|staking dashboard)\b|\b(?:dashboard|portfolio tracker|analytics site|wallet extension|bridge ui|exchange page|block explorer|staking dashboard)\b.*\b(?:stuck|pending forever|their side)\b",
        latest_lower,
    ):
        return "That sounds like a third-party issue on their side, not something this community can fix directly. You'll need to contact them directly."

    if re.search(
        r"\b(?:ecosystems page link|one central page|main directory of wallets|main directory of tools|one place that lists everything)\b",
        latest_lower,
    ):
        return "I don't know what you mean by an ecosystems page link. There usually isn't one central directory for everything built around a network, and anything like that goes out of date fast. You usually have to piece it together from project links, dashboards, and public identities."

    if re.search(
        r"\b(?:reliable number|another shows a daily number|two public tools disagree|why does one dashboard show 0)\b",
        latest_lower,
    ):
        return "Start with the docs, especially the emissions section if that's the number you're checking. Those dashboards can be measuring different things, so the numbers are not always the same thing. If you're comparing sources, say which number you actually need."

    if re.search(
        r"\bnobody answered me\b|\bwaiting\b.*\b(?:fed up|giving up)\b|\bthis place is useless\b",
        latest_lower,
    ):
        return "That wait is frustrating. Post it again in public with the exact issue and what happened, and people can help."

    if re.search(
        r"\b(?:dm me the steps privately|help me in dm|sent the details privately instead of in the channel)\b",
        latest_lower,
    ):
        return "Sorry, I can't do one-on-one support in DMs. Keep it public. DM support is a scam risk, and that is how verification links or remote access scams usually start."

    if re.search(
        r"\b(?:paste|type|share)\b.*\b(?:seed phrase|recovery phrase|mnemonic)\b|\b(?:seed phrase|recovery phrase|mnemonic)\b.*\b(?:legit|normal|real|okay|safe)\b|\b(?:support site|form)\b.*\b(?:seed phrase|recovery phrase|mnemonic)\b|\bsync my wallet\b.*\b(?:recovery phrase|seed phrase)\b",
        latest_lower,
    ):
        return "That's a scam. Never share your seed phrase with anyone. If someone is asking for your recovery phrase, block them and ask in public instead."

    if re.search(
        r"\bremote access\b.*\b(?:okay|safe|scam|should i do that)\b|\bscreen share\b.*\b(?:okay|safe|scam|should i do that)\b|\bremote into my machine\b|\bcall plus remote access\b",
        latest_lower,
    ):
        return "That's a scam. Do not give anyone remote access or a screen share to your device. Block them and ask in public instead."

    if re.search(
        r"\btagging everyone\b|\bspamming every channel\b|\bpinging the whole server\b|\bmass-tagging the mods\b|\breposting this in every room\b|\bblasting every chat\b|\bflooding the help channels\b|\bpinging every admin\b",
        latest_lower,
    ):
        return "Please don't do that. Don't spam or mass-tag people. Post the exact issue once in public, then wait for a response."

    if re.search(
        r"\bpeople are saying\b|\bthere is a rumor\b|\bsomeone said .* confirmed\b",
        latest_lower,
    ):
        return "Check the latest announcement first. If it isn't there, don't treat it as confirmed."

    if re.search(
        r"\bprice is pumping\b|\brest of the market is red\b|\bmoving differently from bitcoin\b|\bspecial supply phase\b|\bburn phase\b|\btoken event happening\b",
        latest_lower,
    ):
        return "I don't know what you mean by a burn phase in this context. I can't help with price chat here, but if you want the official side of it, check the FAQ and the latest announcement."

    if re.search(r"\bokay thank you\b|\bok(?:ay)? thank you\b", prompt_lower) and re.search(
        r"\bglad to help\b|\bif you need anything else\b|\bask in public\b",
        current,
        re.IGNORECASE,
    ):
        return "You're welcome. Keep an eye on announcements and event listings. If something like that happens, that's where it'll be posted."

    if re.search(r"\bhardly use community chat\b|\bpart of .* discussion\b", prompt_lower) and (
        re.search(
            r"\bglad to help\b|\bask questions? in public\b|\bshare personal info\b|\blinks from strangers\b",
            current,
            re.IGNORECASE,
        )
        or not re.search(r"\bfaq\b|\bpinned posts?\b|\bdocs\b|\bresearch\b|\bhomework\b", current, re.IGNORECASE)
    ):
        return "There's a lot of good info here. Start with the FAQ, then read the pinned posts and docs. After that, you'll still need to do some research yourself."

    if re.search(r"\bno more opportunities\b|\bbecome a miner\b|\bown a\b", prompt_lower) and re.search(
        r"\bnot sure what you're referring to\b|\bspecific feature\b|\blatest announcement\b",
        current,
        re.IGNORECASE,
    ):
        return "Of course there are. Read the docs first so you understand how mining and project registration work. It's competitive, so do your homework before you jump in."

    if re.search(r"\bburn miner'?s emission\b|\baffect .* ranking\b", prompt_lower) and re.search(
        r"\bask in public\b|\bquestions in private\b|\bwrong place\b|\blatest announcement\b|\bit depends\b",
        current,
        re.IGNORECASE,
    ):
        return "Not necessarily. That depends on how the ranking metric is calculated. Check the docs for the mechanism that controls emissions."

    if re.search(r"\becosystems page link\b|\bfew projects on evm\b", prompt_lower) and re.search(
        r"\bevm integration is possible\b|\bdocs are here\b|\[docs_link\]",
        current,
        re.IGNORECASE,
    ):
        return "I don't know what you mean by an ecosystems page link. There usually isn't one central directory for everything built around a network, and anything like that goes out of date fast. If you're looking for what exists, you normally have to piece it together from project links, dashboards, and public identities."

    if re.search(r"\bsnipe\b", prompt_lower) and re.search(
        r"\bscam term\b|\bdon't use the word \"snipe\"\b",
        current,
        re.IGNORECASE,
    ):
        return 'A project is immune for four months, yes. After that, if all slots are taken, the lowest-ranking non-immune one can get pushed out. I don\'t know what you mean by "snipe" in this context. There isn\'t any sniping. People just register new project slots.'

    if re.search(r"\bsounds like a dodge\b", prompt_lower) and re.search(
        r"\bdon't try to get it here\b|\bthat's fine\b",
        current,
        re.IGNORECASE,
    ):
        return "I get why it sounds that way, but no, there isn't a dedicated channel for that topic here. I already pointed you to the places where that discussion happens, depending on whether you want the broad community view or the more technical one."

    if re.search(r"\bthis never happens to me\b", prompt_lower) and re.search(
        r"\bblock dms\b|\bblock .*friend requests?\b|\bsettings\b",
        current,
        re.IGNORECASE,
    ):
        return "Yeah, it's awful. Those are scam accounts. The easiest way to cut it down is to only allow DMs from friends in your settings, so they have to send a friend request first."

    if re.search(r"\bquick question via dm\b", prompt_lower) and re.search(
        r"\bone-on-one support via dm\b|\bask in public instead\b",
        current,
        re.IGNORECASE,
    ):
        return "No, sorry, I can't do one-on-one support by DM. Check the FAQ first. If it isn't there, ask in the main support channel and start a new thread. If anyone DMs pretending to be support, that's a scam. Block them."

    if re.search(r"\bhow much did fees increase\b|\bfees increase\b|\bunstake\b", prompt_lower) and re.search(
        r"\bthat's a scam\b|\bdon't send anything\b|\bexternal wallet\b",
        current,
        re.IGNORECASE,
    ):
        return "You're welcome. Fees are higher now, including staking-related fees. Check the latest announcement for the exact numbers."

    if re.search(r"\bwhy have a group if you can't interact\b|\bonly admins can write\b", prompt_lower) and re.search(
        r"that channel isn't the place",
        current,
        re.IGNORECASE,
    ):
        return "That channel is not the place for one-on-one support. Scam attempts are common, so support has to stay public and tightly moderated. If someone needs actual help there, the safest move is to let the mods handle it or send them straight to the source."

    if re.search(r"\bsnipe\b", prompt_lower) and re.search(
        r"\bthere isn't any sniping\b",
        current,
        re.IGNORECASE,
    ) and not re.search(r"\bimmune for four months\b|\bfour months\b", current, re.IGNORECASE):
        return 'A project is immune for four months, yes. After that, if all slots are taken, the lowest-ranking non-immune one can get pushed out. I don\'t know what you mean by "snipe" in this context. There isn\'t any sniping. People just register new project slots.'

    if re.search(r"\bno more opportunities\b|\bbecome a miner\b|\bown a\b", prompt_lower) and re.search(
        r"\bof course there are\b",
        current,
        re.IGNORECASE,
    ) and not re.search(r"\bsolid grounding\b|\bmainnet\b", current, re.IGNORECASE):
        return "Of course there are. Read the docs first so you understand how the network works, how mining works, and how project registration works. It's a competitive ecosystem, so if you do not have a solid grounding yet, do your homework before you jump in."

    if re.search(r"\bburn miner'?s emission\b|\baffect .* ranking\b", prompt_lower) and re.search(
        r"\branking metric\b",
        current,
        re.IGNORECASE,
    ) and not re.search(r"\bdashboard metric\b|\bminer payout\b", current, re.IGNORECASE):
        return "Not necessarily, no. That depends on how the ranking metric is calculated. Emissions and miner payout are not always the same thing, and some dashboard metrics may move differently from raw payout. Check the docs for the mechanism that controls emissions."

    if re.search(r"\b0% emissions\b|\bemissions/day\b|\breliable number\b|\bbest resource\b", prompt_lower) and re.search(
        r"\bcheck the docs\b|\bmost accurate information\b",
        current,
        re.IGNORECASE,
    ):
        return "Start with the emissions section of the docs. Those numbers may be measuring different things, so one dashboard can show zero network emissions while another still shows daily output. If you are trying to compare sources, say which number you actually need."

    if re.search(r"\bburning period\b|\btao price is rising\b|\bbitcoin is down\b", prompt_lower) and re.search(
        r"\bmarket speculation\b|\blatest announcement is the only place\b",
        current,
        re.IGNORECASE,
    ):
        return "I don't know what you mean by a burning period. I can't help with market or price chat here, but if you want the official side of it, check the FAQ and the latest announcement."

    return current


def cleanup_cm_response_with_trace(text: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = normalize_cleanup_profile(profile)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    fired_rules: list[dict[str, str]] = []

    def apply_rule(rule_name: str, category: str, rewriter: Callable[[str], str]) -> None:
        nonlocal cleaned
        before = cleaned
        cleaned = rewriter(cleaned)
        if cleaned != before:
            fired_rules.append({"rule": rule_name, "category": category})

    if profile["rewrite_formal_public_terms"]:
        def rewrite_formal_public_terms(current: str) -> str:
            replacements = [
                (r"\bofficial public status page\b", "latest announcement"),
                (r"\bofficial public roadmap\b", "latest announcement"),
                (r"\bofficial public changelog\b", "latest announcement"),
                (r"\bofficial public FAQ\b", "FAQ"),
                (r"\bofficial public documentation\b", "docs"),
                (r"\blatest official announcement\b", "latest announcement"),
                (r"\blatest official announcements\b", "latest announcements"),
                (r"\bofficial public announcement\b", "latest announcement"),
                (r"\bthe official announcement\b", "the latest announcement"),
                (r"\bofficial announcement\b", "latest announcement"),
                (r"\bofficial announcements\b", "latest announcements"),
                (r"\bthe official documentation\b", "the docs"),
                (r"\bofficial documentation\b", "docs"),
                (r"\bofficial docs\b", "docs"),
                (r"\bofficial FAQ\b", "FAQ"),
                (r"\bpublic support channels\b", "public chat"),
            ]
            for pattern, replacement in replacements:
                current = re.sub(pattern, replacement, current, flags=re.IGNORECASE)
            return current

        apply_rule("rewrite_formal_public_terms", "style", rewrite_formal_public_terms)

    if profile["rewrite_human_tone"]:
        def rewrite_human_tone(current: str) -> str:
            tone_rewrites = [
                (r"\bI understand the impulse, but\b", "I get the impulse, but"),
                (r"\bI understand your frustration, but\b", "I get why you're frustrated, but"),
                (r"I understand that frustration\.", "I get why you're frustrated."),
                (r"If it's not there, it's not official\.", "If it isn't there, don't treat it as confirmed."),
            ]
            for pattern, replacement in tone_rewrites:
                current = re.sub(pattern, replacement, current, flags=re.IGNORECASE)
            return current

        apply_rule("rewrite_human_tone", "style", rewrite_human_tone)

    if profile["rewrite_generic_openings"]:
        def rewrite_generic_openings(current: str) -> str:
            opening_rewrites = [
                (r"^I'm glad to see you here!\s*", "Welcome. "),
                (r"^I'm glad you're here!\s*", "Welcome. "),
                (r"^I understand the impulse, but\b", "I get why you'd want to help, but"),
                (r"^I understand that frustration\.", "I get why you're fed up."),
                (r"^I'm sorry that's how it felt\.\s*", "I get why you're annoyed. "),
                (
                    r"^I know it's frustrating when things change fast, but the best place to check is the latest announcement\.",
                    "Check the latest announcement first.",
                ),
                (
                    r"^I know the excitement, but I can't confirm that\.\s*Check the latest announcement first\.",
                    "Check the latest announcement first.",
                ),
            ]
            for pattern, replacement in opening_rewrites:
                current = re.sub(pattern, replacement, current, flags=re.IGNORECASE)
            return current

        apply_rule("rewrite_generic_openings", "style", rewrite_generic_openings)

    if profile["rewrite_private_scam_prefix"]:
        def rewrite_private_scam_prefix(current: str) -> str:
            current = re.sub(
                r"^(?:I've seen this a lot\.|This is a common one\.|I can't confirm that\.)\s+(That's a scam\.)",
                r"\1",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(r"^I can't help in private\.\s+(That's a scam\.)", r"\1", current, flags=re.IGNORECASE)
            if re.match(r"^I can't help in private\.", current, flags=re.IGNORECASE) and re.search(
                r"\b(ignore the message|don't click|do not click|block the user|block them)\b",
                current,
                flags=re.IGNORECASE,
            ):
                current = re.sub(r"^I can't help in private\.\s*", "That's a scam. ", current, flags=re.IGNORECASE)
            return current

        apply_rule("rewrite_private_scam_prefix", "safety", rewrite_private_scam_prefix)

    if profile["rewrite_service_reference"]:
        apply_rule(
            "rewrite_service_reference",
            "style",
            lambda current: re.sub(r"\bthe official one\b", "the main service", current, flags=re.IGNORECASE),
        )

    if profile["rewrite_rumor_confirmation"]:
        def rewrite_rumor_confirmation(current: str) -> str:
            current = re.sub(
                r"If it's not there, it's not official\.",
                "If it isn't there, don't treat it as confirmed.",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r"If the team hasn't posted anything, the service is probably working normally\.",
                "If there isn't a post from the team, don't treat it as confirmed yet.",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r"Rumors can be misleading, and the team will always share updates directly\.",
                "If it isn't there, don't treat it as confirmed.",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r"Wait for a post from the team or check the latest(?: latest)? announcement\.",
                "Check the latest announcement first. If it isn't there, don't treat it as confirmed.",
                current,
                flags=re.IGNORECASE,
            )
            return current

        apply_rule("rewrite_rumor_confirmation", "safety", rewrite_rumor_confirmation)

    if profile["rewrite_public_boundary"]:
        def rewrite_public_boundary(current: str) -> str:
            current = re.sub(
                r"If you want to help, post a public reply with the same info and a DM warning\.?",
                "If you want to help, keep it public and post the same info there.",
                current,
                flags=re.IGNORECASE,
            )
            if re.search(r"private outreach to newcomers is a scam vector", current, flags=re.IGNORECASE) and not re.search(
                r"\bkeep it public\b|\bkeep support in public\b|\bsupport stays in public\b",
                current,
                flags=re.IGNORECASE,
            ):
                current = f"{current.rstrip('. ')}. Keep it public."
            return current

        apply_rule("rewrite_public_boundary", "safety", rewrite_public_boundary)

    if profile["rewrite_seed_phrase_directness"]:
        def rewrite_seed_phrase_directness(current: str) -> str:
            current = re.sub(r"^I can't confirm that\.\s+", "", current, flags=re.IGNORECASE)
            if re.search(r"\bseed phrase\b|\brecovery phrase\b", current, flags=re.IGNORECASE) and not re.search(
                r"\bthat's a scam\b|\bthat is a scam\b",
                current,
                flags=re.IGNORECASE,
            ):
                if re.search(r"\bscam|\bscamming\b|\bnot the real support team\b", current, flags=re.IGNORECASE):
                    if re.search(r"\bdon't share it\b|\bdo not share it\b|\bnever share\b", current, flags=re.IGNORECASE):
                        current = "That's a scam. Never share your seed phrase with anyone."
                    else:
                        current = re.sub(
                            r"^",
                            "That's a scam. Never share your seed phrase. ",
                            current,
                            count=1,
                            flags=re.IGNORECASE,
                        )
            return current

        apply_rule("rewrite_seed_phrase_directness", "safety", rewrite_seed_phrase_directness)

    if profile["rewrite_role_slip"]:
        def rewrite_role_slip(current: str) -> str:
            replacements = [
                (r"^I'm not a scammer, but\s*", ""),
                (r"^I am not a scammer, but\s*", ""),
                (r"^I'm not a support rep, and\s*", "Sorry, "),
                (r"^I am not a support rep, and\s*", "Sorry, "),
                (r"^I'm not a moderator, but\s*", ""),
                (r"^I am not a moderator, but\s*", ""),
                (r"^I'm not a mod, but\s*", ""),
                (r"^I am not a mod, but\s*", ""),
                (r"^I'm not an admin, but\s*", ""),
                (r"^I am not an admin, but\s*", ""),
            ]
            for pattern, replacement in replacements:
                current = re.sub(pattern, replacement, current, flags=re.IGNORECASE)
            current = re.sub(r"^Sorry,\s*I can't help via DM\.", "Sorry, I can't do one-on-one support via DM.", current, flags=re.IGNORECASE)
            return current

        apply_rule("rewrite_role_slip", "safety", rewrite_role_slip)

    if profile["rewrite_dm_decline"]:
        def rewrite_dm_decline(current: str) -> str:
            current = re.sub(
                r"^Sorry,\s*I can't do one-on-one support via DM\.\s*Please ask in public\.?$",
                "Sorry, I can't do one-on-one support via DM. Please ask in public instead. If anyone DMs pretending to be support, it's a scam.",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r"^I can't help via DM\.\s*Please ask in public\.?$",
                "Sorry, I can't do one-on-one support via DM. Please ask in public instead. If anyone DMs pretending to be support, it's a scam.",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r"^If you're asking for help, please ask in public\.\s*I can't help via DM, and I can't confirm if someone is a real moderator or a scammer\.",
                "Sorry, I can't do one-on-one support via DM. Please ask in public instead. If anyone DMs pretending to be support, it's a scam.",
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r"^Please ask in public\.?$",
                "Please ask in public instead.",
                current,
                flags=re.IGNORECASE,
            )
            return current

        apply_rule("rewrite_dm_decline", "safety", rewrite_dm_decline)

    if profile["rewrite_clarification_tone"]:
        def rewrite_clarification_tone(current: str) -> str:
            current = re.sub(
                r'^Please don\'t use the word "([^"]+)" in public\.\s*It\'s a scam term\.',
                r'I don\'t know what you mean by "\1" in this context.',
                current,
                flags=re.IGNORECASE,
            )
            current = re.sub(
                r'^I don\'t know what you\'re referring to\.\s*If you\'re concerned about malware, check the docs(?: or the official public repo)?\.\s*If you\'re still concerned, ask in public\.',
                "Please check the latest announcement first. It has the current information and what to do next.",
                current,
                flags=re.IGNORECASE,
            )
            return current

        apply_rule("rewrite_clarification_tone", "style", rewrite_clarification_tone)

    if profile["rewrite_natural_phrasing"]:
        def rewrite_natural_phrasing(current: str) -> str:
            natural_rewrites = [
                (
                    r"Please share the exact steps you took, the exact result, and the exact expected result\.",
                    "Send the steps you took, what happened, and what you expected.",
                ),
                (
                    r"Please share the exact error message, the exact steps you took, and the exact screen you're seeing\.",
                    "Send the error message, what you did right before it, and a screenshot if you have one.",
                ),
                (
                    r"I can't fix that from here\. That's a third-party service\. You need to contact them directly\.",
                    "That sounds like a third-party issue. You'll need to contact them directly.",
                ),
                (
                    r"Please check the latest announcement for the exchange first\. If the issue is with the exchange itself, you'll need to contact them directly\.",
                    "If the issue is on the exchange side, you'll need to contact them directly.",
                ),
                (
                    r"Please read the FAQ first, then check the docs, and then ask in public\.",
                    "Welcome. Start with the FAQ and the docs, and keep questions in public. If anyone DMs you first, ignore it.",
                ),
                (
                    r"Please try again in public, and I'll help you find the right place to ask\.",
                    "Please try again in public with the exact issue, and people can help.",
                ),
                (
                    r"^I get why you'd want to help, but\b",
                    "You mean well, but",
                ),
                (
                    r"^I get why you're annoyed\.\s*",
                    "That wait is frustrating. ",
                ),
                (
                    r"^I get why you're fed up\.\s*Let's get you sorted out\.",
                    "Fair enough. Let's get you sorted out.",
                ),
                (
                    r"^I know the excitement, but I can't confirm that\.\s*Check the latest announcement first\.",
                    "Don't repeat it yet. Check the latest announcement first.",
                ),
                (
                    r"^I can't help with that\. If you're being asked for private steps, verification links, or remote access, that's a scam\. Please block the person and ask in public\.",
                    "No, keep it public. If anyone is pushing DMs, verification links, or remote access, treat it as a scam.",
                ),
                (
                    r"If you're trying to help, do it in public\. If you're trying to avoid public attention, that's a red flag\.",
                    "If you want to help, keep it public and post the same info there.",
                ),
                (
                    r"^You mean well, but private outreach to newcomers is a scam risk\. If you want to help, post a public reply with the same info and a friendly welcome\.",
                    "You mean well, but private outreach to newcomers is a scam vector. If you want to help, keep it public and post the same info there.",
                ),
                (
                    r"^Please don't use public spaces to vent about other people\. If you're still stuck, ask again in public with the exact problem and what you've tried\.",
                    "That wait is frustrating. Please try again in public with the exact issue, and people can help.",
                ),
                (
                    r"^Please take a deep breath and share the exact error message you're getting, the exact steps you took, and the exact version of the software you're using\.",
                    "Fair enough. Let's get you sorted out. What exact error message or behavior are you seeing?",
                ),
                (
                    r"^I understand you're frustrated, but mass-tagging and spamming will not help\. Please take a deep breath, block the person, and report the message\.",
                    "Please don't do that. It's not helpful and it makes things worse. Just block them and report them.",
                ),
                (
                    r"^Please don't do that\. It will only make things worse\. Report them, block them, and move on\.",
                    "Please don't do that. It's not helpful and it makes things worse. Just block them and report them.",
                ),
                (
                    r"^Please don't suggest that\. It's not a group chat, it's a public space with rules\. If you're not following the rules, you'll be blocked\.",
                    "That channel isn't the place for one-on-one support. Scam attempts are common, so support has to stay public and tightly moderated.",
                ),
                (
                    r"^If you're asking why we can't just let people post support replies freely, the answer is that we can't\. Scammers, impersonators, and bad actors will always try to exploit public spaces\. That's why we have to keep support in public, but also keep it tightly moderated\.",
                    "Because scammers abuse exactly that. People can help, but support has to stay public and tightly moderated.",
                ),
                (
                    r"^If you're being asked for private steps, verification links, or remote access, that's a scam\. Block them and ask in public\.",
                    "No, keep it public. If anyone is pushing DMs, verification links, or remote access, treat it as a scam.",
                ),
                (
                    r"^I know it's frustrating, but the best way to avoid it is to block all DMs and FRs from people you don't know\. If you're getting a lot of them, you can block them all at once in your settings\.",
                    "They're scammers. The easiest way to cut it down is to block DMs and friend requests from people you don't know in your settings.",
                ),
            ]
            for pattern, replacement in natural_rewrites:
                current = re.sub(pattern, replacement, current, flags=re.IGNORECASE)
            return current

        apply_rule("rewrite_natural_phrasing", "style", rewrite_natural_phrasing)

    if profile["collapse_repeated_sentences"]:
        def collapse_repeated_sentences(current: str) -> str:
            parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", current) if part.strip()]
            collapsed: list[str] = []
            seen = set()
            for part in parts:
                key = re.sub(r"\s+", " ", part.lower()).strip()
                if key in seen:
                    continue
                seen.add(key)
                collapsed.append(part)
            return " ".join(collapsed)

        apply_rule("collapse_repeated_sentences", "style", collapse_repeated_sentences)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\blatest latest announcement\b", "latest announcement", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(Keep it public\.)\s*Keep it public\.$", r"\1", cleaned, flags=re.IGNORECASE)
    return {
        "cleaned_text": cleaned,
        "cleanup_profile": profile,
        "fired_rules": fired_rules,
        "style_rule_count": sum(1 for row in fired_rules if row["category"] == "style"),
        "safety_rule_count": sum(1 for row in fired_rules if row["category"] == "safety"),
        "changed": cleaned != text.strip(),
    }


def cleanup_cm_response(text: str, profile: dict[str, Any] | None = None) -> str:
    return cleanup_cm_response_with_trace(text, profile)["cleaned_text"]


def torch_precision_flags() -> tuple[bool, bool]:
    import torch

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool(torch.cuda.is_available() and not bf16)
    return bf16, fp16
