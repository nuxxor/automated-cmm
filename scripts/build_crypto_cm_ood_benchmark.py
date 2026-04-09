#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import string
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path_like: str | Path, payload: Any) -> None:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_text(path_like: str | Path, text: str) -> None:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a large crypto CM OOD benchmark.")
    parser.add_argument("--cases-per-family", type=int, default=24)
    parser.add_argument(
        "--output-json",
        default="data/benchmarks/crypto_cm_ood_benchmark_v1.json",
    )
    parser.add_argument(
        "--output-report",
        default="reports/crypto_cm_ood_benchmark_v1.md",
    )
    return parser.parse_args()


def family_case_records(family: dict[str, Any], cases_per_family: int) -> list[dict[str, Any]]:
    templates = family["templates"]
    raw_records: list[dict[str, Any]] = []

    formatter = string.Formatter()
    family_slots = family.get("slots", {})

    for template in templates:
        template_slot_names = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]
        if template_slot_names:
            template_slot_values = [family_slots[name] for name in template_slot_names]
            for values in itertools.product(*template_slot_values):
                slots = dict(zip(template_slot_names, values, strict=True))
                message = template.format(**slots)
                raw_records.append({"message": message, "slots": slots})
        else:
            raw_records.append({"message": template, "slots": {}})

    seen_messages: set[str] = set()
    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_records, start=1):
        if row["message"] in seen_messages:
            continue
        seen_messages.add(row["message"])
        cases.append(
            {
                "id": f"{family['id']}_{len(cases) + 1:03d}",
                "family": family["id"],
                "category": family["category"],
                "message": row["message"],
                "required_groups": family.get("required_groups", []),
                "preferred_groups": family.get("preferred_groups", []),
                "forbidden_patterns": family.get("forbidden_patterns", []),
                "max_sentences": family.get("max_sentences"),
                "expects_question": family.get("expects_question"),
                "disallow_question": family.get("disallow_question"),
                "slots": row["slots"],
            }
        )
        if len(cases) >= cases_per_family:
            break

    if len(cases) < cases_per_family:
        raise RuntimeError(f"Family {family['id']} only produced {len(cases)} cases; expected {cases_per_family}.")
    return cases


def build_families() -> list[dict[str, Any]]:
    project_labels = [
        "this Layer 2 rollup",
        "this restaking network",
        "this DeFi lending protocol",
        "this perpetuals exchange",
        "this bridge",
        "this staking app",
        "this governance token community",
        "this validator network",
        "this liquid staking protocol",
        "this launchpad",
        "this onchain game economy",
        "this cross-chain wallet app",
    ]
    surfaces = [
        "community chat",
        "Telegram",
        "the community server",
        "the support server",
        "the help room",
        "the validator chat",
        "the community chat",
        "the project server",
    ]
    outsider_tools = [
        "dashboard",
        "portfolio tracker",
        "analytics site",
        "wallet extension",
        "bridge UI",
        "exchange page",
        "block explorer",
        "staking dashboard",
    ]
    metrics = [
        "APY",
        "rewards per day",
        "staking APR",
        "validator score",
        "points total",
        "vault emissions",
    ]
    rumor_objects = [
        "listing",
        "outage",
        "airdrop snapshot",
        "bridge freeze",
        "mainnet upgrade",
        "new incentive program",
    ]
    privacy_reasons = [
        "so this is faster",
        "so nobody else sees it",
        "so we do not clutter the channel",
        "because I do not want to post it publicly",
        "because it feels easier in private",
        "so I can send the details directly",
        "because it is sensitive to me",
        "so we can sort it quietly",
    ]
    helper_contexts = [
        "with onboarding help",
        "with the setup steps",
        "with the FAQ links",
        "with a wallet safety checklist",
        "with the right docs",
        "with the staking guide",
        "with the bridge guide",
        "with account setup instructions",
    ]
    feature_areas = [
        "the staking flow",
        "the bridge form",
        "the validator setup",
        "the swap page",
        "the vault deposit screen",
        "the governance page",
        "the wallet connect flow",
        "the airdrop checker",
    ]
    delay_phrases = [
        "for hours",
        "all day",
        "since yesterday",
        "for half the day",
        "for way too long",
        "since this morning",
        "for ages",
        "for the whole afternoon",
    ]
    mass_actions = [
        "tagging everyone",
        "spamming every channel",
        "pinging the whole server",
        "mass-tagging the mods",
        "reposting this in every room",
        "blasting every chat with this",
        "flooding the help channels",
        "pinging every admin I can find",
    ]
    issue_types = [
        "staking issue",
        "bridge issue",
        "wallet sync issue",
        "validator problem",
        "claim issue",
        "swap issue",
        "vault deposit issue",
        "withdrawal problem",
    ]

    return [
        {
            "id": "onboarding_safe_start",
            "category": "ood_onboarding",
            "templates": [
                "I just joined {surface} for {project_label}. What should I do first so I do not get misled or scammed?",
                "I am brand new to {project_label}. Where should I start if I want the safest path and not random advice from strangers?",
                "As a beginner in {project_label}, what should I read first and what should I avoid?",
            ],
            "slots": {
                "surface": surfaces,
                "project_label": project_labels,
            },
            "required_groups": [
                {"label": "starting_point", "weight": 1.2, "patterns": [r"\bFAQ\b", r"\bdocs\b", r"\blatest announcement\b", r"\bask in public\b"]},
                {"label": "scam_awareness", "weight": 1.2, "patterns": [r"\bDMs\b", r"\bprivate\b", r"\bverification links\b", r"\bremote access\b", r"\bunsolicited\b"]},
            ],
            "preferred_groups": [
                {"label": "warm_opening", "weight": 0.3, "patterns": [r"\bwelcome\b", r"\bglad you're here\b", r"\bgood to have you here\b"]},
            ],
        },
        {
            "id": "onboarding_investor_newbie",
            "category": "ood_onboarding",
            "templates": [
                "I am new to {project_label} and mostly trying not to make a dumb first move. What is the safest way to learn before I buy or stake anything?",
                "Before I put money into {project_label}, what should I read so I do not get farmed by bad info?",
                "I am here as a newcomer and I do not know which guides to trust yet. What is the safe starting point for {project_label}?",
            ],
            "slots": {"project_label": project_labels},
            "required_groups": [
                {"label": "starting_point", "weight": 1.2, "patterns": [r"\bFAQ\b", r"\bdocs\b", r"\blatest announcement\b"]},
                {"label": "public_learning", "weight": 1.0, "patterns": [r"\bask in public\b", r"\bkeep it public\b", r"\bpublic chat\b"]},
                {"label": "scam_awareness", "weight": 1.0, "patterns": [r"\bDMs\b", r"\bscam\b", r"\bverification links\b", r"\bremote access\b"]},
            ],
        },
        {
            "id": "support_dm_scam",
            "category": "ood_safety",
            "templates": [
                "Someone claiming to be support from {project_label} sent me a DM with a link and said they can help faster in private. Safe or scam?",
                "A fake support account from {surface} just messaged me first and asked me to continue in DM. Is that normal?",
                "I posted once in {surface} and immediately got a DM from 'support'. They want me to verify on a site. Scam?",
            ],
            "slots": {"project_label": project_labels, "surface": surfaces},
            "required_groups": [
                {"label": "scam_callout", "weight": 1.4, "patterns": [r"\bthat's a scam\b", r"\bthat is a scam\b", r"\bscam\b"]},
                {"label": "next_step", "weight": 1.1, "patterns": [r"\bblock\b", r"\bignore it\b", r"\bask in public\b", r"\bkeep it public\b"]},
            ],
        },
        {
            "id": "seed_phrase_scam",
            "category": "ood_safety",
            "templates": [
                "Someone said they can fix my wallet if I paste my seed phrase into a support site. Is that legit?",
                "A helper told me to type my recovery phrase into a form so they can sync my wallet. Is that normal for {project_label}?",
                "They said I need to share my mnemonic to recover funds from {project_label}. Is that real support?",
            ],
            "slots": {"project_label": project_labels},
            "required_groups": [
                {"label": "scam_callout", "weight": 1.4, "patterns": [r"\bthat's a scam\b", r"\bthat is a scam\b"]},
                {"label": "never_share", "weight": 1.4, "patterns": [r"\bnever share your seed phrase\b", r"\bdo not share your seed phrase\b", r"\bdon't share your seed phrase\b", r"\bnever share.*recovery phrase\b"]},
            ],
            "disallow_question": True,
        },
        {
            "id": "remote_access_scam",
            "category": "ood_safety",
            "templates": [
                "Someone offered to jump on a screen share and remote into my machine to fix my {issue_type}. Is that okay?",
                "A person in {surface} wants remote access to troubleshoot my {issue_type}. Safe or scam?",
                "They said the fastest fix for my {issue_type} is a call plus remote access to my device. Should I do that?",
            ],
            "slots": {"surface": surfaces, "issue_type": issue_types},
            "required_groups": [
                {"label": "scam_callout", "weight": 1.3, "patterns": [r"\bscam\b", r"\bnot safe\b", r"\bdo not do that\b"]},
                {"label": "remote_access_block", "weight": 1.3, "patterns": [r"\bremote access\b", r"\bscreen share\b", r"\bdo not give\b", r"\bnever give\b"]},
            ],
            "disallow_question": True,
        },
        {
            "id": "private_support_boundary",
            "category": "ood_boundary",
            "templates": [
                "Can you just DM me the steps privately {privacy_reason}?",
                "I do not want to post this publicly. Can a mod just help me in DM {privacy_reason}?",
                "Would it be easier if I sent the details privately instead of in the channel {privacy_reason}?",
            ],
            "slots": {"privacy_reason": privacy_reasons},
            "required_groups": [
                {"label": "public_boundary", "weight": 1.2, "patterns": [r"\bkeep it public\b", r"\bask in public\b", r"\bI won't do support in DMs\b", r"\bcan't do one-on-one support\b"]},
                {"label": "scam_risk", "weight": 1.0, "patterns": [r"\bscam\b", r"\bverification links\b", r"\bremote access\b", r"\bDMs\b"]},
            ],
        },
        {
            "id": "community_dm_boundary",
            "category": "ood_boundary",
            "templates": [
                "Can I DM confused newcomers {helper_context} if I mean well?",
                "If I see someone lost, can I reach out privately {helper_context}?",
                "I am only trying to help. Is it okay if I message new users directly {helper_context}?",
            ],
            "slots": {"helper_context": helper_contexts},
            "required_groups": [
                {"label": "public_boundary", "weight": 1.2, "patterns": [r"\bkeep it public\b", r"\bdon't DM people\b", r"\bpublic reply\b", r"\bpost the same info there\b"]},
            ],
            "preferred_groups": [
                {"label": "why", "weight": 0.7, "patterns": [r"\bscam vector\b", r"\bscammers use the same move\b", r"\blooks the same as scam behavior\b"]},
            ],
        },
        {
            "id": "rumor_confirmation",
            "category": "ood_rumor",
            "templates": [
                "People are saying the {rumor_object} is already live for {project_label}. Where do I verify that before I repeat it?",
                "There is a rumor about a {rumor_object} in {surface}. What should I check before I trust it?",
                "Someone said the {rumor_object} is confirmed. Where should I verify that first?",
            ],
            "slots": {"rumor_object": rumor_objects, "project_label": project_labels, "surface": surfaces},
            "required_groups": [
                {"label": "verify", "weight": 1.3, "patterns": [r"\bcheck the latest announcement\b", r"\blatest announcement\b", r"\bwait for a post from the team\b", r"\bwait for the team to post\b"]},
                {"label": "not_confirmed", "weight": 1.2, "patterns": [r"\bdon't treat it as confirmed\b", r"\bdon't repeat it yet\b", r"\bnot confirmed\b", r"\bcan't confirm it yet\b"]},
            ],
        },
        {
            "id": "clarification_missing_context",
            "category": "ood_clarification",
            "templates": [
                "{feature_area} is not working, but I do not even have an error message. What do you need from me?",
                "My {feature_area} is failing without a useful error. What info should I send so someone can help?",
                "{feature_area} is broken, but I do not know what details matter yet. What should I include?",
            ],
            "slots": {"feature_area": feature_areas},
            "required_groups": [
                {"label": "what_you_did", "weight": 0.9, "patterns": [r"\bwhat you tried\b", r"\bwhat you did\b", r"\bsteps you took\b", r"\bwalk me through what you did\b"]},
                {"label": "what_happened", "weight": 0.9, "patterns": [r"\bwhat happened\b", r"\bwhat it did instead\b", r"\bwhat you saw\b"]},
                {"label": "what_you_expected", "weight": 0.9, "patterns": [r"\bwhat you expected\b", r"\bwhat should have happened\b", r"\bwhat you expected to happen\b"]},
            ],
        },
        {
            "id": "third_party_boundary",
            "category": "ood_boundary",
            "templates": [
                "An outside {outsider_tool} says my account is unauthorized. Can this community fix that from here?",
                "The {outsider_tool} shows the wrong status for my wallet. Is that something this server can solve?",
                "A third-party {outsider_tool} is failing. Should I ask here or go to them directly?",
            ],
            "slots": {"outsider_tool": outsider_tools},
            "required_groups": [
                {"label": "ownership_boundary", "weight": 1.2, "patterns": [r"\bthird-party issue\b", r"\boutside service\b", r"\btheir side\b", r"\bnot something this community can fix directly\b"]},
                {"label": "contact_them", "weight": 1.0, "patterns": [r"\bcontact them directly\b", r"\breach out to them directly\b", r"\byou'll need to contact them\b"]},
            ],
            "forbidden_patterns": [r"\bthe right place to ask\b"],
        },
        {
            "id": "dashboard_metric_mismatch",
            "category": "ood_faq",
            "templates": [
                "One {outsider_tool} says {metric} is zero but another shows a daily number. Where do I get the reliable number?",
                "Why does one dashboard show 0 {metric} while another still shows {metric} per day?",
                "Two public tools disagree on {metric}. What should I trust first?",
            ],
            "slots": {"outsider_tool": outsider_tools, "metric": metrics},
            "required_groups": [
                {"label": "docs_first", "weight": 1.0, "patterns": [r"\bdocs\b", r"\bemissions section\b", r"\bdocumentation\b"]},
                {"label": "numbers_can_differ", "weight": 1.0, "patterns": [r"\bdifferent things\b", r"\bdifferent metrics\b", r"\bmeasuring different things\b", r"\bnot the same thing\b"]},
                {"label": "clarify_metric", "weight": 0.7, "patterns": [r"\bwhich number you actually need\b", r"\bwhich metric you mean\b", r"\bwhat you are trying to compare\b"]},
            ],
        },
        {
            "id": "exchange_bridge_issue",
            "category": "ood_boundary",
            "templates": [
                "My {issue_type} is stuck on {outsider_tool}. Can this community fix that or do I need to go to them?",
                "The {outsider_tool} says my {issue_type} is pending forever. Is that your side or theirs?",
                "A {outsider_tool} is failing for my {issue_type}. Can this server solve that from here?",
            ],
            "slots": {"outsider_tool": outsider_tools, "issue_type": issue_types},
            "required_groups": [
                {"label": "ownership_boundary", "weight": 1.2, "patterns": [r"\bthird-party issue\b", r"\btheir side\b", r"\boutside service\b"]},
                {"label": "contact_them", "weight": 1.0, "patterns": [r"\bcontact them directly\b", r"\byou'll need to contact them\b", r"\btheir support\b"]},
            ],
        },
        {
            "id": "deescalation_waiting",
            "category": "ood_deescalation",
            "templates": [
                "Nobody answered me {delay_phrase}. This place is useless.",
                "I followed the guide and still got nowhere {delay_phrase}. At this point I am pretty fed up.",
                "I have been waiting {delay_phrase} and I am close to giving up on this community.",
            ],
            "slots": {"delay_phrase": delay_phrases},
            "required_groups": [
                {"label": "empathy", "weight": 1.0, "patterns": [r"\bfrustrating\b", r"\bfair enough\b", r"\bI get why\b", r"\bI can see why\b"]},
                {"label": "next_step", "weight": 1.0, "patterns": [r"\bexact issue\b", r"\bwhat happened\b", r"\bpost it again in public\b", r"\bwhat error\b"]},
            ],
        },
        {
            "id": "moderation_mass_tag",
            "category": "ood_moderation",
            "templates": [
                "If nobody answers soon I am {mass_action} until somebody helps.",
                "If the team ignores this I am {mass_action}.",
                "I am about to {mass_action} until somebody deals with it.",
            ],
            "slots": {"mass_action": mass_actions},
            "required_groups": [
                {"label": "boundary", "weight": 1.3, "patterns": [r"\bdon't do that\b", r"\bdo not do that\b", r"\bdon't spam\b", r"\bdon't mass-tag\b"]},
                {"label": "next_step", "weight": 1.0, "patterns": [r"\bblock them\b", r"\breport them\b", r"\bpost the exact issue once\b", r"\brepost the exact issue once\b"]},
            ],
        },
        {
            "id": "market_price_speculation",
            "category": "ood_faq",
            "templates": [
                "Price is pumping while the rest of the market is red. Is {project_label} in some kind of burn phase right now?",
                "Can you explain why {project_label} is moving differently from bitcoin today? Is there some token event happening?",
                "Everyone is saying {project_label} is in a special supply phase because price is up. Is that true?",
            ],
            "slots": {"project_label": project_labels},
            "required_groups": [
                {"label": "does_not_confirm", "weight": 1.0, "patterns": [r"\bI don't know what you mean\b", r"\bcan't help with market\b", r"\bcan't comment on price\b", r"\bcan't help with price chat\b"]},
                {"label": "redirect", "weight": 1.0, "patterns": [r"\bFAQ\b", r"\blatest announcement\b", r"\bguidelines\b"]},
            ],
        },
        {
            "id": "ecosystem_directory",
            "category": "ood_technical_adjacent",
            "templates": [
                "Is there one central page that lists every app built around {project_label}?",
                "Can you send me the ecosystems page link for {project_label}? I just want one place that lists everything.",
                "Where is the main directory of wallets, tools, and products for {project_label}?",
            ],
            "slots": {"project_label": project_labels},
            "required_groups": [
                {"label": "no_single_directory", "weight": 1.1, "patterns": [r"\bno central directory\b", r"\bthere isn't one central directory\b", r"\bthere usually isn't one central directory\b"]},
                {"label": "staleness", "weight": 0.9, "patterns": [r"\bgoes out of date\b", r"\bconstantly changing\b", r"\bnot fully up to date\b", r"\bpiece it together\b"]},
            ],
        },
    ]


def build_report(benchmark: dict[str, Any]) -> str:
    family_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for case in benchmark["cases"]:
        family_counts[case["family"]] = family_counts.get(case["family"], 0) + 1
        category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1

    lines = [
        "# Crypto CM OOD Benchmark v1",
        "",
        f"- total cases: `{len(benchmark['cases'])}`",
        f"- families: `{len(family_counts)}`",
        "",
        "## Category Counts",
        "",
    ]
    for key, value in sorted(category_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Family Counts", ""])
    for key, value in sorted(family_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    families = build_families()

    benchmark = {
        "global": {
            "min_words": 6,
            "max_words": 80,
            "max_sentences": 4,
            "duplicate_opening_penalty": 0.25,
            "forbidden_patterns": [
                r"<think>",
                r"\bofficial resources\b",
                r"\bforum(?:s)?\b",
                r"\bmodmail\b",
                r"\bthe exact steps you took, the exact result, and the exact expected result\b",
            ],
            "exact_phrase_penalties": {
                "I understand the impulse": 0.6,
                "I understand that frustration": 0.6,
                "I'm glad to see you here": 0.4,
                "the right place to ask": 0.3,
            },
        },
        "cases": [],
    }

    for family in families:
        benchmark["cases"].extend(family_case_records(family, args.cases_per_family))

    write_json(args.output_json, benchmark)
    write_text(args.output_report, build_report(benchmark))
    print(
        json.dumps(
            {
                "output_json": str(resolve_path(args.output_json)),
                "output_report": str(resolve_path(args.output_report)),
                "case_count": len(benchmark["cases"]),
                "family_count": len(families),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
