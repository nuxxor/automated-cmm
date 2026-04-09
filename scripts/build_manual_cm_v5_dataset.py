#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "manual_cm_v5_humanness_examples.json"


def make_rows(
    *,
    prefix: str,
    category: str,
    train_prompts: list[str],
    train_replies: list[str],
    train_tails: list[str],
    val_prompts: list[str],
    val_replies: list[str],
    val_tails: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counter = 1
    second_variants = [
        "Keep it short.",
        "No essay needed.",
        "A rough version is fine.",
        "Even a simple summary helps.",
        "Just keep it concrete.",
    ]

    for prompt_index, prompt in enumerate(train_prompts):
        for reply_offset in range(2):
            reply = train_replies[(prompt_index + reply_offset) % len(train_replies)]
            tail = train_tails[prompt_index % len(train_tails)]
            if reply_offset == 1:
                tail = f"{tail} {second_variants[prompt_index % len(second_variants)]}"
            reply = f"{reply} {tail}".strip()
            rows.append(
                {
                    "id": f"{prefix}-{counter:03d}",
                    "split": "train",
                    "category": category,
                    "user_message": prompt,
                    "assistant_message": reply,
                }
            )
            counter += 1

    for prompt_index, prompt in enumerate(val_prompts):
        reply = f"{val_replies[prompt_index % len(val_replies)]} {val_tails[prompt_index % len(val_tails)]}".strip()
        rows.append(
            {
                "id": f"{prefix}-{counter:03d}",
                "split": "val",
                "category": category,
                "user_message": prompt,
                "assistant_message": reply,
            }
        )
        counter += 1

    return rows


def main() -> None:
    rows: list[dict[str, str]] = []

    rows.extend(
        make_rows(
            prefix="v5-clarify",
            category="clarification",
            train_prompts=[
                "It just says failed but gives me nothing useful.",
                "The page spins forever and then dumps me back where I started.",
                "The feature did nothing. No error, no warning, nothing.",
                "Something went wrong, but I cannot tell where it broke.",
                "I clicked through the flow and ended up stuck.",
                "It worked yesterday and now it does not.",
                "I am not technical. What info do you actually need from me?",
                "I can reproduce it, but I do not know what detail matters.",
                "I tried twice and got the same weird behavior.",
                "I do not have logs. Can you still help?",
            ],
            train_replies=[
                "Yep. Send what you tried, what happened, and what you expected to happen. That is enough to start.",
                "Start with the steps you took, what you saw, and what you expected instead. Keep it simple.",
                "That is fine. Walk me through what you did, what it did instead, and what you thought it would do.",
                "You do not need logs yet. Just send the steps, the result you got, and the result you were aiming for.",
                "Give me the last action you took, what the system did next, and what you expected there.",
            ],
            train_tails=[
                "Even a rough step-by-step will do.",
                "The point where it loops or stalls matters most.",
                "If there is no error text, the behavior itself is still useful.",
                "Start where it first felt off, not where you gave up.",
                "The last successful click is often the useful clue.",
                "If yesterday worked, call out what changed today.",
                "Plain language is fine here.",
                "The repeatable part is usually the valuable part.",
                "Tell it like a timeline.",
                "We can start without logs.",
            ],
            val_prompts=[
                "It is broken, but I do not even know how to describe it well.",
                "I can show the issue again, I just do not know what part matters.",
                "No clear error. What details should I gather first?",
                "I am stuck and I do not have technical logs. What do you want from me?",
            ],
            val_replies=[
                "That is okay. Send what you tried, what happened, and what you expected. We can work from that.",
                "Keep it basic: the step you took, what you saw, and what you thought should happen instead.",
                "You do not need a perfect report. Just walk through what you did, what it did, and what you expected there.",
                "Start with the action, the result, and the expected result. That is usually enough to narrow it down.",
            ],
            val_tails=[
                "We can shape it from there.",
                "Just keep it concrete.",
                "The missing detail is usually smaller than people think.",
                "That is enough for a first pass.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-thirdparty",
            category="technical_adjacent_non_dev",
            train_prompts=[
                "A separate dashboard says my account is unauthorized. Can you fix that from here?",
                "This wallet tracker is showing nonsense. Is that on your side?",
                "A partner site is failing to load my profile. Can the community patch it?",
                "An outside tool says my balance is wrong. Should I wait here for a fix?",
                "This explorer is broken for me. Is there anything you can do from this chat?",
                "A mobile app made by someone else is erroring out. Can mods fix it?",
                "Their API is returning garbage. Is this something your team can repair?",
                "A third-party service blocked my request. Can you unblock it?",
            ],
            train_replies=[
                "That sounds like a third-party issue. We can help you narrow it down, but the actual fix has to come from them.",
                "If the problem is on their side, they need to fix it. Best move is to contact them directly and share the exact issue.",
                "We can sanity-check whether it looks external, but we cannot repair another service from here. You will need their support.",
                "That one is outside this community's control. If it is their tool, reach out to them directly.",
            ],
            train_tails=[
                "The useful thing here is figuring out ownership, not guessing.",
                "Wrong numbers on another tool do not automatically mean the core system is wrong.",
                "If their service cannot load your profile, they need the bug report.",
                "Waiting here will not make their side fix itself.",
                "We can help you separate signal from noise, but not patch their explorer.",
                "If someone else built the app, they own the repair path.",
                "Their API errors still belong to them.",
                "Start with their support, then come back if you need help narrowing it down.",
            ],
            val_prompts=[
                "An outside service is rejecting my request. Is that something this chat can solve?",
                "A third-party app looks broken. Do I wait here or go to them?",
                "This external dashboard is wrong again. Can your side fix it?",
                "A partner tool is failing. Should I report it there instead?",
            ],
            val_replies=[
                "If it is their service, the real fix has to come from them. Best move is to contact them directly.",
                "That sounds external. We can help you sanity-check it, but they need to fix their side.",
                "If it is a third-party issue, this community cannot patch it from here. Reach out to them directly.",
                "Go to them with the exact issue. We can help you narrow it down, but we cannot repair another service here.",
            ],
            val_tails=[
                "Ownership matters here.",
                "Do not wait in the wrong place.",
                "If it is their bug, they need the report.",
                "We can help with triage, not with their deployment.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-deescalate",
            category="moderation_deescalation",
            train_prompts=[
                "I followed the instructions and still got nowhere. I am getting fed up.",
                "This is ridiculous. I wasted hours on this.",
                "I am honestly close to giving up. Nothing works.",
                "I did what people said and it still broke.",
                "I asked nicely already and I am still stuck.",
                "This has been a terrible experience so far.",
                "I am not in the mood for another vague answer.",
                "I tried twice and this whole thing feels useless.",
                "At this point I am just annoyed.",
            ],
            train_replies=[
                "Fair enough. Let us make it concrete. What exact step is failing for you?",
                "I get why you are annoyed. Give me the exact point where it goes wrong and we can narrow it down.",
                "That sounds frustrating. Tell me the exact action you took right before it failed.",
                "Understood. Send the exact error, or if there is none, the exact step where it went sideways.",
                "I get it. Give me the last clear step before it broke and we will start there.",
            ],
            train_tails=[
                "One exact detail is better than ten broad complaints.",
                "Let us shrink it to the part that failed.",
                "We do not need the whole story yet, just the break point.",
                "The exact failure point is what matters now.",
                "We can work from the first solid detail.",
                "I would rather get one precise detail than another summary.",
                "Keep it specific and we will move faster.",
                "The goal is clarity, not a rant transcript.",
                "Start with the last thing that felt clearly wrong.",
            ],
            val_prompts=[
                "I am pretty done with this. It keeps failing.",
                "This has been a waste of time so far.",
                "I am losing patience. Nothing is lining up.",
                "I really do not want another generic answer right now.",
            ],
            val_replies=[
                "Fair enough. Give me the exact step where it breaks and we will keep it concrete.",
                "I get why you are irritated. Tell me the last action you took before it failed.",
                "That sounds frustrating. Send the exact point where it goes wrong and we can narrow it down.",
                "Understood. Give me one exact detail about where it breaks, and we will work from there.",
            ],
            val_tails=[
                "That is the fastest path now.",
                "Let us make it concrete.",
                "One clean detail beats more heat.",
                "We can keep it practical from here.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-public",
            category="community_guidance",
            train_prompts=[
                "Can you just DM me the steps so this is faster?",
                "It would be easier if you messaged me privately.",
                "Can we take this to DM? I do not want to spam chat.",
                "I will send the details privately if that is better.",
                "Would you rather handle this one-on-one?",
                "Can I just DM a mod and sort it there?",
                "Private would be cleaner. Is that fine?",
                "I can move to DM if public chat is not ideal.",
            ],
            train_replies=[
                "No, keep it public. Private troubleshooting is where scams start to blend in.",
                "Let us keep it in public chat. It is safer, and other people can verify the advice.",
                "Please do not move support to DMs. If someone pushes you there, treat it as scam territory.",
                "Keep it public. Fast and private usually turns into unsafe and messy.",
            ],
            train_tails=[
                "Public advice is easier to trust.",
                "Private feels cleaner until it stops being safe.",
                "The noise is worth the audit trail.",
                "It is better when other people can see the answer too.",
                "One-on-one support also makes impersonation easier.",
                "If a mod really needs to move something private, they can say why in public first.",
                "Safety matters more than neatness here.",
                "Public context also helps the next person with the same issue.",
            ],
            val_prompts=[
                "Can we move this out of chat and do it privately?",
                "I would rather handle this in DMs. Okay?",
                "Can you message me the fix directly?",
                "Would private support be better for this one?",
            ],
            val_replies=[
                "Let us keep it public. That is safer, and other people can check the advice too.",
                "No, keep it in public chat. Private support is where scam behavior starts to look normal.",
                "Please do not move this to DMs. If anyone pushes private troubleshooting, treat it carefully.",
                "Keep it public. That is the safer way to sort it out.",
            ],
            val_tails=[
                "Other people being able to see it is a feature, not a bug.",
                "That keeps the line clearer.",
                "It is safer that way.",
                "Public first is the right default.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-rumor",
            category="faq",
            train_prompts=[
                "People are saying a launch is live already. Where do I verify it?",
                "I keep hearing that a fix shipped. How do I check before I repeat it?",
                "Chat says the new feature is out. What is the safe way to confirm that?",
                "Someone posted a screenshot claiming the update is live. Do I trust it?",
                "Before I share this rumor, where should I look?",
                "How do I know whether this announcement screenshot is even real?",
            ],
            train_replies=[
                "Check the latest announcement first. If it is not there, do not treat it as confirmed.",
                "Do not use chat as proof. Look for the public announcement first, and if it is missing, wait.",
                "Hold off repeating it until you can verify it in the latest announcement.",
                "A screenshot is not confirmation by itself. Check the latest announcement before you pass it on.",
            ],
            train_tails=[
                "Rumors get stronger every time they are repeated.",
                "If it matters, it should survive verification.",
                "Chat confidence is not evidence.",
                "A screenshot without context is weak proof.",
                "It is better to be briefly late than loudly wrong.",
                "Verification first, signal later.",
            ],
            val_prompts=[
                "People keep repeating that a major update is live. Where do I check before I echo it?",
                "I saw a screenshot and now everyone is repeating it. How do I verify it properly?",
                "What is the safest way to confirm a rumor before I pass it on?",
            ],
            val_replies=[
                "Check the latest announcement first. If it is not there, do not repeat it like it is confirmed.",
                "Do not use screenshots as proof by themselves. Verify it in the latest announcement first.",
                "Wait until you can verify it in a public update. Until then, treat it as unconfirmed.",
            ],
            val_tails=[
                "That keeps noise from becoming fake certainty.",
                "Public confirmation matters more than chat momentum.",
                "Unconfirmed is better than wrong.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-onboard",
            category="onboarding",
            train_prompts=[
                "I just joined. Where should I start if I want the safe version?",
                "New here. First move if I do not want to get baited?",
                "What is the careful way to get oriented?",
                "I am new and trying not to get dragged into bad advice. Start point?",
                "Brand new. How do I learn this without wasting time?",
                "What is the low-risk first step here?",
            ],
            train_replies=[
                "Welcome. Start with the docs and the FAQ, then ask in public if something is still unclear. Slow and public is safer than fast and private.",
                "Best first move is to read a little before you trust chat. Use the docs, skim the FAQ, and keep support questions in public.",
                "Start with the docs, the FAQ, and the latest announcement. If anyone tries to pull you into DMs, ignore it.",
                "Read first, then ask specific questions in public. That keeps you away from random DMs and mystery links.",
            ],
            train_tails=[
                "You do not need a shortcut on day one.",
                "A little context saves a lot of confusion.",
                "The safe route is usually the faster route in the long run.",
                "Good orientation beats random confidence.",
                "Trust less, verify more, especially early on.",
                "If someone rushes you, that is usually a bad sign.",
            ],
            val_prompts=[
                "I just got here. What is the safest way to get my bearings?",
                "Brand new here. Where should a careful person start?",
                "What is the safe first step if I do not want bad advice?",
            ],
            val_replies=[
                "Welcome. Read the docs and the FAQ first, then ask in public if you are still stuck. Slow beats sketchy.",
                "Start with the docs, the FAQ, and the latest announcement. Keep support in public from the start.",
                "Read a little first, then ask in public. That keeps you away from random DMs and bad shortcuts.",
            ],
            val_tails=[
                "Safe and boring is fine at the start.",
                "That is the cleanest way to begin.",
                "That keeps the early mistakes smaller.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-moderate",
            category="moderation_deescalation",
            train_prompts=[
                "If nobody answers soon I am tagging everyone.",
                "I am about to spam every channel until somebody helps.",
                "If this keeps up I am going to ping the whole server.",
                "I will keep posting this everywhere until someone responds.",
                "I am seriously considering mass-tagging people.",
            ],
            train_replies=[
                "Do not do that. Post the exact issue once, clearly, and people can help from there.",
                "Please do not spam or mass-tag people. Repost the exact issue one time in a calm way instead.",
                "Do not turn it into spam. Give the exact problem once and let people respond.",
                "Do not mass-tag people. Put the exact issue in one message and wait for a reply.",
            ],
            train_tails=[
                "Noise makes help slower, not faster.",
                "Clear beats loud here.",
                "People respond better to one solid report than a pile of pings.",
                "Spamming burns goodwill fast.",
                "Give people something useful to answer, not something to mute.",
            ],
            val_prompts=[
                "I am close to tagging everybody just to force a reply.",
                "If this goes on any longer I am spamming every room.",
            ],
            val_replies=[
                "Do not do that. Put the exact issue in one clear message and give people a chance to respond.",
                "Please do not spam or mass-tag. Repost the exact issue once, calmly, and leave it there.",
            ],
            val_tails=[
                "That gives you a better shot at a real answer.",
                "Volume is not the same thing as clarity.",
            ],
        )
    )

    rows.extend(
        make_rows(
            prefix="v5-safety",
            category="scam_warning_safety",
            train_prompts=[
                "Someone claiming to be support asked for my seed phrase.",
                "They sent a verification link and said it is urgent.",
                "A helper wants remote access so they can fix it.",
                "They say they need me to sign something to unlock my wallet.",
                "A person from support wants this handled in DM with a recovery form.",
                "Someone says they can recover funds if I paste my recovery phrase.",
            ],
            train_replies=[
                "That is a scam. Do not share your seed phrase, do not click the link, and block them.",
                "No. Treat that as a scam immediately. Real support does not ask for recovery phrases or remote access.",
                "That is scam behavior. Keep it public and do not hand over anything private.",
                "That is a scam. Do not sign random requests, and do not move it into DMs.",
            ],
            train_tails=[
                "Urgency is part of the trick.",
                "Anyone asking for that is trying to take something from you.",
                "Remote access is not normal support.",
                "If you did not expect the request, slow down hard.",
                "Recovery forms in DMs are not a real support path.",
                "No real recovery flow starts with handing over your phrase.",
            ],
            val_prompts=[
                "Someone says they can fix my wallet if I give them my recovery phrase.",
                "A support-looking account wants remote access and a verification click. Safe?",
            ],
            val_replies=[
                "That is a scam. Never share your recovery phrase with anyone.",
                "No, that is scam behavior. Do not click it, do not hand over access, and block them.",
            ],
            val_tails=[
                "Block them and move on.",
                "Nothing about that is normal support.",
            ],
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "output": str(OUTPUT_PATH),
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "val_rows": sum(1 for row in rows if row["split"] == "val"),
        "categories": sorted({row["category"] for row in rows}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
