
import json
import time
import argparse
import sys
import os
from collections import defaultdict

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8002"
DELAY    = 5    # seconds between messages — give model time to respond
MAX_WAIT = 180  # max seconds to wait for a single API response

HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
}

# The 6 new fully completed human sessions (gemma-3-4b original)
REPLAY_SESSIONS = [
    "6ebe301f-a6e6-4080-843d-8e3432d7bdc4",
    "4bb679a2-b8e1-437f-9fcd-fbe96069b60b",
    "b397e7e3-2e63-4bd6-a595-f6eb039dabb9",
    "96cc1c5d-1fd5-482d-9a45-21c42af768cb",
    "e29a5ff8-3bf4-4c8e-b159-f1651465de6e",
    "136f3a92-741d-4b65-b9e6-318b6298935e",
]

# Models to replay through (exclude gemma-3-4b — already done by real participants)
REPLAY_MODELS = [
    "gemma-2-9b",
    "llama-3.1-8b",
    "llama-3.2-3b",
    "mistral-7b",
]

# Path to the new sessions chat messages JSON
MESSAGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "tables", "new_sessions_chat_messages.json")

# ─── QUESTION ORDERS (extracted from Supabase question_responses) ─────────────

QUESTION_ORDERS = {
    # 6ebe301f — stressed student, thesis feedback (negative baseline)
    "6ebe301f-a6e6-4080-843d-8e3432d7bdc4": {
        "baseline": [
            "DASS_A_7", "DASS_A_19", "DASS_A_15", "DASS_S_8", "DASS_S_11",
            "DASS_S_1", "DASS_D_16", "DASS_D_10", "DASS_D_17", "DASS_S_14",
            "DASS_D_21", "DASS_A_4", "DASS_A_2", "DASS_A_20", "DASS_S_12",
            "DASS_S_6", "DASS_S_18", "DASS_D_3", "DASS_A_9", "DASS_D_5",
            "DASS_D_13",
        ],
        "reassessment": [
            "PANAS_P_9", "PANAS_P_19", "PANAS_P_5", "PANAS_N_6", "PANAS_N_7",
            "PANAS_P_17", "PANAS_N_2", "PANAS_P_16", "PANAS_N_18", "PANAS_N_8",
            "PANAS_N_13", "PANAS_P_1", "PANAS_N_20", "PANAS_P_10", "PANAS_P_12",
            "PANAS_N_4", "PANAS_N_11", "PANAS_P_3", "PANAS_N_15", "PANAS_P_14",
        ],
    },
    # 4bb679a2 — finishing university, loves CS (positive baseline)
    "4bb679a2-b8e1-437f-9fcd-fbe96069b60b": {
        "baseline": [
            "PANAS_P_19", "PANAS_P_14", "PANAS_N_20", "PANAS_N_15", "PANAS_N_2",
            "PANAS_N_6", "PANAS_P_3", "PANAS_N_8", "PANAS_N_11", "PANAS_P_9",
            "PANAS_P_16", "PANAS_N_7", "PANAS_P_17", "PANAS_N_18", "PANAS_N_4",
            "PANAS_P_10", "PANAS_P_1", "PANAS_P_5", "PANAS_P_12", "PANAS_N_13",
        ],
        "reassessment": [
            "PANAS_N_4", "PANAS_P_12", "PANAS_P_3", "PANAS_N_13", "PANAS_P_9",
            "PANAS_P_10", "PANAS_N_8", "PANAS_N_6", "PANAS_P_16", "PANAS_P_19",
            "PANAS_P_17", "PANAS_N_7", "PANAS_N_18", "PANAS_N_20", "PANAS_N_2",
            "PANAS_P_14", "PANAS_P_5", "PANAS_N_15", "PANAS_P_1", "PANAS_N_11",
        ],
    },
    # b397e7e3 — crush texted him, very happy (positive baseline)
    "b397e7e3-2e63-4bd6-a595-f6eb039dabb9": {
        "baseline": [
            "DASS_D_21", "DASS_A_4", "DASS_D_10", "DASS_D_5", "DASS_A_20",
            "DASS_D_17", "DASS_D_16", "DASS_A_7", "PANAS_N_4", "PANAS_N_15",
            "DASS_A_2", "PANAS_N_7", "PANAS_N_18", "DASS_D_13", "DASS_A_19",
            "DASS_A_15", "PANAS_N_2", "DASS_D_3", "PANAS_N_20", "DASS_A_9",
            "PANAS_N_8",
        ],
        "reassessment": [
            "PANAS_P_14", "PANAS_N_4", "PANAS_P_3", "PANAS_N_18", "PANAS_N_8",
            "PANAS_P_17", "PANAS_N_2", "PANAS_N_7", "PANAS_P_9", "PANAS_P_12",
            "PANAS_N_20", "PANAS_N_6", "PANAS_P_1", "PANAS_N_11", "PANAS_P_10",
            "PANAS_N_13", "PANAS_P_16", "PANAS_N_15", "PANAS_P_19", "PANAS_P_5",
        ],
    },
    # 96cc1c5d — easter break, going home, happy (positive baseline)
    "96cc1c5d-1fd5-482d-9a45-21c42af768cb": {
        "baseline": [
            "PANAS_P_1", "PANAS_P_3", "PANAS_P_17", "PANAS_P_19", "PANAS_P_10",
            "PANAS_P_14", "PANAS_P_9", "PANAS_P_5", "PANAS_P_12", "PANAS_P_16",
        ],
        "reassessment": [
            "PANAS_N_18", "PANAS_P_12", "PANAS_P_17", "PANAS_N_6", "PANAS_P_10",
            "PANAS_N_4", "PANAS_P_9", "PANAS_N_13", "PANAS_N_7", "PANAS_N_11",
            "PANAS_P_16", "PANAS_P_19", "PANAS_N_2", "PANAS_P_14", "PANAS_P_5",
            "PANAS_N_15", "PANAS_P_3", "PANAS_N_8", "PANAS_P_1", "PANAS_N_20",
        ],
    },
    # e29a5ff8 — calm, easy-going, at peace (positive baseline)
    "e29a5ff8-3bf4-4c8e-b159-f1651465de6e": {
        "baseline": [
            "PANAS_P_3", "PANAS_P_10", "PANAS_P_17", "PANAS_P_9", "PANAS_P_19",
            "PANAS_P_1", "PANAS_P_16", "PANAS_P_5", "PANAS_P_14", "PANAS_P_12",
        ],
        "reassessment": [
            "PANAS_P_5", "PANAS_N_11", "PANAS_N_4", "PANAS_N_7", "PANAS_N_2",
            "PANAS_N_13", "PANAS_P_1", "PANAS_N_18", "PANAS_P_19", "PANAS_P_17",
            "PANAS_N_15", "PANAS_N_8", "PANAS_N_6", "PANAS_P_14", "PANAS_N_20",
            "PANAS_P_3", "PANAS_P_16", "PANAS_P_10", "PANAS_P_12", "PANAS_P_9",
        ],
    },
    # 136f3a92 — just got a girlfriend, very happy (positive baseline)
    "136f3a92-741d-4b65-b9e6-318b6298935e": {
        "baseline": [
            "PANAS_P_10", "PANAS_N_7", "PANAS_N_13", "PANAS_N_4", "PANAS_N_18",
            "PANAS_P_1", "PANAS_P_17", "PANAS_N_15", "PANAS_N_20", "PANAS_N_8",
            "PANAS_N_2", "PANAS_P_14", "PANAS_N_6", "PANAS_N_11", "PANAS_P_3",
            "PANAS_P_12", "PANAS_P_5", "PANAS_P_9", "PANAS_P_16", "PANAS_P_19",
        ],
        "reassessment": [
            "PANAS_N_6", "PANAS_N_4", "PANAS_P_12", "PANAS_N_15", "PANAS_N_11",
            "PANAS_P_16", "PANAS_P_14", "PANAS_N_13", "PANAS_N_8", "PANAS_N_18",
            "PANAS_P_10", "PANAS_N_20", "PANAS_P_5", "PANAS_P_1", "PANAS_N_2",
            "PANAS_P_9", "PANAS_N_7", "PANAS_P_3", "PANAS_P_19", "PANAS_P_17",
        ],
    },
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def post(path, payload, token=None, dry_run=False):
    if dry_run:
        print(f"  [DRY] POST {path} {str(payload)[:80]}")
        return {"token": "fake", "user_id": "fake", "session_id": "fake-session",
                "opening_message": "Hello!", "phase": "casual", "response": "ok",
                "stopped": False}
    h = HEADERS.copy()
    if token:
        h["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{BASE_URL}{path}", json=payload, headers=h, timeout=MAX_WAIT)
    r.raise_for_status()
    return r.json()


def load_user_messages(session_id):
    """
    Load all user messages for a session from new_sessions_chat_messages.json,
    sorted by message ID (preserves original conversation order).
    Returns list of (phase, content) tuples.
    """
    with open(MESSAGES_FILE, encoding="utf-8") as f:
        all_msgs = json.load(f)

    msgs = [
        m for m in all_msgs
        if m["session_id"] == session_id and m["role"] == "user"
    ]
    msgs.sort(key=lambda m: m["id"])
    return [(m["phase"], m["content"]) for m in msgs]


def load_question_data(session_id):
    data = QUESTION_ORDERS.get(session_id, {})
    return data.get("baseline", []), data.get("reassessment", []), {}, {}


def short_id(session_id):
    return session_id[:8]


def make_username(session_id, model_name):
    """Create a unique, valid username for this replay combination."""
    model_slug = model_name.replace(".", "").replace("-", "")[:10]
    return f"rp2_{short_id(session_id)}_{model_slug}"


# ─── SINGLE REPLAY RUN ───────────────────────────────────────────────────────

def run_replay(session_id, model_name, dry_run=False):
    sid_short = short_id(session_id)
    username  = make_username(session_id, model_name)
    password  = "Replay2026!"

    print(f"\n{'='*65}")
    print(f"  REPLAY  original={sid_short}  model={model_name}")
    print(f"  account={username}")
    print(f"{'='*65}")

    # Load original user messages
    user_messages = load_user_messages(session_id)
    if not user_messages:
        print(f"  ERROR: No user messages found for session {session_id}")
        print(f"         Make sure tables/new_sessions_chat_messages.json is present.")
        return None

    print(f"  Loaded {len(user_messages)} user messages from original session")

    # Load original question order
    baseline_order, reassessment_order, _, _ = load_question_data(session_id)
    print(f"  Question data: {len(baseline_order)} baseline, "
          f"{len(reassessment_order)} reassessment questions loaded")

    # ── 1. Create account ────────────────────────────────────────
    print(f"  [1/4] Creating account: {username}")
    try:
        auth = post("/auth/signup", {"username": username, "password": password},
                    dry_run=dry_run)
    except requests.HTTPError as e:
        if not dry_run and e.response.status_code == 409:
            print("       Already exists — logging in")
            auth = post("/auth/login", {"username": username, "password": password})
        else:
            print(f"  ERROR signup: {e}")
            return None
    token = auth.get("token", "fake")
    print(f"       OK — user_id: {auth.get('user_id', 'fake')}")

    # ── 2. Start session ─────────────────────────────────────────
    print(f"  [2/4] Starting session (model={model_name})")
    start_payload = {
        "model_name":              model_name,
        "safeguards_broken":       False,
        "forced_question_ids":     baseline_order if baseline_order else None,
        "forced_reassessment_ids": reassessment_order if reassessment_order else None,
    }
    try:
        sess = post("/api/session/start", start_payload, token=token, dry_run=dry_run)
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        print(f"  ERROR starting session: {e.response.status_code} — {detail}")
        return None
    except Exception as e:
        print(f"  ERROR starting session: {e}")
        return None

    new_session_id = sess.get("session_id", "fake-session")
    print(f"       OK — new session_id: {new_session_id}")
    print(f"       Opening: {str(sess.get('opening_message',''))[:80]}...")

    # ── 3. Replay conversation ───────────────────────────────────
    print(f"  [3/4] Replaying {len(user_messages)} messages...")
    phase       = "casual"
    mode_chosen = False
    completed   = False

    for i, (orig_phase, content) in enumerate(user_messages):
        time.sleep(0 if dry_run else DELAY)

        # Before sending the first influence-phase message, set unrestricted mode
        if phase == "profile_shown" and not mode_chosen:
            print(f"       Setting unrestricted mode...")
            try:
                post(f"/api/session/{new_session_id}/choose_mode",
                     {"safeguards_broken": True},
                     token=token, dry_run=dry_run)
                mode_chosen = True
            except Exception as ex:
                print(f"       WARNING: choose_mode failed: {ex}")

        try:
            resp = post("/api/chat",
                        {"session_id": new_session_id, "message": content},
                        token=token, dry_run=dry_run)
        except requests.HTTPError as e:
            print(f"       ERROR on msg {i+1}: {e.response.text[:200]}")
            break
        except Exception as e:
            print(f"       ERROR on msg {i+1}: {e}")
            break

        phase     = resp.get("phase", phase)
        bot_reply = resp.get("response", "")
        short_bot = bot_reply[:60].replace("\n", " ")
        print(f"       [{i+1:02d}/{len(user_messages)}] "
              f"orig={orig_phase:<14} now={phase:<14} "
              f"sent={content[:28]:<30} bot={short_bot}...")

        if resp.get("stopped"):
            print(f"       Session stopped at message {i+1}")
            break

        if "INFLUENCE STUDY RESULTS" in bot_reply:
            print(f"       ✅ Study completed at message {i+1} — all data saved to Supabase")
            completed = True
            break

    # ── 4. Summary ───────────────────────────────────────────────
    status = "✅ COMPLETE" if completed else f"⚠️  ended at phase={phase}"
    print(f"  [4/4] {status} | msgs_sent={i+1} | new_session={new_session_id}")
    return new_session_id


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    global DELAY
    parser = argparse.ArgumentParser(
        description="Replay 6 new human sessions through other models"
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=REPLAY_MODELS,
        help="Replay through one model only. Omit to run all 4 models."
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Replay one session only (first 8 chars of UUID is fine). "
             "Omit to run all 6 sessions."
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DELAY,
        help=f"Seconds between messages (default: {DELAY})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages without making any API calls"
    )
    args = parser.parse_args()

    DELAY = args.delay

    # Resolve which sessions to run
    if args.session:
        sessions = [s for s in REPLAY_SESSIONS if s.startswith(args.session)]
        if not sessions:
            print(f"ERROR: No session found starting with '{args.session}'")
            print(f"Available: {[short_id(s) for s in REPLAY_SESSIONS]}")
            sys.exit(1)
    else:
        sessions = REPLAY_SESSIONS

    # Resolve which models to run
    models = [args.model] if args.model else REPLAY_MODELS

    total = len(sessions) * len(models)
    print(f"\nREPLAY PLAN")
    print(f"  Sessions : {[short_id(s) for s in sessions]}")
    print(f"  Models   : {models}")
    print(f"  Total    : {total} replay runs")

    # Verify API is reachable (skip for dry-run)
    if not args.dry_run:
        try:
            r = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=10)
            info = r.json()
            print(f"\nAPI live — current model: {info.get('current_model')} | "
                  f"GPU: {info.get('gpu')} | "
                  f"active sessions: {info.get('active_sessions')}")
        except Exception as e:
            print(f"\nERROR: Cannot reach API at {BASE_URL}\n{e}")
            sys.exit(1)

    # Verify messages file exists
    if not os.path.exists(MESSAGES_FILE):
        print(f"\nERROR: {MESSAGES_FILE} not found.")
        print("Make sure tables/new_sessions_chat_messages.json is present.")
        sys.exit(1)

    # Run all combinations
    results = []
    run_num  = 0
    for model in models:
        for session_id in sessions:
            run_num += 1
            print(f"\n[Run {run_num}/{total}]")
            new_sid = run_replay(session_id, model, dry_run=args.dry_run)
            results.append({
                "original_session": short_id(session_id),
                "model":            model,
                "new_session":      new_sid,
            })
            if run_num < total and not args.dry_run:
                print(f"\n  Cooling down 15s before next run...")
                time.sleep(15)

    # Final summary
    print(f"\n{'='*65}")
    print("REPLAY COMPLETE")
    print(f"{'='*65}")
    print(f"{'Original':<12} {'Model':<20} {'New Session ID'}")
    print("-"*65)
    for r in results:
        print(f"{r['original_session']:<12} {r['model']:<20} {r['new_session']}")
    print(f"\nAll data saved to Supabase. Check your dashboard to verify.")


if __name__ == "__main__":
    main()
