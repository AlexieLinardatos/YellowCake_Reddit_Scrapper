import requests
import sys
import json
import re
from urllib.parse import unquote

# =========================
# CONFIG
# =========================
CLIENT_ID = "dTSp2P_mT9OTNmehKt_Q8Q"
CLIENT_SECRET = "190768062621529734110534502513734991218"
REDIRECT_URI = "http://localhost:8501"

# AUTH_CODE is the value after code= in:
# http://localhost:8501/?code=XXXX
AUTH_CODE = "OX.NzUzZggbMXedHont-r9JEXqOgF.F0igLl52lpOSMNDj0ZKDbctpRrtHa-5jTgpSKuldx50u9tOtPfypaQqm5CuwRDs6BXuJD9CSbYdhZ-c8SwHQpurxk6P6WvaelQJbr0c4VivpXBzNxGTAgmLRzA-zN8hdKZu3OdD5987tY.oFhBgGmpMR8oEQkaw7gsYQHoCXYnsgUAp3tpcOJhHQZ.CSKKkOezQuWYAw-cQPWLPBCFCMuBR2ua9rxdq78wOCYNMpTE9yneCD2.v7mHlICw-bI2fbaCREnFGeethRQ%3D"
AUTH_CODE = unquote(AUTH_CODE)

ANSWERS_PATH = "answers.json"

# =========================
# Helpers (matching + debug)
# =========================
def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s-]", "", s)  # remove punctuation
    return s

def debug_question(q: dict):
    """Print the exact SurveyMonkey choice texts + IDs for a question."""
    qid = q.get("id")
    heading = (q.get("headings", [{}])[0].get("heading") or "").strip()
    fam = q.get("family")
    sub = q.get("subtype")
    ans = q.get("answers") or {}

    print("\n=== QUESTION DEBUG ===")
    print("Question ID:", qid)
    print("Heading:", heading)
    print("Family/Subtype:", fam, "/", sub)
    print("answers keys:", list(ans.keys()))

    if "choices" in ans and ans.get("choices"):
        print("CHOICES:")
        for ch in ans.get("choices", []):
            print(" -", ch.get("id"), "=>", repr(ch.get("text")))
    if "rows" in ans and ans.get("rows"):
        print("ROWS (matrix questions need row_id + choice_id):")
        for r in ans.get("rows", []):
            print(" -", r.get("id"), "=>", repr(r.get("text")))

def find_choice_id(question_obj: dict, answer_text: str):
    """
    Find a choice_id for a choice-based question, using robust matching:
    - normalized exact
    - numeric prefix (e.g., "10" matches "10 - Extremely likely")
    - contains match as last resort
    """
    if answer_text is None:
        return None

    ans = question_obj.get("answers", {}) or {}
    choices = ans.get("choices", []) or []
    if not choices:
        return None

    a_raw = str(answer_text).strip()
    a = norm(a_raw)

    # 1) exact normalized match
    for ch in choices:
        if norm(ch.get("text", "")) == a:
            return ch.get("id")

    # 2) numeric support: "10" matches "10 - Extremely likely"
    if re.fullmatch(r"\d+", a_raw):
        for ch in choices:
            t = (ch.get("text", "") or "").strip()
            if t.startswith(a_raw):
                return ch.get("id")

    # 3) contains match (last resort)
    for ch in choices:
        ct = norm(ch.get("text", ""))
        if a and (a in ct or ct in a):
            return ch.get("id")

    return None

def is_choice_question(q: dict) -> bool:
    """Heuristic: if question has answers.choices, it needs choice_id (not text)."""
    ans = q.get("answers") or {}
    return bool(ans.get("choices"))

# =========================
# 1) Exchange code -> access token
# =========================
print("Exchanging OAuth code for access token...")

token_resp = requests.post(
    "https://api.surveymonkey.com/oauth/token",
    data={
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": AUTH_CODE,
    },
    timeout=20,
)

if token_resp.status_code != 200:
    print("❌ Token exchange HTTP error:", token_resp.status_code)
    print(token_resp.text)
    sys.exit(1)

token_data = token_resp.json()
if "access_token" not in token_data:
    print("❌ Token exchange failed:", token_data)
    sys.exit(1)

ACCESS_TOKEN = token_data["access_token"]
ACCESS_URL = token_data.get("access_url", "https://api.surveymonkey.com")
print("✅ Access token acquired")

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json",
}

# =========================
# 2) List surveys
# =========================
print("\nFetching your surveys...")
surveys_resp = requests.get(f"{ACCESS_URL}/v3/surveys", headers=headers, timeout=20)
surveys_data = surveys_resp.json()

surveys = surveys_data.get("data", [])
if not surveys:
    print("❌ No surveys found. Response:")
    print(surveys_data)
    sys.exit(1)

for i, s in enumerate(surveys, 1):
    print(f"{i}. {s.get('title', '(untitled)')}  [id={s.get('id')}]")

choice = int(input("\nSelect a survey number: ")) - 1
SURVEY_ID = surveys[choice]["id"]
print(f"\nUsing survey ID: {SURVEY_ID}")

# =========================
# 3) Fetch survey details
# =========================
print("\nFetching survey details...")
details_resp = requests.get(
    f"{ACCESS_URL}/v3/surveys/{SURVEY_ID}/details", headers=headers, timeout=20
)
details = details_resp.json()

if "error" in details:
    print("❌ Error fetching survey details:")
    print(details)
    sys.exit(1)

# =========================
# 3b) Export questions + types (always)
# =========================
lines = []
for page in details.get("pages", []):
    for q in page.get("questions", []):
        headings = q.get("headings", [])
        if not headings:
            continue
        txt = (headings[0].get("heading") or "").strip()
        if not txt:
            continue
        q_family = (q.get("family") or "").strip()
        q_subtype = (q.get("subtype") or "").strip()
        q_type = f"{q_family}/{q_subtype}" if q_family or q_subtype else "unknown"
        lines.append(f"{txt} | type={q_type}")

if lines:
    with open("survey_questions.txt", "w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            f.write(f"{i}. {line}\n")
    print(f"✅ Wrote {len(lines)} questions to survey_questions.txt")

# =========================
# 4) Pick a collector
# =========================
print("\nFetching collectors for this survey...")
collectors_data = requests.get(
    f"{ACCESS_URL}/v3/surveys/{SURVEY_ID}/collectors",
    headers=headers,
    timeout=20
).json()

collectors = collectors_data.get("data", [])
if not collectors:
    print("❌ No collectors found for this survey.")
    print("Create a Web Link collector in SurveyMonkey UI (recommended) or via API.")
    print("Response:", collectors_data)
    sys.exit(1)

for i, c in enumerate(collectors, 1):
    print(f"{i}. {c.get('name','(no name)')}  [type={c.get('type')}]  [id={c.get('id')}]")

collector_choice = int(input("\nSelect a collector number: ")) - 1
COLLECTOR_ID = collectors[collector_choice]["id"]
print(f"✅ Using collector: {COLLECTOR_ID}")

# =========================
# 5) Load answers.json
# =========================
try:
    with open(ANSWERS_PATH, "r", encoding="utf-8") as f:
        answers_by_heading = json.load(f)
except Exception as e:
    print(f"❌ Could not load {ANSWERS_PATH}: {e}")
    sys.exit(1)

# =========================
# 6) Build response payload
# =========================
pages_payload = []
missing_choice_mappings = []

for page in details.get("pages", []):
    page_id = page.get("id")
    q_payloads = []

    for q in page.get("questions", []):
        headings = q.get("headings", []) or []
        heading = (headings[0].get("heading") if headings else "") or ""
        heading = heading.strip()

        if not heading:
            continue

        if heading not in answers_by_heading:
            continue  # no local answer for this question

        user_answer = answers_by_heading[heading]

        # Choice-based question: must provide choice_id
        if is_choice_question(q):
            cid = find_choice_id(q, str(user_answer))
            if cid:
                q_payloads.append({
                    "id": q.get("id"),
                    "answers": [{"choice_id": cid}]
                })
            else:
                # record missing mapping + show debug for this question
                missing_choice_mappings.append((heading, q.get("id"), user_answer))
                # don't add invalid answer payload (will 400)
                continue
        else:
            # Open-ended: text is fine
            q_payloads.append({
                "id": q.get("id"),
                "answers": [{"text": str(user_answer)}]
            })

    if q_payloads:
        pages_payload.append({
            "id": page_id,
            "questions": q_payloads
        })

if missing_choice_mappings:
    print("\n❌ Some answers did NOT match the survey's choice text, so we cannot submit yet.")
    print("These questions require a choice_id, but your answer text didn't match any choice.\n")
    for heading, qid, ans in missing_choice_mappings:
        print(f"- QID={qid} | Heading={heading}")
        print(f"  Your answer: {repr(ans)}")
        # Print exact available choices for THIS question
        for page in details.get("pages", []):
            for qq in page.get("questions", []):
                if str(qq.get("id")) == str(qid):
                    debug_question(qq)

    print("\n✅ Fix: update answers.json to match one of the printed CHOICES exactly,")
    print("   OR change your answer to a number if the choices are numeric (0-10, 1-5, etc).")
    print("\nThen re-run the script.")
    sys.exit(1)

if not pages_payload:
    print("❌ No answers matched any questions. Check that headings in answers.json match the survey headings.")
    sys.exit(1)

response_body = {"pages": pages_payload}

# =========================
# 7) POST create response
# =========================
print("\nSubmitting response to SurveyMonkey...")

create_resp = requests.post(
    f"{ACCESS_URL}/v3/collectors/{COLLECTOR_ID}/responses",
    headers={**headers, "Content-Type": "application/json"},
    json=response_body,
    timeout=20
)

print("Status:", create_resp.status_code)
print("Response:", create_resp.text)

if create_resp.status_code not in (200, 201):
    print("❌ Failed to create response.")
    print("Most common causes now:")
    print("- Collector is closed / not accepting responses")
    print("- A question needs (row_id + choice_id) for matrix questions")
    print("- Your text still didn't match a choice")
    sys.exit(1)

print("✅ Response created!")

#
