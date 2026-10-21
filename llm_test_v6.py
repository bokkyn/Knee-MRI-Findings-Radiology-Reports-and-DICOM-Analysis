

import os
import re
import sys
import json
import time
import random
import signal
import threading
import unicodedata
import urllib.request
import urllib.error

import pandas as pd






BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FINAL = os.path.join(BASE_DIR, "labels_final.csv")
INPUT_RESEARCH = os.path.join(BASE_DIR, "labels_research.csv")


OUTPUT_CSV = os.path.join(BASE_DIR, "labels_llm_test.csv")
OUTPUT_TXT = os.path.join(BASE_DIR, "labels_llm_test_readable.txt")
OUTPUT_RAW = os.path.join(BASE_DIR, "labels_llm_test_raw.jsonl")


OUTPUT_CSV_TMP = OUTPUT_CSV + ".tmp"






OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

OLLAMA_TIMEOUT = 300

MAX_RETRIES = 2
RETRY_DELAY_SEC = 5






RANDOM_SEED = 2941203







TEST_LIMIT = None








CHECKPOINT_EVERY = 1






SILVER_SKIP_THRESHOLD = 5
SHORT_REPORT_THRESHOLD = 200


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


random.seed(RANDOM_SEED)






STOP_REQUESTED = False


def request_stop():
    global STOP_REQUESTED

    if not STOP_REQUESTED:
        STOP_REQUESTED = True
        print("\n")
        print("=" * 72)
        print("ZAHTJEV ZA PREKID PRIMLJEN.")
        print("Trenutna studija ce se dovrsiti, zatim ce se spremiti checkpoint.")
        print("=" * 72)


def signal_handler(signum, frame):
    request_stop()


signal.signal(signal.SIGINT, signal_handler)


def terminal_listener():
    """
    Omogućava:
        q + Enter
        quit + Enter
        stop + Enter
        exit + Enter

    Trenutna studija se ne prekida nasilno.
    Nakon nje se sprema checkpoint i program završava.
    """

    global STOP_REQUESTED

    while not STOP_REQUESTED:
        try:
            command = input().strip().lower()

            if command in ("q", "quit", "stop", "exit"):
                request_stop()
                return

        except EOFError:
            return
        except Exception:
            return



try:
    if sys.stdin is not None and sys.stdin.isatty():
        listener_thread = threading.Thread(
            target=terminal_listener,
            daemon=True
        )
        listener_thread.start()
except Exception:
    pass






def normalize_text(text):
    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")

    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    return text.strip()






def categorize_study(row):

    n_silver = 0
    n_unresolved = 0
    n_conflict = 0
    n_uncertain = 0

    for t in TARGETS:

        status = str(row.get(f"{t}_status", ""))

        if status == "SILVER":
            n_silver += 1

        elif status == "UNRESOLVED":
            n_unresolved += 1

        elif status == "CONFLICT":
            n_conflict += 1

        elif status == "UNCERTAIN":
            n_uncertain += 1

    report_len = len(
        normalize_text(row.get("Report", ""))
    )

    if n_silver >= SILVER_SKIP_THRESHOLD:
        return "skip"

    if (
        report_len < SHORT_REPORT_THRESHOLD
        and n_unresolved == 0
        and n_conflict == 0
    ):
        return "skip"

    if n_silver == 0:
        return "llm_full"

    return "llm_unresolved_only"






SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist reading a knee MRI report in any language. Extract probabilities for specified findings.

For each finding, return a probability between 0.00 and 1.00.

=== CRITICAL RULES ===

1. EXPLICIT NEGATION -> return 0.00-0.05:
   If the report says the finding is ABSENT: "no X", "without X", "X is normal", "X is intact", "X is unremarkable", "X is unauffällig", "sin X", "no hay X", "kein X", "ohne X", "без X", "χωρίς X", "yok", "nema X", "geen X", "pas de X", "aucun X", "intacte X", "regelrecht X".
   Example: "No fracture" -> Fracture = 0.02

2. EXPLICIT POSITIVE -> return 0.80-1.00:
   If the report explicitly states the finding: "ACL tear", "meniscus rupture", "fracture", "effusion", "Baker's cyst".
   Severity modifiers:
   - "severe", "large", "complete", "full-thickness", "massive" -> 0.90-1.00
   - "moderate" -> 0.80-0.90
   - "mild", "small", "minimal", "trace", "low-grade", "grade 1", "grade 2" -> 0.65-0.80

3. UNCERTAINTY -> return 0.55-0.75:
   If the report uses: "possible", "suspected", "suspicious", "cannot exclude", "may represent", "likely", "R/O", "rule out", "probable", "posible", "sospech", "može odgovarati", "vjerojatno".

4. NOT MENTIONED -> return CONTEXTUAL value (NEVER 0.50):
   If a finding is NOT mentioned in the report:
   - If the report confirms NORMAL state for that anatomical region
     (e.g., "ligaments are normal" for ACL, "menisci are normal" for meniscus),
     return 0.05-0.15.
   - If the report is DETAILED (>= 400 characters) and mentions most
     structures, return 0.15-0.30.
   - If the report is SHORT (< 200 characters) or template-like,
     return 0.35-0.45.
   - NEVER return exactly 0.50.

=== ANATOMICAL DISTINCTIONS (CRITICAL) ===

5. ACL vs PCL - DO NOT CONFUSE:
   - ACL = anterior cruciate ligament = prednji križni = vorderes Kreuzband
     = voorste kruisband = ön çapraz bağ = LCA = πρόσθιος χιαστός
     = предна кръстна връзка
   - PCL = posterior cruciate ligament = stražnji križni = hinteres Kreuzband
     = achterste kruisband = arka çapraz bağ = LCP = οπίσθιος χιαστός
     = задна кръстна връзка
   If the report mentions PCL injury, do NOT set ACL high.
   If the report mentions ACL injury, do NOT set PCL high.

6. Medial vs Lateral - DO NOT SWAP:
   - Medial = medijalni = innen = medial = interno = iç = έσω = медиален = mediaal
   - Lateral = lateralni = außen = lateral = externo = dış = έξω = латерален = lateraal
   Do NOT swap medial and lateral findings.

7. MENISCUS "NORMAL" -> 0.00-0.15:
   If the report says "meniscus is normal", "meniscus is intact",
   "meniscus shows normal morphology and signal", "meniskus normal",
   "meniscus intakt", "menisc normal", "menisk bez znakova rupture",
   return 0.00-0.15 for that meniscus.
   Do NOT return 0.85 just because the word "meniscus" appears.

=== SPECIFIC FINDINGS ===

8. SYNOVITIS - only if EXPLICITLY mentioned:
   Only return high if "synovitis", "synovial thickening",
   "synovial hypertrophy", "sinovitis", "sinovite", "Synovialitis",
   "υμενίτιδα", "синовит" appear explicitly. Do NOT infer from effusion.

9. FRACTURE - only if EXPLICITLY mentioned:
   Only return high if "fracture", "fracture line", "cortical break",
   "fraktura", "Fraktur", "fractuur", "καταγμα", "фрактура" appear
   explicitly. Do NOT infer from bone marrow edema.

10. CONTUSION - only if EXPLICITLY mentioned:
    Only return high if "bone contusion", "bone bruise", "bone marrow edema",
    "koštani edem", "Knochenmarködem", "botoedeem", "kontuzija" appear
    explicitly.

11. EFFUSION - "small", "minimal", "trace" ARE POSITIVE:
    - "small effusion", "minimal effusion", "trace effusion",
      "manja količina izljeva", "minimalan izljev", "Lichte hydrops",
      "μικρή συλλογή", "минимален излив", "minimal mayii" -> 0.65-0.75
    - "moderate effusion" -> 0.80-0.90
    - "large effusion" -> 0.90-1.00
    Do NOT return 0.30-0.50 for small effusion.

12. CHONDRAL GRADING -> OA:
    - Grade 1 chondral change / chondromalacia grade 1 / chondropathy grade 1
      / "hondromalacija I°" / "χονδροπάθεια grade I" -> OA = 0.55
    - Grade 2 -> OA = 0.70
    - Grade 3 -> OA = 0.85
    - Grade 4 -> OA = 0.95
    - "chondromalacia patellae" of any grade -> PF OA = 0.75+
    - If grade 1 is mentioned in a compartment (medial/lateral/patellofemoral),
      return 0.55 for that compartment's OA. Do NOT return 0.10.

13. ANATOMICAL LATERALITY:
    - "medial compartment" / "medial tibiofemoral" -> Medial OA
    - "lateral compartment" / "lateral tibiofemoral" -> Lateral OA
    - "patellofemoral" / "patellar chondromalacia" / "trochlear" -> PF OA

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object starting with { and ending with }.

Example:
{"ACL": 0.05, "MCL": 0.50, "Medial Meniscus": 0.85, "Lateral Meniscus": 0.30, "Medial OA": 0.10, "Lateral OA": 0.10, "PF OA": 0.20, "Effusion": 0.60, "Synovitis": 0.30, "Baker's": 0.05, "Contusion": 0.05, "Fracture": 0.02}

DO NOT return a list.
DO NOT return ["ACL", "MCL", ...].
DO NOT return key-value pairs in brackets.
Return EXACTLY a JSON object with curly braces {}.
"""






def build_user_prompt(report, targets):

    target_lines = "\n".join(
        f"- {t}" for t in targets
    )

    return (
        f"Extract probabilities for these findings:\n"
        f"{target_lines}\n\n"
        f"Report:\n\"\"\"\n"
        f"{report}\n"
        f"\"\"\"\n\n"
        f"Return ONLY a JSON object with curly braces {{}} "
        f"containing exactly these keys: "
        f"{json.dumps(targets)}\n"
        f"Start your response with {{ and end with }}."
    )






def _validate_and_extract(obj, targets, verbose=False):

    if not isinstance(obj, dict):
        return None

    out = {}

    for t in targets:

        v = obj.get(t, None)

        if v is None:

            for k in obj:

                if str(k).lower() == t.lower():
                    v = obj[k]
                    break

        try:

            v = float(v)

            v = max(
                0.0,
                min(1.0, v)
            )

        except (TypeError, ValueError):

            v = None

        out[t] = v

    if all(v is None for v in out.values()):
        return None

    return out


def parse_json_response(content, targets, verbose=False):

    if not content:
        return None

    text = content.strip()

    
    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    
    
    

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        try:

            obj = json.loads(
                text[start:end + 1]
            )

            result = _validate_and_extract(
                obj,
                targets,
                verbose
            )

            if result is not None:

                if verbose:
                    print(
                        "  [PARSE OK] Format 1: cisti JSON"
                    )

                return result

        except json.JSONDecodeError:
            pass

    
    
    

    match = re.search(
        r'\[\s*((?:"[^"]+"\s*,\s*)*"[^"]+")\s*\]'
        r'\s*[:=]\s*'
        r'\[\s*((?:"[^"]+"\s*,\s*)*"[^"]+")\s*\]',
        text,
        re.DOTALL
    )

    if match:

        try:

            keys = json.loads(
                "[" + match.group(1) + "]"
            )

            vals = json.loads(
                "[" + match.group(2) + "]"
            )

            if len(keys) == len(vals):

                obj = dict(
                    zip(keys, vals)
                )

                result = _validate_and_extract(
                    obj,
                    targets,
                    verbose
                )

                if result is not None:

                    if verbose:
                        print(
                            "  [PARSE OK] Format 2"
                        )

                    return result

        except json.JSONDecodeError:
            pass

    
    
    

    list_kv_match = re.search(
        r'\[\s*"[^"]+"\s*:.*\]',
        text,
        re.DOTALL
    )

    if list_kv_match:

        block = list_kv_match.group(0)

        block_json = (
            "{" +
            block[1:-1] +
            "}"
        )

        try:

            obj = json.loads(block_json)

            result = _validate_and_extract(
                obj,
                targets,
                verbose
            )

            if result is not None:

                if verbose:
                    print(
                        "  [PARSE OK] Format 3"
                    )

                return result

        except json.JSONDecodeError:
            pass

    
    
    

    mixed_match = re.search(
        r'\[\s*((?:"[^"]+"\s*,\s*)*"[^"]+")\s*\]'
        r'\s*[:=]\s*'
        r'(\{.*\})',
        text,
        re.DOTALL
    )

    if mixed_match:

        try:

            obj = json.loads(
                mixed_match.group(2)
            )

            result = _validate_and_extract(
                obj,
                targets,
                verbose
            )

            if result is not None:

                if verbose:
                    print(
                        "  [PARSE OK] Format 4"
                    )

                return result

        except json.JSONDecodeError:
            pass

    
    
    

    numbers = re.findall(
        r'(?<!\d)(?:0?\.\d+|1\.0|0|1)(?!\d)',
        text
    )

    numbers = [
        float(n)
        for n in numbers
        if 0 <= float(n) <= 1
    ]

    if len(numbers) >= len(targets):

        vals = numbers[:len(targets)]

        obj = dict(
            zip(targets, vals)
        )

        if verbose:
            print(
                "  [PARSE OK] Format 5: fallback"
            )

        return obj

    if verbose:

        print(
            "  [PARSE FAIL] Svi pokusaji propali."
        )

        print(
            "  [PARSE FAIL] Content:",
            content[:500]
        )

    return None






def call_ollama(report, targets, verbose=True):

    payload = {

        "model": OLLAMA_MODEL,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    report,
                    targets
                )
            }
        ],

        "stream": False,

        "keep_alive": "30m",

        "options": {
            "temperature": 0.0,
            "num_predict": 200,
            "num_ctx": 4096
        }
    }

    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            req,
            timeout=OLLAMA_TIMEOUT
        ) as resp:

            raw = resp.read().decode(
                "utf-8"
            )

            obj = json.loads(raw)

            content = (
                obj
                .get("message", {})
                .get("content", "")
            )

            parsed = parse_json_response(
                content,
                targets,
                verbose=verbose
            )

            return parsed, content

    except urllib.error.HTTPError as e:

        if verbose:
            print(
                f"  [OLLAMA HTTP ERROR] "
                f"{e.code}: {e.reason}"
            )

        return None, None

    except urllib.error.URLError as e:

        if verbose:
            print(
                f"  [OLLAMA URL ERROR] {e}"
            )

        return None, None

    except json.JSONDecodeError as e:

        if verbose:
            print(
                f"  [OLLAMA JSON ERROR] {e}"
            )

        return None, None

    except Exception as e:

        if verbose:
            print(
                f"  [OLLAMA ERROR] "
                f"{type(e).__name__}: {e}"
            )

        return None, None


def call_ollama_with_retry(
    report,
    targets,
    verbose=True
):

    for attempt in range(
        MAX_RETRIES + 1
    ):

        if attempt > 0:

            print(
                f"  [RETRY {attempt}/{MAX_RETRIES}] "
                f"cekam {RETRY_DELAY_SEC}s..."
            )

            time.sleep(
                RETRY_DELAY_SEC
            )

        result, raw = call_ollama(
            report,
            targets,
            verbose=verbose
        )

        if result is not None:
            return result, raw

        if verbose and attempt < MAX_RETRIES:

            print(
                f"  [FAIL pokusaj "
                f"{attempt + 1}]"
            )

    return None, None






def atomic_to_csv(df, path):

    tmp_path = path + ".tmp"

    df.to_csv(
        tmp_path,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(
        tmp_path,
        path
    )






def load_existing_results():

    if not os.path.exists(OUTPUT_CSV):

        print(
            "\nNema prethodnog checkpointa."
        )

        return pd.DataFrame(), set()

    try:

        existing_df = pd.read_csv(
            OUTPUT_CSV,
            low_memory=False
        )

    except Exception as e:

        print(
            "\n[UPOZORENJE] Ne mogu ucitati "
            f"postojeci output CSV: {e}"
        )

        print(
            "Program se NECE prepisivati preko "
            "postojeceg CSV-a."
        )

        raise

    if (
        "StudyInstanceUID"
        not in existing_df.columns
    ):

        raise RuntimeError(
            "Postojeci OUTPUT_CSV nema "
            "StudyInstanceUID stupac."
        )

    existing_df["StudyInstanceUID"] = (
        existing_df["StudyInstanceUID"]
        .astype(str)
    )

    existing_df = (
        existing_df
        .drop_duplicates(
            subset=["StudyInstanceUID"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    processed_uids = set(
        existing_df["StudyInstanceUID"]
    )

    print(
        f"\nPostojeci checkpoint: "
        f"{len(existing_df):,} studija"
    )

    return existing_df, processed_uids






def save_checkpoint(
    existing_df,
    new_results
):

    if not new_results:

        print(
            "  [CHECKPOINT] Nema novih rezultata."
        )

        return existing_df

    new_df = pd.DataFrame(
        new_results
    )

    if existing_df is None or len(existing_df) == 0:

        combined = new_df.copy()

    else:

        combined = pd.concat(
            [
                existing_df,
                new_df
            ],
            ignore_index=True,
            sort=False
        )

    combined["StudyInstanceUID"] = (
        combined["StudyInstanceUID"]
        .astype(str)
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["StudyInstanceUID"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    atomic_to_csv(
        combined,
        OUTPUT_CSV
    )

    print(
        f"  [CHECKPOINT] Spremljeno ukupno: "
        f"{len(combined):,}"
    )

    return combined






def append_raw_log(entry):

    with open(
        OUTPUT_RAW,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                entry,
                ensure_ascii=False
            ) + "\n"
        )

        f.flush()






def write_readable_txt(
    out_df,
    research_df,
    elapsed_total
):

    if out_df is None or len(out_df) == 0:

        print(
            "[TXT] Nema rezultata za zapis."
        )

        return

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 90 + "\n"
        )

        f.write(
            "RSNA KNEE - LLM RESULTS\n"
        )

        f.write(
            "=" * 90 + "\n\n"
        )

        f.write(
            f"Model: {OLLAMA_MODEL}\n"
        )

        f.write(
            f"Timeout: {OLLAMA_TIMEOUT}s\n"
        )

        f.write(
            f"Random seed: {RANDOM_SEED}\n"
        )

        f.write(
            f"Broj spremljenih studija: "
            f"{len(out_df)}\n"
        )

        f.write(
            f"Vrijeme ovog pokretanja: "
            f"{elapsed_total:.1f}s\n"
        )

        f.write("\n")

        
        
        

        f.write(
            "-" * 90 + "\n"
        )

        f.write(
            "LLM VREDNOSTI (statistika)\n"
        )

        f.write(
            "-" * 90 + "\n"
        )

        f.write(
            f"{'Target':<20} "
            f"{'min':>6} "
            f"{'max':>6} "
            f"{'mean':>6} "
            f"{'n':>6}\n"
        )

        f.write(
            "-" * 90 + "\n"
        )

        for t in TARGETS:

            col = f"{t}_LLM"

            if col not in out_df.columns:
                continue

            vals = out_df[col].dropna()

            if len(vals) == 0:
                continue

            f.write(
                f"{t:<20} "
                f"{vals.min():>6.2f} "
                f"{vals.max():>6.2f} "
                f"{vals.mean():>6.2f} "
                f"{len(vals):>6}\n"
            )

        f.write("\n")

        
        
        

        f.write(
            "=" * 90 + "\n"
        )

        f.write(
            "DETALJI PO STUDIJI\n"
        )

        f.write(
            "=" * 90 + "\n"
        )

        
        report_map = {}

        if "StudyInstanceUID" in research_df.columns:

            temp_research = (
                research_df[
                    [
                        c for c in [
                            "StudyInstanceUID",
                            "Report"
                        ]
                        if c in research_df.columns
                    ]
                ]
                .drop_duplicates(
                    subset=["StudyInstanceUID"]
                )
            )

            if "Report" in temp_research.columns:

                report_map = dict(
                    zip(
                        temp_research[
                            "StudyInstanceUID"
                        ].astype(str),
                        temp_research[
                            "Report"
                        ].fillna("")
                    )
                )

        for _, row in out_df.iterrows():

            uid = str(
                row["StudyInstanceUID"]
            )

            report = report_map.get(
                uid,
                ""
            )

            f.write(
                "\n" + "#" * 90 + "\n"
            )

            f.write(
                f"StudyInstanceUID: {uid}\n"
            )

            f.write(
                f"Category: "
                f"{row.get('category', '')}\n"
            )

            f.write(
                f"Language: "
                f"{row.get('detected_language_name', '')}\n"
            )

            f.write(
                f"Report length: "
                f"{row.get('report_length', '')} chars\n"
            )

            f.write(
                f"Elapsed: "
                f"{row.get('elapsed_seconds', '')}s\n"
            )

            f.write(
                "-" * 90 + "\n"
            )

            f.write(
                "REPORT\n"
            )

            f.write(
                "-" * 90 + "\n"
            )

            f.write(
                normalize_text(report)
                + "\n\n"
            )

            f.write(
                "-" * 90 + "\n"
            )

            f.write(
                "RESULTS\n"
            )

            f.write(
                "-" * 90 + "\n"
            )

            f.write(
                f"{'Target':<20} "
                f"{'GOLD':>6} "
                f"{'SILVER':>8} "
                f"{'LLM':>8} "
                f"{'FINAL':>6} "
                f"{'SOURCE':>16}\n"
            )

            f.write(
                "-" * 90 + "\n"
            )

            for t in TARGETS:

                gold = row.get(
                    f"{t}_GOLD"
                )

                silver = row.get(
                    f"{t}_SILVER"
                )

                llm = row.get(
                    f"{t}_LLM"
                )

                fin = row.get(
                    f"{t}_FINAL"
                )

                src = row.get(
                    f"{t}_SOURCE",
                    ""
                )

                g = (
                    "—"
                    if pd.isna(gold)
                    else str(int(gold))
                )

                s = (
                    "—"
                    if pd.isna(silver)
                    else str(int(silver))
                )

                l = (
                    "—"
                    if pd.isna(llm)
                    else f"{float(llm):.2f}"
                )

                fi = (
                    "—"
                    if pd.isna(fin)
                    else str(int(fin))
                )

                f.write(
                    f"{t:<20} "
                    f"{g:>6} "
                    f"{s:>8} "
                    f"{l:>8} "
                    f"{fi:>6} "
                    f"{str(src):>16}\n"
                )






def main():

    global STOP_REQUESTED

    t_start = time.time()

    print("\n" + "=" * 72)
    print("RSNA KNEE - QWEN2.5 / OLLAMA CPU")
    print("=" * 72)

    print(
        f"\nModel: {OLLAMA_MODEL}"
    )

    print(
        f"Timeout: {OLLAMA_TIMEOUT}s"
    )

    print(
        f"Test limit: {TEST_LIMIT}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        "\nZa prekid: upisi q + Enter."
    )

    print(
        "Trenutna studija ce se dovrsiti "
        "prije prekida."
    )

    
    
    

    if not os.path.exists(INPUT_FINAL):

        print(
            "\nGREŠKA: nema labels_final.csv"
        )

        sys.exit(1)

    if not os.path.exists(INPUT_RESEARCH):

        print(
            "\nGREŠKA: nema labels_research.csv"
        )

        sys.exit(1)

    
    
    

    print(
        "\nUcitavam labels_final.csv..."
    )

    final_df = pd.read_csv(
        INPUT_FINAL,
        low_memory=False
    )

    print(
        "Ucitavam labels_research.csv..."
    )

    research_df = pd.read_csv(
        INPUT_RESEARCH,
        low_memory=False
    )

    
    
    

    merge_cols = [
        "StudyInstanceUID",
        "Report",
        "detected_language",
        "detected_language_name",
        "report_length_chars"
    ]

    merge_cols = [
        c
        for c in merge_cols
        if c in research_df.columns
    ]

    df = final_df.merge(
        research_df[merge_cols],
        on="StudyInstanceUID",
        how="left"
    )

    print(
        f"\nUcitano: {len(df):,} studija"
    )

    
    
    

    print(
        "\nKategorizacija..."
    )

    df["category"] = df.apply(
        categorize_study,
        axis=1
    )

    print(
        "\nKategorije:"
    )

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    
    
    

    existing_df, processed_uids = (
        load_existing_results()
    )

    
    
    

    df["StudyInstanceUID"] = (
        df["StudyInstanceUID"]
        .astype(str)
    )

    
    
    

    before_count = len(df)

    df_remaining = df[
        ~df["StudyInstanceUID"].isin(
            processed_uids
        )
    ].copy()

    skipped_checkpoint = (
        before_count - len(df_remaining)
    )

    print(
        f"\nCheckpoint:"
    )

    print(
        f"  Vec obradeno: "
        f"{skipped_checkpoint:,}"
    )

    print(
        f"  Preostalo: "
        f"{len(df_remaining):,}"
    )

    
    
    

    candidates = df_remaining[
        df_remaining["category"] != "skip"
    ].copy()

    
    

    llm_full_candidates = (
        df_remaining[
            df_remaining["category"]
            == "llm_full"
        ]
        .copy()
    )

    if len(llm_full_candidates) >= 1:

        candidates = llm_full_candidates

    
    
    

    if TEST_LIMIT is not None:

        n_select = min(
            TEST_LIMIT,
            len(candidates)
        )

        if n_select > 0:

            sample_indices = random.sample(
                range(len(candidates)),
                n_select
            )

            candidates = (
                candidates
                .iloc[sample_indices]
                .reset_index(drop=True)
            )

    else:

        candidates = (
            candidates
            .reset_index(drop=True)
        )

    
    
    

    if len(candidates) == 0:

        print(
            "\n" + "=" * 72
        )

        print(
            "NEMA NOVIH STUDIJA ZA OBRADU."
        )

        print(
            "=" * 72
        )

        if len(existing_df) > 0:

            write_readable_txt(
                existing_df,
                research_df,
                time.time() - t_start
            )

            print(
                f"\nCheckpoint vec sadrzi "
                f"{len(existing_df):,} studija."
            )

        return

    
    
    

    print(
        f"\nOdabrano {len(candidates)} "
        f"novih studija:"
    )

    for _, row in candidates.iterrows():

        uid = str(
            row["StudyInstanceUID"]
        )

        lang = row.get(
            "detected_language_name",
            "?"
        )

        print(
            f"  {uid[:50]}... "
            f"lang={lang}"
        )

    
    
    

    new_results = []

    processed_this_run = 0

    failed_this_run = 0

    
    
    

    for i, row in candidates.iterrows():

        
        
        if STOP_REQUESTED:

            print(
                "\n[STOP] Prekid prije sljedece studije."
            )

            break

        uid = str(
            row["StudyInstanceUID"]
        )

        report = normalize_text(
            row.get(
                "Report",
                ""
            )
        )

        category = row["category"]

        lang = row.get(
            "detected_language_name",
            "?"
        )

        
        
        

        if category == "llm_full":

            llm_targets = list(
                TARGETS
            )

        else:

            llm_targets = [

                t

                for t in TARGETS

                if str(
                    row.get(
                        f"{t}_status",
                        ""
                    )
                )
                in (
                    "UNRESOLVED",
                    "CONFLICT",
                    "UNCERTAIN"
                )
            ]

        print(
            "\n" + "-" * 72
        )

        print(
            f"[{i + 1}/{len(candidates)}] "
            f"{uid[:50]}..."
        )

        print(
            f"  lang={lang}"
        )

        print(
            f"  category={category}"
        )

        print(
            f"  targets={len(llm_targets)}"
        )

        print(
            f"  report_len={len(report)} chars"
        )

        
        
        

        if not llm_targets:

            print(
                "  [SKIP] Nema LLM targeta."
            )

            continue

        
        
        

        t0 = time.time()

        llm_out, raw_content = (
            call_ollama_with_retry(
                report,
                llm_targets,
                verbose=True
            )
        )

        t1 = time.time()

        elapsed = t1 - t0

        
        
        

        if llm_out is None:

            failed_this_run += 1

            print(
                f"  [FAIL] nakon "
                f"{elapsed:.1f}s"
            )

            append_raw_log(
                {
                    "StudyInstanceUID": uid,
                    "status": "FAIL",
                    "elapsed": round(
                        elapsed,
                        2
                    ),
                    "raw_content": raw_content
                }
            )

            
            
            
            
            

            continue

        
        
        

        processed_this_run += 1

        print(
            f"  [OK] u {elapsed:.1f}s"
        )

        append_raw_log(
            {
                "StudyInstanceUID": uid,
                "status": "OK",
                "elapsed": round(
                    elapsed,
                    2
                ),
                "raw_content": raw_content,
                "parsed": llm_out
            }
        )

        
        
        

        result = {

            "StudyInstanceUID": uid,

            "category": category,

            "detected_language_name": lang,

            "report_length": len(report),

            "llm_targets_count": len(
                llm_targets
            ),

            "elapsed_seconds": round(
                elapsed,
                2
            )
        }

        
        
        

        for t in TARGETS:

            gold = row.get(t)

            status = str(
                row.get(
                    f"{t}_status",
                    ""
                )
            )

            
            
            

            silver_label = (
                gold
                if status == "SILVER"
                else None
            )

            gold_label = (
                gold
                if status == "GOLD"
                else None
            )

            
            
            

            if t in llm_targets:

                llm_value = llm_out.get(
                    t,
                    None
                )

            else:

                llm_value = None

            
            
            

            if pd.notna(gold_label):

                final_value = int(
                    gold_label
                )

                final_source = "GOLD"

            elif status == "SILVER":

                if pd.notna(gold):

                    final_value = int(
                        gold
                    )

                    final_source = "SILVER"

                else:

                    final_value = None

                    final_source = (
                        "SILVER_MISSING"
                    )

            elif llm_value is not None:

                if llm_value >= 0.75:

                    final_value = 1

                    final_source = (
                        "LLM_POS"
                    )

                elif llm_value <= 0.25:

                    final_value = 0

                    final_source = (
                        "LLM_NEG"
                    )

                else:

                    final_value = None

                    final_source = (
                        "LLM_UNCERTAIN"
                    )

            else:

                final_value = None

                final_source = (
                    "UNRESOLVED"
                )

            
            
            

            result[
                f"{t}_GOLD"
            ] = (
                int(gold_label)
                if pd.notna(gold_label)
                else None
            )

            result[
                f"{t}_SILVER"
            ] = (
                int(silver_label)
                if silver_label is not None
                and pd.notna(silver_label)
                else None
            )

            result[
                f"{t}_LLM"
            ] = llm_value

            result[
                f"{t}_FINAL"
            ] = final_value

            result[
                f"{t}_SOURCE"
            ] = final_source

        
        
        

        new_results.append(
            result
        )

        
        processed_uids.add(uid)

        
        
        

        if (
            len(new_results)
            >= CHECKPOINT_EVERY
        ):

            existing_df = save_checkpoint(
                existing_df,
                new_results
            )

            new_results = []

        
        
        

        if STOP_REQUESTED:

            print(
                "\n[STOP] Trenutna studija "
                "je zavrsena."
            )

            print(
                "[STOP] Checkpoint je spremljen."
            )

            break

    
    
    

    if new_results:

        existing_df = save_checkpoint(
            existing_df,
            new_results
        )

        new_results = []

    
    
    

    total_elapsed = (
        time.time() - t_start
    )

    if existing_df is not None and len(existing_df) > 0:

        print(
            "\nGeneriram readable TXT..."
        )

        write_readable_txt(
            existing_df,
            research_df,
            total_elapsed
        )

        print(
            f"Spremljeno: {OUTPUT_TXT}"
        )

    
    
    

    print(
        "\n" + "=" * 72
    )

    print(
        "RUN ZAVRSEN"
    )

    print(
        "=" * 72
    )

    print(
        f"Uspjesno ovaj run: "
        f"{processed_this_run}"
    )

    print(
        f"Fail ovaj run: "
        f"{failed_this_run}"
    )

    print(
        f"Ukupno spremljeno: "
        f"{len(existing_df):,}"
    )

    print(
        f"Vrijeme ovog runa: "
        f"{total_elapsed:.1f}s"
    )

    print(
        f"\nCSV: {OUTPUT_CSV}"
    )

    print(
        f"RAW: {OUTPUT_RAW}"
    )

    print(
        f"TXT: {OUTPUT_TXT}"
    )

    if STOP_REQUESTED:

        print(
            "\nSTOP: program je zaustavljen "
            "na sigurnom checkpointu."
        )

        print(
            "Kod sljedeceg pokretanja "
            "nastavlja dalje."
        )

    print(
        "\n" + "=" * 72
    )






if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\n[CTRL+C] Program prekinut."
        )

        print(
            "Sve prethodno spremljene studije "
            "ostaju u CSV-u."
        )

    except Exception as e:

        print(
            "\n" + "=" * 72
        )

        print(
            "KRITICNA GRESKA"
        )

        print(
            "=" * 72
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        print(
            "\nAko je checkpoint vec postojao, "
            "nije obrisan."
        )

        raise