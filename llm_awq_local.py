

"""
RSNA KNEE - LLM CLASSIFICATION (LOKALNO, CHECKPOINT VERSION)
=============================================================

Qwen2.5 AWQ/FP16 lokalna obrada radioloških izvještaja.

Glavne mogućnosti:
- automatski checkpoint nakon SVAKOG obrađenog reporta
- automatski checkpoint svakih N studija
- nastavak nakon crasha / restarta računala
- već obrađeni StudyInstanceUID se automatski preskaču
- možeš u terminal upisati:
      q
      quit
      stop
  i program će završiti nakon trenutne studije
- trenutni rezultat se uvijek prvo sprema pa tek onda prelazi na sljedeću studiju
- postojeći CSV se NE prepisuje obrađenim rezultatima
- raw JSONL se također nastavlja
- ako program pukne, pri sljedećem pokretanju nastavlja od prvog neobrađenog UID-a

PUTANJE:
  ./qwen_awq_model/
  ./labels_final.csv
  ./labels_research.csv

OUTPUT:
  ./labels_llm_output.csv
  ./labels_llm_output_readable.txt
  ./labels_llm_output_raw.jsonl

ZA TEST:
  TEST_LIMIT = 4

ZA CIJELI DATASET:
  TEST_LIMIT = None
"""

import os
import re
import sys
import json
import time
import random
import unicodedata
import threading
import queue
import signal

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer






BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "qwen_awq_model")
INPUT_FINAL = os.path.join(BASE_DIR, "labels_final.csv")
INPUT_RESEARCH = os.path.join(BASE_DIR, "labels_research.csv")

OUTPUT_CSV = os.path.join(BASE_DIR, "labels_llm_output.csv")
OUTPUT_TXT = os.path.join(BASE_DIR, "labels_llm_output_readable.txt")
OUTPUT_RAW = os.path.join(BASE_DIR, "labels_llm_output_raw.jsonl")





TEST_LIMIT = 2




RANDOM_SEED = 20244351004

SILVER_SKIP_THRESHOLD = 5
SHORT_REPORT_THRESHOLD = 200


CHECKPOINT_EVERY = 1



MAX_REPORT_CHARS = None

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
    STOP_REQUESTED = True
    print("\n")
    print("=" * 72)
    print("STOP ZATRAŽEN")
    print("Program će završiti nakon trenutne studije.")
    print("Trenutni rezultat će biti spremljen.")
    print("=" * 72)






def signal_handler(sig, frame):
    request_stop()


signal.signal(signal.SIGINT, signal_handler)






def terminal_listener():
    """
    Omogućuje:
        q
        quit
        stop

    u terminalu.

    Ne prekida trenutni inference.
    Samo postavlja STOP_REQUESTED=True.
    """

    global STOP_REQUESTED

    while True:
        try:
            command = input().strip().lower()

            if command in ("q", "quit", "stop", "exit"):
                STOP_REQUESTED = True

                print("\n")
                print("=" * 72)
                print("STOP ZATRAŽEN IZ TERMINALA")
                print("Završavam nakon trenutne studije...")
                print("=" * 72)

                break

        except EOFError:
            break
        except Exception:
            break



listener_thread = threading.Thread(
    target=terminal_listener,
    daemon=True
)

listener_thread.start()






def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.replace(
        "\xa0",
        " "
    )

    text = text.replace(
        "\u200b",
        " "
    )

    text = re.sub(
        r"[ \t\r\f\v]+",
        " ",
        text
    )

    return text.strip()






def categorize_study(row):

    n_silver = 0
    n_unresolved = 0
    n_conflict = 0

    for t in TARGETS:

        status = str(
            row.get(
                f"{t}_status",
                ""
            )
        )

        if status == "SILVER":
            n_silver += 1

        elif status == "UNRESOLVED":
            n_unresolved += 1

        elif status == "CONFLICT":
            n_conflict += 1

    report_len = len(
        normalize_text(
            row.get(
                "Report",
                ""
            )
        )
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
   If the report says the finding is ABSENT: "no X", "without X", "X is normal", 
   "X is intact", "X is unremarkable", "X is unauffällig", "sin X", "no hay X", 
   "kein X", "ohne X", "без X", "χωρίς X", "yok", "nema X", "geen X", 
   "pas de X", "aucun X", "intacte X", "regelrecht X".

2. EXPLICIT POSITIVE -> return 0.80-1.00:
   Only for "complete", "severe", "large", "full-thickness", "massive".
   "moderate" -> 0.70-0.85
   "mild", "small", "minimal", "trace", "low-grade" -> 0.35-0.55

3. UNCERTAINTY -> return 0.55-0.75:
   "possible", "suspected", "suspicious", "cannot exclude", "may represent", 
   "likely", "R/O", "rule out", "probable", "posible", "sospech".

4. NOT MENTIONED -> CONTEXTUAL (NEVER 0.50):
   - NORMAL for that region -> 0.05-0.15
   - DETAILED (>=400 chars) -> 0.15-0.30
   - SHORT (<200 chars) -> 0.35-0.45
   - NEVER return exactly 0.50

5. ACL vs PCL - DO NOT CONFUSE:
   ACL = anterior cruciate = prednji križni = vorderes Kreuzband 
       = ön çapraz bağ = LCA = πρόσθιος χιαστός.
   PCL = posterior cruciate = stražnji križni = hinteres Kreuzband
       = arka çapraz bağ = LCP = οπίσθιος χιαστός.

6. Medial vs Lateral - DO NOT SWAP:
   Greek: "έσω" = MEDIAL, "έξω" = LATERAL.
   Turkish: "medial" = MEDIAL, "lateral" = LATERAL.
   Dutch: "mediale" = MEDIAL, "laterale" = LATERAL.
   Croatian: "medijalni" = MEDIAL, "lateralni" = LATERAL.

7. MENISCUS "NORMAL" -> 0.00-0.15.

8. SYNOVITIS - only if EXPLICIT:
   Only if "synovitis", "synovial thickening", "υμενίτιδα", "синовит".

9. FRACTURE - only if EXPLICIT:
   Only if "fracture", "fraktur", "καταγμα", "фрактура".

10. CONTUSION - only if EXPLICIT + TRAUMA.

11. EFFUSION - "small/minimal/trace" ARE POSITIVE -> 0.65-0.75.

12. CHONDRAL GRADING -> OA:
    Grade 1 -> 0.55; Grade 2 -> 0.70; Grade 3 -> 0.85; Grade 4 -> 0.95.
    "chondromalacia patellae" -> PF OA = 0.75+.

13. BAKER'S SYNONYM:
    "popliteal cyst", "popliteale cyste", "quiste poplíteo" -> Baker's = 0.85+.

14. MCL "NORMAL" -> 0.05-0.15.

=== OUTPUT FORMAT ===

Return ONLY a valid JSON object starting with { and ending with }.

Example:
{"ACL": 0.05, "MCL": 0.05, "Medial Meniscus": 0.85, "Lateral Meniscus": 0.05, "Medial OA": 0.10, "Lateral OA": 0.10, "PF OA": 0.20, "Effusion": 0.70, "Synovitis": 0.05, "Baker's": 0.85, "Contusion": 0.05, "Fracture": 0.02}

DO NOT return a list.
Return EXACTLY a JSON object with curly braces {}.
"""


def build_user_prompt(report, targets):

    target_lines = "\n".join(
        f"- {t}"
        for t in targets
    )

    return (
        f"Extract probabilities for these findings:\n"
        f"{target_lines}\n\n"
        f"Report:\n"
        f"\"\"\"\n"
        f"{report}\n"
        f"\"\"\"\n\n"
        f"Return ONLY a JSON object with curly braces {{}} "
        f"containing exactly these keys: "
        f"{json.dumps(targets)}\n"
        f"Start your response with {{ and end with }}."
    )






def _validate_and_extract(obj, targets):

    if not isinstance(obj, dict):
        return None

    out = {}

    for t in targets:

        v = obj.get(
            t,
            None
        )

        if v is None:

            for k in obj:

                if k.lower() == t.lower():
                    v = obj[k]
                    break

        try:

            v = float(v)

            v = max(
                0.0,
                min(1.0, v)
            )

        except (
            TypeError,
            ValueError
        ):

            v = None

        out[t] = v

    if all(
        v is None
        for v in out.values()
    ):
        return None

    return out


def parse_json_response(
    content,
    targets,
    verbose=False
):

    if not content:
        return None

    text = content.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:

            obj = json.loads(
                text[start:end + 1]
            )

            result = _validate_and_extract(
                obj,
                targets
            )

            if result is not None:
                return result

        except json.JSONDecodeError:
            pass

    
    numbers = re.findall(
        r'(\d+\.\d+|\d+)',
        text
    )

    numbers = [
        float(n)
        for n in numbers
        if 0 <= float(n) <= 1
    ]

    if len(numbers) >= len(targets):

        return dict(
            zip(
                targets,
                numbers[:len(targets)]
            )
        )

    if verbose:

        print(
            f"  [PARSE FAIL] "
            f"Content: {content[:500]}"
        )

    return None






def atomic_to_csv(df, path):

    temp_path = path + ".tmp"

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig"
    )

    os.replace(
        temp_path,
        path
    )






def load_existing_results():

    if not os.path.exists(OUTPUT_CSV):

        print("\nNema postojećeg checkpointa.")

        return pd.DataFrame()

    try:

        existing = pd.read_csv(
            OUTPUT_CSV,
            low_memory=False
        )

        print(
            f"\nPronađen postojeći checkpoint: "
            f"{len(existing):,} studija"
        )

        return existing

    except Exception as e:

        print(
            "\nUPOZORENJE: postojeći CSV "
            "nije moguće učitati."
        )

        print(
            f"Greška: {e}"
        )

        return pd.DataFrame()






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
            )
            + "\n"
        )






print("=" * 72)
print("RSNA KNEE - LLM CLASSIFICATION (LOKALNO)")
print("=" * 72)

print(
    f"\nCUDA available: "
    f"{torch.cuda.is_available()}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

else:

    print(
        "NEMA GPU — model ide na CPU (SPORO!)"
    )


if not os.path.exists(MODEL_PATH):

    print(
        f"\nGREŠKA: Model ne postoji:"
    )

    print(
        MODEL_PATH
    )

    sys.exit(1)


print(
    f"\nUčitavam model iz:"
)

print(
    MODEL_PATH
)

t0 = time.time()


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


if torch.cuda.is_available():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

else:

    print(
        "Učitavam na CPU..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


model.eval()


print(
    f"Model spreman u "
    f"{time.time() - t0:.1f}s"
)






def classify_report(
    report_text,
    target_list
):

    user_prompt = build_user_prompt(
        report_text,
        target_list
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        out[len(inp):]
        for inp, out in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    parsed = parse_json_response(
        response,
        target_list,
        verbose=True
    )

    return parsed, response






print(
    "\nUčitavam CSV-ove..."
)


if not os.path.exists(INPUT_FINAL):

    print(
        f"GREŠKA: {INPUT_FINAL} ne postoji"
    )

    sys.exit(1)


if not os.path.exists(INPUT_RESEARCH):

    print(
        f"GREŠKA: {INPUT_RESEARCH} ne postoji"
    )

    sys.exit(1)


final_df = pd.read_csv(
    INPUT_FINAL,
    low_memory=False
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
    "report_length_chars",
]

merge_cols = [
    c
    for c in merge_cols
    if c in research_df.columns
]


df = final_df.merge(
    research_df[merge_cols],
    on="StudyInstanceUID",
    how="left",
)


print(
    f"Učitano: "
    f"{len(df):,} studija"
)






df["category"] = df.apply(
    categorize_study,
    axis=1
)


print("\nKategorije:")

print(
    df["category"]
    .value_counts()
    .to_string()
)






existing_df = load_existing_results()


if (
    not existing_df.empty
    and "StudyInstanceUID" in existing_df.columns
):

    processed_uids = set(
        existing_df[
            "StudyInstanceUID"
        ].astype(str)
    )

else:

    processed_uids = set()


print(
    f"Već obrađeno: "
    f"{len(processed_uids):,}"
)






candidates = df[
    df["category"] != "skip"
].copy()







if processed_uids:

    candidates = candidates[
        ~candidates[
            "StudyInstanceUID"
        ].astype(str).isin(
            processed_uids
        )
    ].copy()






if TEST_LIMIT is not None:

    n = min(
        TEST_LIMIT,
        len(candidates)
    )

    sample_indices = random.sample(
        range(len(candidates)),
        n
    )

    candidates = candidates.iloc[
        sample_indices
    ].reset_index(
        drop=True
    )

else:

    candidates = candidates.reset_index(
        drop=True
    )


print(
    f"\nPreostalo za obradu: "
    f"{len(candidates):,}"
)


if len(candidates) == 0:

    print(
        "\nNema novih studija za obradu."
    )

    sys.exit(0)


for _, row in candidates.iterrows():

    print(
        f"  "
        f"{str(row['StudyInstanceUID'])[:50]}..."
        f" lang={row.get('detected_language_name', '?')}"
    )






results = []

t_start = time.time()

successful = 0
failed = 0
skipped = 0


for i, row in candidates.iterrows():

    
    
    
    

    if STOP_REQUESTED:

        print(
            "\nSTOP prije sljedeće studije."
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

    if MAX_REPORT_CHARS is not None:

        report = report[
            :MAX_REPORT_CHARS
        ]


    category = row[
        "category"
    ]

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
                "UNCERTAIN",
            )
        ]


    if not llm_targets:

        print(
            f"\n[{i+1}/{len(candidates)}] "
            f"{uid[:40]}..."
        )

        print(
            "  Nema targeta za LLM."
        )

        skipped += 1

        continue


    print("\n")
    print("=" * 72)

    print(
        f"[{i+1}/{len(candidates)}] "
        f"{uid[:50]}..."
    )

    print(
        f"lang={lang} "
        f"cat={category} "
        f"targets={len(llm_targets)}"
    )

    print(
        f"report_len={len(report)} chars"
    )

    print(
        "Za prekid upiši 'q', 'quit' ili 'stop' "
        "u terminal."
    )

    print("=" * 72)


    
    
    

    t0 = time.time()

    try:

        llm_out, raw_content = classify_report(
            report,
            llm_targets
        )

    except KeyboardInterrupt:

        request_stop()

        print(
            "Inference prekinut."
        )

        break

    except Exception as e:

        elapsed = time.time() - t0

        print(
            f"\n  [ERROR] "
            f"{type(e).__name__}: {e}"
        )

        failed += 1

        
        
        
        
        
        

        append_raw_log(
            {
                "StudyInstanceUID": uid,
                "status": "ERROR",
                "elapsed": round(
                    elapsed,
                    2
                ),
                "error": str(e),
            }
        )

        continue


    elapsed = time.time() - t0


    
    
    

    if llm_out is None:

        print(
            f"  FAIL nakon "
            f"{elapsed:.1f}s"
        )

        failed += 1

        append_raw_log(
            {
                "StudyInstanceUID": uid,
                "status": "FAIL",
                "elapsed": round(
                    elapsed,
                    2
                ),
                "raw_content": raw_content,
            }
        )

        
        
        
        
        

        continue


    
    
    

    print(
        f"  OK u "
        f"{elapsed:.1f}s"
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
            "parsed": llm_out,
        }
    )


    result = {

        "StudyInstanceUID": uid,

        "category": category,

        "detected_language_name": lang,

        "report_length": len(report),

        "elapsed_seconds": round(
            elapsed,
            2
        ),
    }


    
    
    

    for t in TARGETS:

        gold = row.get(t)

        status = str(
            row.get(
                f"{t}_status",
                ""
            )
        )


        gold_label = (
            gold
            if status == "GOLD"
            else None
        )


        silver_label = (
            gold
            if status == "SILVER"
            else None
        )


        llm_value = (
            llm_out.get(t, None)
            if t in llm_targets
            else None
        )


        
        
        

        if pd.notna(gold_label):

            final_value = int(
                gold_label
            )

            final_source = "GOLD"


        elif status == "SILVER":

            final_value = int(
                gold
            )

            final_source = "SILVER"


        elif llm_value is not None:

            if llm_value >= 0.75:

                final_value = 1

                final_source = "LLM_POS"


            elif llm_value <= 0.25:

                final_value = 0

                final_source = "LLM_NEG"


            else:

                final_value = None

                final_source = "LLM_UNCERTAIN"


        else:

            final_value = None

            final_source = "UNRESOLVED"


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


    
    
    

    results.append(
        result
    )

    successful += 1


    
    
    

    if (
        successful % CHECKPOINT_EVERY == 0
    ):

        new_df = pd.DataFrame(
            results
        )


        if existing_df.empty:

            checkpoint_df = new_df.copy()

        else:

            checkpoint_df = pd.concat(
                [
                    existing_df,
                    new_df
                ],
                ignore_index=True
            )


        
        
        

        checkpoint_df = checkpoint_df.drop_duplicates(
            subset=[
                "StudyInstanceUID"
            ],
            keep="last"
        )


        atomic_to_csv(
            checkpoint_df,
            OUTPUT_CSV
        )


        print(
            f"\n  CHECKPOINT spremljen:"
        )

        print(
            f"  {OUTPUT_CSV}"
        )

        print(
            f"  Ukupno spremljeno: "
            f"{len(checkpoint_df):,}"
        )


        
        
        

        existing_df = checkpoint_df.copy()

        processed_uids = set(
            existing_df[
                "StudyInstanceUID"
            ].astype(str)
        )


        
        
        
        

        results = []


    
    
    

    if STOP_REQUESTED:

        print(
            "\nTrenutna studija je spremljena."
        )

        print(
            "Program se sada sigurno zaustavlja."
        )

        break






if results:

    new_df = pd.DataFrame(
        results
    )


    if existing_df.empty:

        final_df = new_df.copy()

    else:

        final_df = pd.concat(
            [
                existing_df,
                new_df
            ],
            ignore_index=True
        )


    final_df = final_df.drop_duplicates(
        subset=[
            "StudyInstanceUID"
        ],
        keep="last"
    )


    atomic_to_csv(
        final_df,
        OUTPUT_CSV
    )


    existing_df = final_df

    results = []






elapsed_total = time.time() - t_start


print("\n")
print("=" * 72)
print("TRENUTNO STANJE")
print("=" * 72)

print(
    f"Uspješno u ovoj sesiji: "
    f"{successful}"
)

print(
    f"FAIL u ovoj sesiji: "
    f"{failed}"
)

print(
    f"Skip: "
    f"{skipped}"
)

print(
    f"Ukupno spremljeno: "
    f"{len(existing_df):,}"
)

print(
    f"Vrijeme ove sesije: "
    f"{elapsed_total:.1f}s"
)


if successful > 0:

    print(
        f"Prosjek uspješnih: "
        f"{elapsed_total / successful:.1f}s/studij"
    )






def generate_readable_txt(
    output_df,
    research_df
):

    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 90
            + "\n"
        )

        f.write(
            "RSNA KNEE - LLM CLASSIFICATION RESULTS\n"
        )

        f.write(
            "=" * 90
            + "\n\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n"
        )

        f.write(
            f"Broj studija: "
            f"{len(output_df)}\n\n"
        )


        for _, row in output_df.iterrows():

            uid = row[
                "StudyInstanceUID"
            ]


            report_row = research_df[
                research_df[
                    "StudyInstanceUID"
                ].astype(str)
                == str(uid)
            ]


            report = (
                report_row.iloc[0]["Report"]
                if len(report_row)
                else ""
            )


            f.write(
                "\n"
                + "#" * 90
                + "\n"
            )


            f.write(
                f"StudyInstanceUID: "
                f"{uid}\n"
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
                f"Elapsed: "
                f"{row.get('elapsed_seconds', '')}s\n"
            )


            f.write(
                "-" * 90
                + "\n"
            )


            f.write(
                "REPORT\n"
            )


            f.write(
                "-" * 90
                + "\n"
            )


            f.write(
                normalize_text(
                    report
                )
                + "\n\n"
            )


            f.write(
                "-" * 90
                + "\n"
            )


            f.write(
                "RESULTS\n"
            )


            f.write(
                "-" * 90
                + "\n"
            )


            f.write(
                f"{'Target':<20} "
                f"{'GOLD':>6} "
                f"{'SILVER':>8} "
                f"{'LLM':>8} "
                f"{'FINAL':>6} "
                f"{'SOURCE':>18}\n"
            )


            f.write(
                "-" * 90
                + "\n"
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
                    f"{src:>18}\n"
                )






if (
    not existing_df.empty
    and "StudyInstanceUID"
    in existing_df.columns
):

    try:

        generate_readable_txt(
            existing_df,
            research_df
        )

        print(
            f"\nReadable TXT spremljen:"
        )

        print(
            OUTPUT_TXT
        )

    except Exception as e:

        print(
            f"\nUPOZORENJE: TXT nije generiran:"
        )

        print(
            e
        )






print("\n")
print("=" * 72)

if STOP_REQUESTED:

    print(
        "PROGRAM ZAUSTAVLJEN SIGURNO"
    )

    print(
        "Sljedećim pokretanjem nastavlja "
        "od prvog neobrađenog StudyInstanceUID-a."
    )

else:

    print(
        "TRENUTNA OBRADA ZAVRŠENA"
    )

print("=" * 72)

print(
    f"\nCheckpoint CSV:"
)

print(
    OUTPUT_CSV
)

print(
    "\nAko želiš nastaviti:"
)

print(
    "samo ponovno pokreni isti program."
)

print(
    "Već spremljeni StudyInstanceUID-ovi "
    "će biti automatski preskočeni."
)