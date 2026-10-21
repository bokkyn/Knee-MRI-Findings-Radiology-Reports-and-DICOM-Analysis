

"""
Ispis GOLD studija u čitljivom formatu.
========================================

Čita:
  - labels_final.csv      (labele: GOLD, SILVER, ...)
  - labels_research.csv   (Report, jezik)

Filtrira:
  - SAMO studije koje imaju barem jednu GOLD labelu

Ispisuje:
  - gold_studies_readable.txt
"""

import os
import unicodedata
import pandas as pd





BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FINAL = os.path.join(BASE_DIR, "labels_final.csv")
INPUT_RESEARCH = os.path.join(BASE_DIR, "labels_research.csv")
OUTPUT_TXT = os.path.join(BASE_DIR, "gold_studies_readable.txt")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    return text.strip()






if not os.path.exists(INPUT_FINAL) or not os.path.exists(INPUT_RESEARCH):
    print("GREŠKA: trebaš labels_final.csv i labels_research.csv")
    raise SystemExit(1)

final_df = pd.read_csv(INPUT_FINAL, low_memory=False)
research_df = pd.read_csv(INPUT_RESEARCH, low_memory=False)


merge_cols = ["StudyInstanceUID", "Report", "detected_language_name"]
merge_cols = [c for c in merge_cols if c in research_df.columns]

df = final_df.merge(
    research_df[merge_cols],
    on="StudyInstanceUID",
    how="left",
)

print(f"Ukupno studija: {len(df):,}")






def has_gold(row):
    for t in TARGETS:
        status = str(row.get(f"{t}_status", "")).strip().upper()
        if status == "GOLD":
            return True
    return False


gold_df = df[df.apply(has_gold, axis=1)].copy()
print(f"GOLD studija:   {len(gold_df):,}")






gold_per_target = {t: 0 for t in TARGETS}
for _, row in gold_df.iterrows():
    for t in TARGETS:
        if str(row.get(f"{t}_status", "")).strip().upper() == "GOLD":
            gold_per_target[t] += 1

print("\nGOLD labela po targetu:")
for t in TARGETS:
    print(f"  {t:<20} {gold_per_target[t]:>3}")






with open(OUTPUT_TXT, "w", encoding="utf-8") as f:

    
    f.write("=" * 90 + "\n")
    f.write("RSNA KNEE - GOLD STUDIJE (ekspertne oznake)\n")
    f.write("=" * 90 + "\n\n")

    f.write("VAŽNO:\n")
    f.write("  - Ovo su studije koje imaju barem jednu GOLD (ekspertnu) labelu.\n")
    f.write("  - GOLD = prava ekspertna oznaka, ne parser, ne LLM.\n")
    f.write("  - Ovim labelama VJERUJEMO i one su prioritet nad svime.\n\n")

    f.write(f"Ukupno studija:       {len(df):,}\n")
    f.write(f"GOLD studija:         {len(gold_df):,}\n")
    f.write(f"Ukupno GOLD labela:   {sum(gold_per_target.values()):,}\n\n")

    f.write("-" * 90 + "\n")
    f.write("GOLD LABELE PO TARGETU\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Target':<20} {'GOLD labela':>12}\n")
    f.write("-" * 90 + "\n")
    for t in TARGETS:
        f.write(f"{t:<20} {gold_per_target[t]:>12}\n")
    f.write("\n")

    
    f.write("=" * 90 + "\n")
    f.write("DETALJI PO STUDIJU\n")
    f.write("=" * 90 + "\n")

    for idx, (_, row) in enumerate(gold_df.iterrows(), 1):
        uid = row["StudyInstanceUID"]
        lang = row.get("detected_language_name", "?")
        report = normalize_text(row.get("Report", ""))

        
        n_gold = 0
        for t in TARGETS:
            if str(row.get(f"{t}_status", "")).strip().upper() == "GOLD":
                n_gold += 1

        f.write("\n" + "#" * 90 + "\n")
        f.write(f"[{idx}/{len(gold_df)}] StudyInstanceUID: {uid}\n")
        f.write(f"Jezik: {lang}\n")
        f.write(f"Broj GOLD labela: {n_gold}\n")
        f.write("-" * 90 + "\n")
        f.write("REPORT\n")
        f.write("-" * 90 + "\n")
        f.write(report + "\n\n")
        f.write("-" * 90 + "\n")
        f.write("GOLD LABELE\n")
        f.write("-" * 90 + "\n")

        for t in TARGETS:
            status = str(row.get(f"{t}_status", "")).strip().upper()
            value = row.get(t)
            source = str(row.get(f"{t}_source", "")).strip()

            
            if status != "GOLD":
                continue

            if pd.isna(value):
                val_str = "?"
            else:
                val_str = str(int(value))

            f.write(f"  {t:<20} = {val_str}   (source: {source})\n")

        f.write("\n")






print(f"\nSpremljeno: {OUTPUT_TXT}")
print("\n" + "=" * 72)
print("GOTOVO")
print("=" * 72)