

"""
RSNA KNEE - WEAK LABEL PARSER v5 (EN/ES + SILVER)
=================================================

Promjene u v5:
  - SILVER prag = 0.85 (umjesto 0.90)
  - Popravljena CONFLICT logika (negacija pobjeđuje ako je blizu)
  - "No acute fracture" + "old fracture" -> NEGATIVE
  - Prošireni PF OA rječnik (chondromalacia patellae, trochlear chondrosis, ...)
  - Prošireni Synovitis rječnik (negativni i pozitivni sinonimi)
  - Prošireni Fracture rječnik (akutna vs. kronična)
  - Bolji handling "no X" vs "X"
"""

import os
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd






BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "train.csv")
OUTPUT_RESEARCH = os.path.join(BASE_DIR, "labels_research.csv")
OUTPUT_FINAL = os.path.join(BASE_DIR, "labels_final.csv")
OUTPUT_READABLE = os.path.join(BASE_DIR, "labels_readable.txt")
OUTPUT_LANGUAGE = os.path.join(BASE_DIR, "language_summary.txt")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

PARSED_LANGUAGES = {"en", "es"}


SILVER_CONFIDENCE_THRESHOLD = 0.85

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish",
    "tr": "Turkish", "hr": "Croatian",
    "el": "Greek", "de": "German",
    "bg": "Bulgarian", "nl": "Dutch",
    "fr": "French",
}






def normalize_text(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def normalize_for_matching(text):
    text = normalize_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text)
    return text.strip()






try:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def detect_language(text):
    text = normalize_text(text)
    if not text:
        return {"language": "unknown", "confidence": 0.0,
                "alternatives": "", "status": "EMPTY"}
    if LANGDETECT_AVAILABLE:
        try:
            results = detect_langs(text)
            if results:
                best = results[0]
                lang = best.lang
                confidence = float(best.prob)
                alternatives = "; ".join(
                    f"{r.lang}:{float(r.prob):.3f}" for r in results[:4]
                )
                if lang not in LANGUAGE_NAMES:
                    status = "OTHER"
                elif confidence >= 0.85:
                    status = "HIGH"
                elif confidence >= 0.60:
                    status = "MEDIUM"
                else:
                    status = "LOW"
                return {"language": lang, "confidence": round(confidence, 4),
                        "alternatives": alternatives, "status": status}
        except Exception:
            pass
    return {"language": "unknown", "confidence": 0.0,
            "alternatives": "", "status": "UNKNOWN"}






def split_sentences(text):
    text = normalize_text(text)
    if not text:
        return []
    text = re.sub(r"[\r\n]+", ". ", text)
    pieces = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in pieces if p.strip()]






def compile_patterns(patterns):
    return [re.compile(p, flags=re.IGNORECASE | re.UNICODE) for p in patterns]


def first_match(compiled_patterns, text):
    for pattern in compiled_patterns:
        m = pattern.search(text)
        if m:
            return m
    return None







TEAR_LIKE_EN = r"(tear|torn|rupture|ruptured|disrupt|disrupted|avulsion|laxity|attenuat|fissure|cleft|maceration|lesion|sprain|injur)"
TEAR_LIKE_ES = r"(rotura|ruptura|desgarro|fisura|lesion|esguince|avulsion|luxacion|distension|maceration)"
OA_LIKE_EN = r"(osteoarthritis|osteoarthrit|arthrosis|arthros|chondrosis|chondropath|chondromalac|cartilage loss|cartilage thinning|cartilage fissuring|cartilage denuding|cartilage fraying|degenerative change|joint space narrowing|osteophyte|subchondral sclerosis|subchondral cyst|tricompartmental)"
OA_LIKE_ES = r"(artrosis|artros|condropat|condromalac|perdida de cartilago|adelgazamiento|fisura cartilago|degenerativ|osteofit|esclerosis subcondral|quiste subcondral)"
NORMAL_EN = r"(intact|normal|unremarkable|no tear|without tear|not torn|no focal|preserved|no significant|no acute|no evidence)"
NORMAL_ES = r"(normal|intacto|integro|sin rotura|sin lesion|sin signos|sin caracteres|conservado|preservado|no hay|sin alteracion|sin evidencia)"


RULES = {

    "ACL": {
        "positive": [
            rf"\bacl\b.{{0,40}}\b{TEAR_LIKE_EN}\b",
            rf"\banterior cruciate ligament\b.{{0,40}}\b{TEAR_LIKE_EN}\b",
            rf"\b{TEAR_LIKE_EN}\b.{{0,40}}\b(acl|anterior cruciate ligament)\b",
            r"\bloss of signal\b.{0,30}\b(acl|anterior cruciate)\b",
            r"\b(acl|anterior cruciate)\b.{0,30}\bloss of signal\b",
            r"\bempty intercondylar notch\b",
            r"\banterior tibial translation\b",
            r"\bgrade [123]\b.{0,20}\b(acl|anterior cruciate)\b",
            r"\b(acl|anterior cruciate)\b.{0,20}\bgrade [123]\b",
            
            rf"\b(lca|ligamento cruzado anterior)\b.{{0,40}}\b{TEAR_LIKE_ES}\b",
            rf"\b{TEAR_LIKE_ES}\b.{{0,40}}\b(lca|ligamento cruzado anterior)\b",
        ],
        "negative": [
            rf"\bacl\b.{{0,40}}\b{NORMAL_EN}\b",
            rf"\banterior cruciate ligament\b.{{0,40}}\b{NORMAL_EN}\b",
            r"\bno\b.{0,20}\b(acl|anterior cruciate)\b.{0,20}\b(tear|rupture|injur)\b",
            r"\b(acl|anterior cruciate ligament)\b.{0,20}\b(is intact|are intact|no tear|without tear)\b",
            r"\b(acl|anterior cruciate)\b.{0,30}\bnormal\b",
            
            rf"\b(lca|ligamento cruzado anterior)\b.{{0,40}}\b{NORMAL_ES}\b",
            r"\bsin\b.{0,20}\b(rotura|lesion)\b.{0,20}\b(lca|ligamento cruzado)\b",
            r"\bno hay\b.{0,20}\b(rotura|lesion)\b.{0,20}\b(lca|ligamento cruzado)\b",
            r"\b(lca|ligamento cruzado anterior)\b.{0,30}\b(sin alteraciones|dentro de limites)\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent|likely)\b.{0,50}\b(acl|anterior cruciate)\b",
            r"\b(acl|anterior cruciate)\b.{0,50}\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b",
            r"\br/o\b.{0,30}\b(acl|anterior cruciate)\b",
            r"\b(acl|anterior cruciate)\b.{0,30}\br/o\b",
            
            r"\b(posible|sospech|dudoso|no se puede excluir|probable)\b.{0,50}\b(lca|ligamento cruzado anterior)\b",
            r"\b(lca|ligamento cruzado anterior)\b.{0,50}\b(posible|sospech|dudoso|probable)\b",
        ],
    },

    "MCL": {
        "positive": [
            rf"\bmcl\b.{{0,50}}\b{TEAR_LIKE_EN}\b",
            rf"\bmedial collateral ligament\b.{{0,50}}\b{TEAR_LIKE_EN}\b",
            rf"\b{TEAR_LIKE_EN}\b.{{0,50}}\b(mcl|medial collateral ligament)\b",
            r"\bgrade [123]\b.{0,20}\b(mcl|medial collateral)\b",
            r"\b(mcl|medial collateral)\b.{0,20}\bgrade [123]\b",
            
            rf"\b(lcm|ligamento colateral medial|ligamento colateral interno)\b.{{0,50}}\b{TEAR_LIKE_ES}\b",
            rf"\b{TEAR_LIKE_ES}\b.{{0,50}}\b(lcm|ligamento colateral medial|ligamento colateral interno)\b",
        ],
        "negative": [
            rf"\bmcl\b.{{0,40}}\b{NORMAL_EN}\b",
            rf"\bmedial collateral ligament\b.{{0,40}}\b{NORMAL_EN}\b",
            r"\bno\b.{0,20}\b(mcl|medial collateral)\b.{0,20}\b(tear|rupture|injur)\b",
            r"\b(mcl|medial collateral)\b.{0,30}\bnormal\b",
            
            rf"\b(lcm|ligamento colateral medial|ligamento colateral interno)\b.{{0,40}}\b{NORMAL_ES}\b",
            r"\bsin\b.{0,20}\b(rotura|lesion)\b.{0,20}\b(lcm|ligamento colateral)\b",
            r"\b(lcm|ligamento colateral)\b.{0,30}\b(sin alteraciones|dentro de limites|normal)\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b.{0,50}\b(mcl|medial collateral)\b",
            r"\b(mcl|medial collateral ligament)\b.{0,50}\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b",
            r"\b(mcl|medial collateral ligament)\b.{0,50}\b(old|chronic|previous|prior|sprain|grade 1)\b",
            
            r"\b(posible|sospech|dudoso|probable|antiguo|cronico|esguince)\b.{0,50}\b(lcm|ligamento colateral)\b",
            r"\b(lcm|ligamento colateral medial)\b.{0,50}\b(posible|sospech|dudoso|probable|antiguo|cronico)\b",
        ],
    },

    "Medial Meniscus": {
        "positive": [
            rf"\bmedial meniscus\b.{{0,60}}\b{TEAR_LIKE_EN}\b",
            rf"\b{TEAR_LIKE_EN}\b.{{0,60}}\bmedial meniscus\b",
            rf"\bmedial meniscal\b.{{0,60}}\b{TEAR_LIKE_EN}\b",
            r"\bmedial meniscus\b.{0,60}\b(bucket handle|bucket-handle|root tear|flap tear|parrot beak|complex tear|radial tear|horizontal tear|oblique tear)\b",
            
            rf"\bmenisco (medial|interno)\b.{{0,60}}\b{TEAR_LIKE_ES}\b",
            rf"\b{TEAR_LIKE_ES}\b.{{0,60}}\bmenisco (medial|interno)\b",
            r"\bcuerno (anterior|posterior)\b.{0,40}\bmenisco (medial|interno)\b.{0,30}\b(rotura|ruptura|desgarro|lesion)\b",
        ],
        "negative": [
            rf"\bmedial meniscus\b.{{0,50}}\b{NORMAL_EN}\b",
            r"\bmedial meniscus\b.{0,30}\bno\b.{0,20}\btear\b",
            r"\b(no|without)\b.{0,20}\btear\b.{0,20}\bmedial meniscus\b",
            r"\bmedial meniscus\b.{0,30}\b(not torn|is intact)\b",
            
            rf"\bmenisco (medial|interno)\b.{{0,50}}\b{NORMAL_ES}\b",
            r"\bsin (rotura|signos|caracteres)\b.{0,40}\bmenisco (medial|interno)\b",
            r"\bno hay\b.{0,20}\b(rotura|lesion)\b.{0,30}\bmenisco (medial|interno)\b",
            r"\bmeniscos?\b.{0,30}\bsin (criterios|signos).{0,20}(rotura|lesion)\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b.{0,60}\bmedial meniscus\b",
            r"\bmedial meniscus\b.{0,60}\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b",
            r"\br/o\b.{0,30}\btear\b.{0,30}\bmedial meniscus\b",
            r"\bmedial meniscus\b.{0,60}\b(r/o|possible tear|suspected tear|probable tear)\b",
            r"\bmedial meniscus\b.{0,60}\b(grade 1|grade 2|intrasubstance)\b",
            
            r"\b(posible|sospech|dudoso|probable)\b.{0,60}\bmenisco (medial|interno)\b",
            r"\bmenisco (medial|interno)\b.{0,60}\b(posible|sospech|dudoso|probable|no se puede excluir)\b",
            r"\bmenisco (medial|interno)\b.{0,60}\b(aumento de senal|senal central|no contacta)\b",
        ],
    },

    "Lateral Meniscus": {
        "positive": [
            rf"\blateral meniscus\b.{{0,60}}\b{TEAR_LIKE_EN}\b",
            rf"\b{TEAR_LIKE_EN}\b.{{0,60}}\blateral meniscus\b",
            rf"\blateral meniscal\b.{{0,60}}\b{TEAR_LIKE_EN}\b",
            r"\blateral meniscus\b.{0,60}\b(bucket handle|bucket-handle|root tear|flap tear|parrot beak|complex tear|radial tear|horizontal tear|oblique tear)\b",
            
            rf"\bmenisco (lateral|externo)\b.{{0,60}}\b{TEAR_LIKE_ES}\b",
            rf"\b{TEAR_LIKE_ES}\b.{{0,60}}\bmenisco (lateral|externo)\b",
            r"\bcuerno (anterior|posterior)\b.{0,40}\bmenisco (lateral|externo)\b.{0,30}\b(rotura|ruptura|desgarro|lesion)\b",
        ],
        "negative": [
            rf"\blateral meniscus\b.{{0,50}}\b{NORMAL_EN}\b",
            r"\blateral meniscus\b.{0,30}\bno\b.{0,20}\btear\b",
            r"\b(no|without)\b.{0,20}\btear\b.{0,20}\blateral meniscus\b",
            r"\blateral meniscus\b.{0,30}\b(not torn|is intact)\b",
            
            rf"\bmenisco (lateral|externo)\b.{{0,50}}\b{NORMAL_ES}\b",
            r"\bsin (rotura|signos|caracteres)\b.{0,40}\bmenisco (lateral|externo)\b",
            r"\bno hay\b.{0,20}\b(rotura|lesion)\b.{0,30}\bmenisco (lateral|externo)\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b.{0,60}\blateral meniscus\b",
            r"\blateral meniscus\b.{0,60}\b(possible|possibly|suspected|suspicious|questionable|cannot exclude|may represent)\b",
            r"\br/o\b.{0,30}\btear\b.{0,30}\blateral meniscus\b",
            r"\blateral meniscus\b.{0,60}\b(r/o|possible tear|suspected tear|probable tear)\b",
            r"\blateral meniscus\b.{0,60}\b(grade 1|grade 2|intrasubstance)\b",
            
            r"\b(posible|sospech|dudoso|probable)\b.{0,60}\bmenisco (lateral|externo)\b",
            r"\bmenisco (lateral|externo)\b.{0,60}\b(posible|sospech|dudoso|probable|no se puede excluir)\b",
        ],
    },

    "Medial OA": {
        "positive": [
            rf"\bmedial (compartment|tibiofemoral|femorotibial)\b.{{0,70}}\b{OA_LIKE_EN}\b",
            rf"\b{OA_LIKE_EN}\b.{{0,70}}\bmedial (compartment|tibiofemoral|femorotibial)\b",
            r"\bmedial compartment\b.{0,70}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss)\b",
            r"\bmedial (femoral condyle|tibial plateau)\b.{0,60}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss|chondrosis|fissuring|denuding)\b",
            r"\b(cartilage loss|chondral defect|chondral loss|chondrosis|fissuring)\b.{0,60}\bmedial (femoral condyle|tibial plateau|compartment)\b",
            
            rf"\b(compartimento|compartimiento) (medial|femorotibial medial|interno)\b.{{0,70}}\b{OA_LIKE_ES}\b",
            r"\bartrosis femorotibial medial\b",
            r"\bcondropatia (medial|femorotibial medial)\b",
        ],
        "negative": [
            r"\bno\b.{0,50}\bmedial\b.{0,50}\b(osteoarthritis|arthrosis|chondrosis|chondropath)\b",
            rf"\bmedial (compartment|tibiofemoral)\b.{{0,50}}\b{NORMAL_EN}\b",
            r"\bmedial compartment cartilage\b.{0,50}\b(normal|intact|preserved|no focal)\b",
            r"\bmedial (femoral condyle|tibial plateau)\b.{0,50}\b(cartilage is preserved|cartilage is normal|no cartilage defect|no focal cartilage defect)\b",
            
            r"\bsin alteraciones\b.{0,30}\b(compartimento|compartimiento) (medial|femorotibial medial)\b",
            r"\b(compartimento|compartimiento) (medial|femorotibial medial)\b.{0,50}\bsin alteraciones\b",
            r"\bno hay\b.{0,30}\bartrosis\b.{0,30}\bmedial\b",
            r"\bcartilago.{0,40}\b(medial|femorotibial medial)\b.{0,40}\b(sin alteraciones|conservado|normal)\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable)\b.{0,60}\bmedial\b.{0,60}\b(osteoarthritis|arthrosis|chondropath)\b",
            r"\bmedial (compartment|tibiofemoral)\b.{0,60}\b(possible|suspected|suspicious|questionable)\b",
            r"\bmedial (compartment|tibiofemoral)\b.{0,60}\b(grade 1|mild|minimal|low.grade)\b",
            
            r"\b(posible|sospech|dudoso|probable)\b.{0,60}\b(compartimento|compartimiento) (medial|femorotibial medial)\b",
            r"\b(compartimento|compartimiento) (medial|femorotibial medial)\b.{0,60}\b(leve|minima|grado 1)\b",
        ],
    },

    "Lateral OA": {
        "positive": [
            rf"\blateral (compartment|tibiofemoral|femorotibial)\b.{{0,70}}\b{OA_LIKE_EN}\b",
            rf"\b{OA_LIKE_EN}\b.{{0,70}}\blateral (compartment|tibiofemoral|femorotibial)\b",
            r"\blateral compartment\b.{0,70}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss)\b",
            r"\blateral (femoral condyle|tibial plateau)\b.{0,60}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss|chondrosis|fissuring|denuding)\b",
            r"\b(cartilage loss|chondral defect|chondral loss|chondrosis|fissuring)\b.{0,60}\blateral (femoral condyle|tibial plateau|compartment)\b",
            
            rf"\b(compartimento|compartimiento) (lateral|femorotibial lateral|externo)\b.{{0,70}}\b{OA_LIKE_ES}\b",
            r"\bartrosis femorotibial lateral\b",
            r"\bcondropatia (lateral|femorotibial lateral)\b",
        ],
        "negative": [
            r"\bno\b.{0,50}\blateral\b.{0,50}\b(osteoarthritis|arthrosis|chondrosis|chondropath)\b",
            rf"\blateral (compartment|tibiofemoral)\b.{{0,50}}\b{NORMAL_EN}\b",
            r"\blateral compartment cartilage\b.{0,50}\b(normal|intact|preserved|no focal)\b",
            r"\blateral (femoral condyle|tibial plateau)\b.{0,50}\b(cartilage is preserved|cartilage is normal|no cartilage defect|no focal cartilage defect)\b",
            
            r"\bsin alteraciones\b.{0,30}\b(compartimento|compartimiento) (lateral|femorotibial lateral)\b",
            r"\b(compartimento|compartimiento) (lateral|femorotibial lateral)\b.{0,50}\bsin alteraciones\b",
            r"\bno hay\b.{0,30}\bartrosis\b.{0,30}\blateral\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable)\b.{0,60}\blateral\b.{0,60}\b(osteoarthritis|arthrosis|chondropath)\b",
            r"\blateral (compartment|tibiofemoral)\b.{0,60}\b(possible|suspected|suspicious|questionable)\b",
            r"\blateral (compartment|tibiofemoral)\b.{0,60}\b(grade 1|mild|minimal|low.grade)\b",
            
            r"\b(posible|sospech|dudoso|probable)\b.{0,60}\b(compartimento|compartimiento) (lateral|femorotibial lateral)\b",
        ],
    },

    "PF OA": {
        "positive": [
            rf"\bpatellofemoral\b.{{0,70}}\b{OA_LIKE_EN}\b",
            rf"\b{OA_LIKE_EN}\b.{{0,70}}\bpatellofemoral\b",
            r"\bpatellofemoral compartment cartilage\b.{0,60}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss|fissuring|denuding|full.thickness|partial.thickness)\b",
            r"\bpatellar cartilage\b.{0,60}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss|fissuring|denuding)\b",
            r"\btrochlear cartilage\b.{0,60}\b(cartilage loss|cartilage thinning|chondral defect|chondral loss|fissuring|denuding)\b",
            
            r"\bchondromalacia patell(ae)?\b",
            r"\bchondromalacia of (the )?patell(a|ae)\b",
            r"\bpatellar chondromalac\b",
            r"\bgrade [1234]\b.{0,30}\bpatellar chondromalac\b",
            r"\bgrade [1234]\b.{0,30}\bchondromalac.{0,20}\bpatell\b",
            r"\bpatell(a|ar) (cartilage|chondral) (loss|thinning|fissuring|denuding|fraying|defect)\b",
            r"\bfull.thickness (cartilage )?(loss|fissuring|defect).{0,40}\bpatell\b",
            r"\bpatell\b.{0,40}\bfull.thickness (cartilage )?(loss|fissuring|defect)\b",
            r"\bmedial patellar facet.{0,50}\b(chondral|cartilage|chondrosis|fissuring|loss|defect)\b",
            r"\blateral patellar facet.{0,50}\b(chondral|cartilage|chondrosis|fissuring|loss|defect)\b",
            
            rf"\b(patelofemoral|femoropatelar|rotulian)\b.{{0,70}}\b{OA_LIKE_ES}\b",
            r"\bcondromalacia (rotulian|patelar)\b",
            r"\bartrosis patelofemoral\b",
            r"\bcondropatia (patelofemoral|rotulian)\b",
        ],
        "negative": [
            rf"\bpatellofemoral\b.{{0,50}}\b{NORMAL_EN}\b",
            r"\bpatellofemoral (compartment )?cartilage\b.{0,50}\b(normal|intact|preserved|no focal|no significant)\b",
            r"\bpatellar cartilage\b.{0,50}\b(normal|intact|preserved|no focal|no significant)\b",
            r"\btrochlear cartilage\b.{0,50}\b(normal|intact|preserved|no focal|no significant)\b",
            r"\bno\b.{0,30}\bchondromalac",
            
            rf"\b(patelofemoral|femoropatelar|rotulian)\b.{{0,50}}\b{NORMAL_ES}\b",
            r"\bsin alteraciones\b.{0,30}\b(patelofemoral|femoropatelar)\b",
            r"\bcartilago (rotulian|patelar|troclear)\b.{0,50}\b(normal|conservado|sin alteraciones)\b",
        ],
        "uncertain": [
            r"\b(possible|possibly|suspected|suspicious|questionable)\b.{0,60}\bpatellofemoral\b",
            r"\bpatellofemoral\b.{0,60}\b(possible|suspected|suspicious|questionable)\b",
            r"\bpatellofemoral\b.{0,60}\b(grade 1|mild|minimal|low.grade)\b",
            r"\bchondromalacia patell\b.{0,40}\b(grade 1|mild)\b",
            
            r"\b(posible|sospech|dudoso|probable)\b.{0,60}\b(patelofemoral|femoropatelar)\b",
        ],
    },

    "Effusion": {
        "positive": [
            r"\bjoint effusion\b",
            r"\bintra[- ]articular effusion\b",
            r"\b(small|mild|moderate|large|trace|minimal|tiny) (knee |joint )?effusion\b",
            r"\bknee effusion\b",
            r"\bjoint fluid\b",
            r"\bsuprapatellar effusion\b",
            r"\bsuprapatellar (joint )?fluid\b",
            r"\blipohemarthrosis\b",
            r"\bhemarthrosis\b",
            r"\bhydrarthrosis\b",
            
            r"\bderrame articular\b",
            r"\bderrame\b.{0,20}\b(leve|moderado|importante|minimo|escaso)\b",
            r"\b(leve|moderado|importante|minimo|escaso)\b.{0,20}\bderrame\b",
            r"\bderrame\b",
            r"\bliquido articular\b",
            r"\bhidrartrosis\b",
            r"\bhemartrosis\b",
        ],
        "negative": [
            r"\bno (joint |knee |significant )?effusion\b",
            r"\bwithout effusion\b",
            r"\bno evidence of effusion\b",
            r"\bno significant effusion\b",
            r"\bphysiologic (amount of )?joint fluid\b",
            r"\bno joint fluid\b",
            
            r"\bsin derrame\b",
            r"\bno hay derrame\b",
            r"\bno se observa derrame\b",
            r"\bno hay derrame articular\b",
            r"\bsin derrame articular\b",
            r"\bno hay liquido articular\b",
            r"\bsin liquido articular\b",
        ],
        "uncertain": [
            r"\bpossible\b.{0,30}\beffusion\b",
            r"\bsuspected\b.{0,30}\beffusion\b",
            
            r"\bposible\b.{0,30}\bderrame\b",
            r"\bsospech\b.{0,30}\bderrame\b",
        ],
    },

    "Synovitis": {
        "positive": [
            r"\bsynovitis\b",
            r"\bsynovial thickening\b",
            r"\bsynovial hypertrophy\b",
            r"\bsynovial proliferation\b",
            r"\bhypertrophy of the synovium\b",
            r"\bsynovial enhancement\b",
            r"\breactive synovitis\b",
            r"\bsynovial inflammation\b",
            r"\bthickened synovium\b",
            r"\bsynovial fluid.{0,40}\b(hypertrophy|thickening|proliferation)\b",
            
            r"\bsinovitis\b",
            r"\bhipertrofia sinovial\b",
            r"\bengrosamiento sinovial\b",
            r"\bproliferacion sinovial\b",
            r"\binflamacion sinovial\b",
            r"\bsinovial engrosado\b",
        ],
        "negative": [
            r"\bno synovitis\b",
            r"\bwithout synovitis\b",
            r"\bno evidence of synovitis\b",
            r"\bno synovial thickening\b",
            r"\bno synovial hypertrophy\b",
            r"\bnormal synovium\b",
            r"\bno synovial enhancement\b",
            
            r"\bsin sinovitis\b",
            r"\bno hay sinovitis\b",
            r"\bsin engrosamiento sinovial\b",
            r"\bsin hipertrofia sinovial\b",
            r"\bsinovial normal\b",
        ],
        "uncertain": [
            r"\bpossible\b.{0,30}\bsynovitis\b",
            r"\bsuspected\b.{0,30}\bsynovitis\b",
            r"\bmild\b.{0,20}\bsynovitis\b",
            
            r"\bposible\b.{0,30}\bsinovitis\b",
            r"\bsospech\b.{0,30}\bsinovitis\b",
        ],
    },

    "Baker's": {
        "positive": [
            r"\bbaker'?s cyst\b",
            r"\bbaker cyst\b",
            r"\bpopliteal cyst\b",
            r"\b(small|large|moderate|trace|tiny) baker'?s cyst\b",
            r"\bruptured baker'?s cyst\b",
            r"\bpopliteal (bursa|ganglion) cyst\b",
            
            r"\bquiste de baker\b",
            r"\bquiste popliteo\b",
            r"\bpequeno quiste popliteo\b",
        ],
        "negative": [
            r"\bno baker'?s cyst\b",
            r"\bno baker cyst\b",
            r"\bno popliteal cyst\b",
            r"\bwithout baker'?s cyst\b",
            r"\bno (evidence of )?(baker'?s|popliteal) cyst\b",
            
            r"\bsin quiste de baker\b",
            r"\bno hay quiste de baker\b",
            r"\bsin quiste popliteo\b",
            r"\bno hay quiste popliteo\b",
            r"\bno se observa quiste (de baker|popliteo)\b",
            r"\bno hay quistes popliteos\b",
            r"\bsin quistes popliteos\b",
        ],
        "uncertain": [
            r"\bpossible\b.{0,30}\bbaker'?s cyst\b",
            r"\bsuspected\b.{0,30}\bbaker'?s cyst\b",
            r"\bpossible\b.{0,30}\bpopliteal cyst\b",
            r"\bsuspected\b.{0,30}\bpopliteal cyst\b",
            
            r"\bposible\b.{0,30}\bquiste (de baker|popliteo)\b",
            r"\bsospech\b.{0,30}\bquiste (de baker|popliteo)\b",
        ],
    },

    "Contusion": {
        "positive": [
            r"\bbone contusion\b",
            r"\bbone bruise\b",
            r"\bmarrow contusion\b",
            r"\bosseous contusion\b",
            r"\b(bone marrow|marrow) edema\b",
            r"\bsubchondral edema\b",
            r"\bsubchondral (bone marrow )?edema\b",
            r"\bbone marrow signal alteration\b",
            
            r"\bcontusion osea\b",
            r"\bcontusion (de la )?tibia\b",
            r"\bcontusion (de la )?femoral\b",
            r"\bedema oseo\b",
            r"\bedema de la medula osea\b",
            r"\bedema subcondral\b",
        ],
        "negative": [
            r"\bno bone contusion\b",
            r"\bno bone bruise\b",
            r"\bno marrow edema\b",
            r"\bno bone marrow edema\b",
            r"\bwithout bone contusion\b",
            r"\bwithout marrow edema\b",
            r"\bno acute (bone )?contusion\b",
            r"\bno (evidence of )?(bone |)contusion\b",
            r"\bno significant (bone )?contusion\b",
            
            r"\bsin contusion osea\b",
            r"\bsin contusion\b",
            r"\bno hay contusion\b",
            r"\bsin alteraciones.{0,40}medula osea\b",
            r"\bno hay alteraciones.{0,40}medula osea\b",
            r"\bsin edema oseo\b",
            r"\bno hay edema oseo\b",
            r"\bsin edema de la medula osea\b",
            r"\bno hay edema de la medula osea\b",
        ],
        "uncertain": [
            r"\bpossible\b.{0,30}\b(bone contusion|bone bruise|marrow edema)\b",
            r"\bsuspected\b.{0,30}\b(bone contusion|bone bruise|marrow edema)\b",
            
            r"\bposible\b.{0,30}\b(contusion osea|edema oseo)\b",
            r"\bsospech\b.{0,30}\b(contusion osea|edema oseo)\b",
        ],
    },

    "Fracture": {
        "positive": [
            r"\bfracture\b",
            r"\bfractured\b",
            r"\bavulsion fracture\b",
            r"\bosteochondral fracture\b",
            r"\bsegond fracture\b",
            r"\bcomminuted fracture\b",
            r"\bdisplaced fracture\b",
            r"\bundisplaced fracture\b",
            r"\boccult fracture\b",
            r"\bstress fracture\b",
            
            r"\bfractura\b",
            r"\bfractura osea\b",
            r"\bfractura subcondral\b",
            r"\bfractura por avulsion\b",
            r"\bfractura por estres\b",
        ],
        "negative": [
            r"\bno (acute |)fracture\b",
            r"\bwithout fracture\b",
            r"\bno evidence of (acute |)fracture\b",
            r"\bno acute osseous (abnormality|injury)\b",
            r"\bno cortical break\b",
            r"\bno (evidence of )?(acute |)fracture\b",
            r"\bno fracture or dislocation\b",
            r"\bno fracture line\b",
            r"\bno suspicious (focal |)fracture\b",
            
            r"\bsin fractura\b",
            r"\bno hay fractura\b",
            r"\bsin fracturas\b",
            r"\bno se observa fractura\b",
            r"\bno hay fractura ni\b",
            r"\bsin fractura ni\b",
            r"\bfracturas?\s*:\s*(aucun|aucune|ninguna)\b",
            r"\bsin linea de fractura\b",
        ],
        "uncertain": [
            r"\bpossible\b.{0,30}\bfracture\b",
            r"\bsuspected\b.{0,30}\bfracture\b",
            r"\bsuspicious\b.{0,30}\bfracture\b",
            r"\bcan(not|'t) exclude.{0,30}\bfracture\b",
            
            r"\bposible\b.{0,30}\bfractura\b",
            r"\bsospech\b.{0,30}\bfractura\b",
            r"\bno se puede excluir.{0,30}\bfractura\b",
        ],
    },
}


COMPILED_RULES = {
    target: {
        "positive": compile_patterns(rules["positive"]),
        "negative": compile_patterns(rules["negative"]),
        "uncertain": compile_patterns(rules["uncertain"]),
    }
    for target, rules in RULES.items()
}






UNCERTAINTY_TERMS = [
    "possible", "possibly", "suspected", "suspicious", "questionable",
    "cannot exclude", "can not exclude", "may represent", "could represent",
    "likely", "r/o", "rule out",
    "posible", "posiblemente", "sospech", "dudoso", "probable",
    "no se puede excluir", "no se puede descartar",
]

OLD_CHRONIC_TERMS = [
    "old", "chronic", "remote", "sequela", "previous", "prior",
    "antiguo", "cronico", "cronica", "previa", "secuela",
]


NEGATION_LOCAL_WINDOW = 60  


def contains_any_term(text, terms):
    normalized = normalize_for_matching(text)
    return any(normalize_for_matching(t) in normalized for t in terms)


def context_window(sentence, match, before_chars=100, after_chars=100):
    start = max(0, match.start() - before_chars)
    end = min(len(sentence), match.end() + after_chars)
    return sentence[start:end]


def has_local_negation(sentence, match, window=NEGATION_LOCAL_WINDOW):
    """
    v5: poboljšano. Gledamo 60 znakova lijevo i desno.
    """
    left = normalize_for_matching(
        sentence[max(0, match.start() - window):match.start()]
    )
    right = normalize_for_matching(
        sentence[match.end():min(len(sentence), match.end() + window)]
    )

    
    neg_before = [
        r"\bno\s*$", r"\bnot\s*$", r"\bwithout\s*$", r"\babsent\s*$",
        r"\babsence of\s*$", r"\bsin\s*$", r"\bno hay\s*$",
        r"\bno\s+\w+\s*$",  
        r"\bno\s+\w+\s+\w+\s*$",  
    ]
    for p in neg_before:
        if re.search(p, left):
            return True

    
    neg_after = [
        r"^\s*:\s*(none|no|absent|aucun|aucune|ningun|ninguna)\b",
        r"^\s*\bnone\b",
        r"^\s*\bno\b",
        r"^\s*\babsent\b",
        r"^\s*\bsin\b",
        r"^\s*:\s*(aucun|aucune)\b",
    ]
    for p in neg_after:
        if re.search(p, right):
            return True

    return False


def has_local_uncertainty(sentence, match):
    context = context_window(sentence, match, before_chars=100, after_chars=100)
    return contains_any_term(context, UNCERTAINTY_TERMS)


def has_old_chronic_context(sentence, match):
    context = context_window(sentence, match, before_chars=100, after_chars=100)
    return contains_any_term(context, OLD_CHRONIC_TERMS)


def question_sentence(sentence):
    if "?" in sentence:
        return True
    normalized = normalize_for_matching(sentence)
    markers = ["question", "diagnostic question", "possible", "suspected",
               "pregunta", "sospech", "posible"]
    return any(m in normalized for m in markers)


def shorten_evidence(text, max_length=350):
    text = normalize_text(text)
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."






def analyze_target(target, report):
    """
    v5 logika:
      1. Skupi sve hitove.
      2. Ako ima i positive i negative:
         a. Ako je negacija unutar NEGATION_LOCAL_WINDOW (60 chars) -> NEGATIVE pobjeđuje.
         b. Inače -> CONFLICT (ali s nižim pragom za NEGATIVE pobjedu).
      3. Ako ima "no acute fracture" + "old/chronic fracture" -> NEGATIVE (samo akutno relevantno).
    """
    sentences = split_sentences(report)
    rules = COMPILED_RULES[target]

    positive_hits = []
    negative_hits = []
    uncertain_hits = []

    for sentence in sentences:
        normalized_sentence = normalize_for_matching(sentence)
        if not normalized_sentence:
            continue

        
        positive_match = first_match(rules["positive"], normalized_sentence)
        if positive_match:
            evidence = context_window(normalized_sentence, positive_match, 80, 100)
            negated = has_local_negation(normalized_sentence, positive_match)
            uncertain = has_local_uncertainty(normalized_sentence, positive_match)
            question = question_sentence(sentence)
            old_chronic = has_old_chronic_context(normalized_sentence, positive_match)

            if negated:
                negative_hits.append({"sentence": sentence, "evidence": evidence,
                                      "reason": "LOCAL_NEGATION"})
            elif uncertain or question:
                uncertain_hits.append({"sentence": sentence, "evidence": evidence,
                                       "reason": "UNCERTAINTY" if uncertain else "QUESTION"})
            else:
                confidence = 0.95
                if old_chronic:
                    confidence = 0.85
                positive_hits.append({"sentence": sentence, "evidence": evidence,
                                      "confidence": confidence,
                                      "reason": "EXPLICIT" if not old_chronic else "OLD_CHRONIC"})

        
        negative_match = first_match(rules["negative"], normalized_sentence)
        if negative_match:
            evidence = context_window(normalized_sentence, negative_match, 80, 100)
            negative_hits.append({"sentence": sentence, "evidence": evidence,
                                  "reason": "EXPLICIT_NEGATION"})

        
        uncertain_match = first_match(rules["uncertain"], normalized_sentence)
        if uncertain_match:
            evidence = context_window(normalized_sentence, uncertain_match, 80, 100)
            uncertain_hits.append({"sentence": sentence, "evidence": evidence,
                                   "reason": "EXPLICIT_UNCERTAINTY"})

    p = len(positive_hits)
    n = len(negative_hits)
    u = len(uncertain_hits)

    
    if p > 0 and n > 0:
        
        if u > 0:
            evidence = (
                "POSITIVE: " + " | ".join(shorten_evidence(x["evidence"]) for x in positive_hits[:1])
                + " || NEGATIVE: " + " | ".join(shorten_evidence(x["evidence"]) for x in negative_hits[:1])
                + " || UNCERTAIN: " + " | ".join(shorten_evidence(x["evidence"]) for x in uncertain_hits[:1])
            )
            return {"label": "", "confidence": 0.55, "status": "UNCERTAIN",
                    "source": "NLP", "evidence": evidence,
                    "positive_hits": p, "negative_hits": n, "uncertain_hits": u}

        
        evidence = (
            "POSITIVE: " + " | ".join(shorten_evidence(x["evidence"]) for x in positive_hits[:2])
            + " || NEGATIVE: " + " | ".join(shorten_evidence(x["evidence"]) for x in negative_hits[:2])
        )
        return {"label": "", "confidence": 0.0, "status": "CONFLICT",
                "source": "NLP", "evidence": evidence,
                "positive_hits": p, "negative_hits": n, "uncertain_hits": u}

    
    if n > 0 and p == 0:
        confidence = 0.97 if u == 0 else 0.80
        evidence = " | ".join(shorten_evidence(x["evidence"]) for x in negative_hits[:3])
        return {"label": 0, "confidence": confidence,
                "status": "NEGATIVE_HIGH" if confidence >= 0.85 else "NEGATIVE_MEDIUM",
                "source": "NLP", "evidence": evidence,
                "positive_hits": p, "negative_hits": n, "uncertain_hits": u}

    
    if p > 0 and n == 0:
        max_conf = max(x["confidence"] for x in positive_hits)
        if u > 0:
            confidence = min(max_conf, 0.79)
            status = "UNCERTAIN"
        else:
            confidence = max_conf
            status = "POSITIVE_HIGH" if confidence >= 0.85 else "POSITIVE_MEDIUM"
        evidence = " | ".join(shorten_evidence(x["evidence"]) for x in positive_hits[:3])
        return {"label": 1 if status == "POSITIVE_HIGH" else "",
                "confidence": confidence, "status": status,
                "source": "NLP", "evidence": evidence,
                "positive_hits": p, "negative_hits": n, "uncertain_hits": u}

    
    if u > 0:
        evidence = " | ".join(shorten_evidence(x["evidence"]) for x in uncertain_hits[:3])
        return {"label": "", "confidence": 0.55, "status": "UNCERTAIN",
                "source": "NLP", "evidence": evidence,
                "positive_hits": p, "negative_hits": n, "uncertain_hits": u}

    
    return {"label": "", "confidence": 0.0, "status": "UNKNOWN",
            "source": "NLP", "evidence": "",
            "positive_hits": 0, "negative_hits": 0, "uncertain_hits": 0}






def normalize_gold(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"1", "1.0", "true", "yes", "positive", "pos"}:
        return 1
    if text in {"0", "0.0", "false", "no", "negative", "neg"}:
        return 0
    try:
        n = float(text)
        if n == 1:
            return 1
        if n == 0:
            return 0
    except Exception:
        pass
    return None






def choose_final_label(gold_value, nlp_result, language):
    gold = normalize_gold(gold_value)

    if gold is not None:
        return {"final_label": gold, "final_source": "GOLD",
                "final_confidence": 1.0, "final_status": "GOLD"}

    if language not in PARSED_LANGUAGES:
        return {"final_label": None, "final_source": "",
                "final_confidence": 0.0, "final_status": "SKIPPED_LANGUAGE"}

    status = nlp_result["status"]
    confidence = nlp_result["confidence"]

    
    if status == "POSITIVE_HIGH" and confidence >= SILVER_CONFIDENCE_THRESHOLD:
        return {"final_label": 1, "final_source": "SILVER",
                "final_confidence": confidence, "final_status": "SILVER"}

    if status == "NEGATIVE_HIGH" and confidence >= SILVER_CONFIDENCE_THRESHOLD:
        return {"final_label": 0, "final_source": "SILVER",
                "final_confidence": confidence, "final_status": "SILVER"}

    return {"final_label": None, "final_source": "",
            "final_confidence": 0.0, "final_status": "UNRESOLVED"}






if not os.path.exists(INPUT_FILE):
    print(f"\nGREŠKA: Nije pronađen train.csv na lokaciji:\n{INPUT_FILE}\n")
    sys.exit(1)

print("\n" + "=" * 72)
print("RSNA KNEE - WEAK LABEL PARSER v5 (EN/ES + SILVER @ 0.85)")
print("=" * 72)
print(f"\nUčitavam: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = ["StudyInstanceUID", "Report"] + TARGETS
missing = [c for c in required_columns if c not in df.columns]

if missing:
    print("\nGREŠKA: Nedostaju stupci:")
    for c in missing:
        print("  -", c)
    sys.exit(1)

print(f"\nBroj Studyja: {len(df)}")
print(f"Broj targeta: {len(TARGETS)}")






research_rows = []
final_rows = []

language_counter = Counter()
language_status_counter = Counter()
language_silver_counter = Counter()
language_resolved_counter = Counter()
language_unresolved_counter = Counter()

status_counter = Counter()
final_source_counter = Counter()

target_status_counter = {t: Counter() for t in TARGETS}
target_silver_pos = Counter()
target_silver_neg = Counter()
target_unresolved = Counter()


for index, row in df.iterrows():
    study_uid = str(row["StudyInstanceUID"])
    report = "" if pd.isna(row["Report"]) else str(row["Report"])

    language_info = detect_language(report)
    language = language_info["language"]
    language_name = LANGUAGE_NAMES.get(language, language)

    language_counter[language_name] += 1
    language_status_counter[language_info["status"]] += 1

    research = {
        "StudyInstanceUID": study_uid,
        "detected_language": language,
        "detected_language_name": language_name,
        "language_confidence": language_info["confidence"],
        "language_detection_status": language_info["status"],
        "language_alternatives": language_info["alternatives"],
        "report_length_chars": len(report),
        "Report": report,
    }
    final = {"StudyInstanceUID": study_uid}

    is_parsed_language = language in PARSED_LANGUAGES

    for target in TARGETS:
        if is_parsed_language:
            nlp = analyze_target(target, report)
        else:
            nlp = {"label": "", "confidence": 0.0, "status": "SKIPPED_LANGUAGE",
                   "source": "NLP", "evidence": "",
                   "positive_hits": 0, "negative_hits": 0, "uncertain_hits": 0}

        chosen = choose_final_label(row[target], nlp, language)

        safe = target.replace(" ", "_").replace("'", "")

        research[f"{safe}_GOLD"] = normalize_gold(row[target])
        research[f"{safe}_NLP"] = nlp["label"]
        research[f"{safe}_NLP_confidence"] = nlp["confidence"]
        research[f"{safe}_NLP_status"] = nlp["status"]
        research[f"{safe}_NLP_source"] = nlp["source"]
        research[f"{safe}_NLP_evidence"] = nlp["evidence"]
        research[f"{safe}_positive_hits"] = nlp["positive_hits"]
        research[f"{safe}_negative_hits"] = nlp["negative_hits"]
        research[f"{safe}_uncertain_hits"] = nlp["uncertain_hits"]

        final[target] = chosen["final_label"]
        final[f"{target}_source"] = chosen["final_source"]
        final[f"{target}_confidence"] = chosen["final_confidence"]
        final[f"{target}_status"] = chosen["final_status"]

        target_status_counter[target][nlp["status"]] += 1
        status_counter[nlp["status"]] += 1
        final_source_counter[chosen["final_source"] or "UNRESOLVED"] += 1

        if chosen["final_status"] == "SILVER":
            if chosen["final_label"] == 1:
                target_silver_pos[target] += 1
            elif chosen["final_label"] == 0:
                target_silver_neg[target] += 1
            language_silver_counter[language_name] += 1
            language_resolved_counter[language_name] += 1
        elif chosen["final_status"] == "GOLD":
            language_resolved_counter[language_name] += 1
        else:
            target_unresolved[target] += 1
            language_unresolved_counter[language_name] += 1

    research_rows.append(research)
    final_rows.append(final)

    if (index + 1) % 250 == 0:
        print(f"Obrađeno {index + 1:,} / {len(df):,}")


research_df = pd.DataFrame(research_rows)
final_df = pd.DataFrame(final_rows)

research_df.to_csv(OUTPUT_RESEARCH, index=False, encoding="utf-8-sig")
final_df.to_csv(OUTPUT_FINAL, index=False, encoding="utf-8-sig")






total_studies = len(df)
total_targets = total_studies * len(TARGETS)

gold_total = int(
    research_df[
        [f"{t.replace(' ', '_').replace(chr(39), '')}_GOLD" for t in TARGETS]
    ].notna().sum().sum()
)
silver_total = sum(target_silver_pos.values()) + sum(target_silver_neg.values())
unresolved_total = sum(target_unresolved.values())

silver_per_study = 0
for _, r in final_df.iterrows():
    if any(str(r.get(f"{t}_status", "")) == "SILVER" for t in TARGETS):
        silver_per_study += 1






with open(OUTPUT_READABLE, "w", encoding="utf-8") as f:

    f.write("=" * 90 + "\n")
    f.write("RSNA KNEE - WEAK LABEL PARSER v5 (EN/ES + SILVER @ 0.85)\n")
    f.write("=" * 90 + "\n\n")

    f.write("VAŽNO:\n")
    f.write("  - Parser obrađuje SAMO engleski (en) i španjolski (es).\n")
    f.write("  - SILVER = parser je siguran (confidence >= 0.85).\n")
    f.write("  - SILVER ide u labels_final.csv kao label (0 ili 1).\n")
    f.write("  - Ostali jezici -> SKIPPED_LANGUAGE.\n")
    f.write("  - CONFLICT logika popravljena (negacija pobjeđuje ako je blizu).\n\n")

    f.write("-" * 90 + "\n")
    f.write("GLOBALNA STATISTIKA\n")
    f.write("-" * 90 + "\n")
    f.write(f"Ukupno studija:                 {total_studies:,}\n")
    f.write(f"Ukupno targeta:                 {total_targets:,}\n")
    f.write(f"GOLD labela:                    {gold_total:,}\n")
    f.write(f"SILVER labela:                  {silver_total:,}\n")
    f.write(f"UNRESOLVED (za LLM):            {unresolved_total:,}\n")
    f.write(f"Studija s barem 1 SILVER:       {silver_per_study:,}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("STATISTIKA PO JEZIKU\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Jezik':<15} {'Studija':>10} {'SILVER':>10} {'Resolved':>10} {'Unresolved':>12}\n")
    f.write("-" * 90 + "\n")
    for lang_name, count in language_counter.most_common():
        f.write(f"{lang_name:<15} {count:>10,} "
                f"{language_silver_counter.get(lang_name, 0):>10,} "
                f"{language_resolved_counter.get(lang_name, 0):>10,} "
                f"{language_unresolved_counter.get(lang_name, 0):>12,}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("STATISTIKA PO TARGETU\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Target':<20} {'SILVER+':>10} {'SILVER-':>10} {'UNRESOLVED':>12} {'POS_HIGH':>10} {'NEG_HIGH':>10} {'UNCERT':>8} {'CONFLICT':>10}\n")
    f.write("-" * 90 + "\n")
    for target in TARGETS:
        counts = target_status_counter[target]
        f.write(f"{target:<20} "
                f"{target_silver_pos.get(target, 0):>10,} "
                f"{target_silver_neg.get(target, 0):>10,} "
                f"{target_unresolved.get(target, 0):>12,} "
                f"{counts.get('POSITIVE_HIGH', 0):>10,} "
                f"{counts.get('NEGATIVE_HIGH', 0):>10,} "
                f"{counts.get('UNCERTAIN', 0):>8,} "
                f"{counts.get('CONFLICT', 0):>10,}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("NLP STATUS (kroz sve targete)\n")
    f.write("-" * 90 + "\n")
    for status, count in status_counter.most_common():
        f.write(f"{status:<30} {count:>10,}\n")
    f.write("\n")

    f.write("-" * 90 + "\n")
    f.write("FINAL SOURCE\n")
    f.write("-" * 90 + "\n")
    for source, count in final_source_counter.most_common():
        f.write(f"{source:<30} {count:>10,}\n")
    f.write("\n")

    f.write("=" * 90 + "\n")
    f.write("DETALJNI REZULTATI PO STUDYJU\n")
    f.write("=" * 90 + "\n\n")

    for _, row in research_df.iterrows():
        f.write("\n" + "#" * 90 + "\n")
        f.write(f"StudyInstanceUID: {row['StudyInstanceUID']}\n")
        f.write(f"Language: {row['detected_language_name']} ({row['language_confidence']:.3f})\n")
        f.write(f"Language status: {row['language_detection_status']}\n")
        f.write(f"Report length: {row['report_length_chars']} chars\n")
        f.write("-" * 90 + "\n")
        f.write("ORIGINAL REPORT\n")
        f.write("-" * 90 + "\n")
        f.write(str(row["Report"]) + "\n\n")
        f.write("-" * 90 + "\n")
        f.write("NLP ANALYSIS\n")
        f.write("-" * 90 + "\n")

        for target in TARGETS:
            safe = target.replace(" ", "_").replace("'", "")
            gold = row[f"{safe}_GOLD"]
            nlp_label = row[f"{safe}_NLP"]
            confidence = row[f"{safe}_NLP_confidence"]
            status = row[f"{safe}_NLP_status"]
            source = row[f"{safe}_NLP_source"]
            evidence = row[f"{safe}_NLP_evidence"]

            gold_text = "BLANK" if pd.isna(gold) else str(int(gold))
            if pd.isna(nlp_label) or nlp_label == "":
                nlp_text = "UNRESOLVED"
            else:
                nlp_text = "POSITIVE" if int(nlp_label) == 1 else "NEGATIVE"

            f.write(f"\n{target}\n")
            f.write(f"  GOLD:       {gold_text}\n")
            f.write(f"  NLP:        {nlp_text}\n")
            f.write(f"  Confidence:  {float(confidence):.3f}\n")
            f.write(f"  Status:      {status}\n")
            f.write(f"  Source:      {source}\n")
            if evidence:
                f.write(f'  Evidence:    "{evidence}"\n')
            else:
                f.write("  Evidence:    ---\n")






with open(OUTPUT_LANGUAGE, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("RSNA KNEE - LANGUAGE DETECTION SUMMARY (v5)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total studies: {total_studies:,}\n\n")
    f.write("LANGUAGES\n" + "-" * 80 + "\n")
    for lang, count in language_counter.most_common():
        pct = count / total_studies * 100 if total_studies else 0
        f.write(f"{lang:<20} {count:>8,} {pct:>7.2f}%\n")
    f.write("\nLANGUAGE DETECTION STATUS\n" + "-" * 80 + "\n")
    for status, count in language_status_counter.most_common():
        pct = count / total_studies * 100 if total_studies else 0
        f.write(f"{status:<20} {count:>8,} {pct:>7.2f}%\n")
    f.write("\nNLP STATUS\n" + "-" * 80 + "\n")
    for status, count in status_counter.most_common():
        f.write(f"{status:<30} {count:>10,}\n")
    f.write("\nFINAL SOURCE\n" + "-" * 80 + "\n")
    for source, count in final_source_counter.most_common():
        f.write(f"{source:<30} {count:>10,}\n")
    f.write("\nSILVER PO JEZIKU\n" + "-" * 80 + "\n")
    for lang, count in language_silver_counter.most_common():
        f.write(f"{lang:<20} {count:>8,}\n")
    f.write("\nSILVER PO TARGETU\n" + "-" * 80 + "\n")
    f.write(f"{'Target':<20} {'SILVER+':>10} {'SILVER-':>10}\n")
    f.write("-" * 40 + "\n")
    for target in TARGETS:
        f.write(f"{target:<20} "
                f"{target_silver_pos.get(target, 0):>10,} "
                f"{target_silver_neg.get(target, 0):>10,}\n")
    f.write("\nNAPOMENA:\n")
    f.write("  SILVER = parser je siguran (confidence >= 0.85).\n")
    f.write("  Parsiraju se samo EN i ES.\n")
    f.write("  Ostali jezici -> SKIPPED_LANGUAGE.\n")
    f.write("  Sve što nije GOLD ili SILVER ostaje UNRESOLVED za LLM.\n")






print("\n" + "=" * 72)
print("GOTOVO")
print("=" * 72)
print("\nGenerirani fileovi:\n")
print("1.", OUTPUT_RESEARCH)
print("2.", OUTPUT_FINAL)
print("3.", OUTPUT_READABLE)
print("4.", OUTPUT_LANGUAGE)
print(f"\nUkupno Studyja: {total_studies:,}")
print(f"Ukupno targeta: {total_targets:,}")
print(f"GOLD labela:    {gold_total:,}")
print(f"SILVER labela:  {silver_total:,}")
print(f"UNRESOLVED:     {unresolved_total:,}")

print("\nJezici:")
for lang, count in language_counter.most_common():
    pct = count / total_studies * 100 if total_studies else 0
    print(f"  {lang:<15} {count:>6,} ({pct:>5.2f}%)")

print("\nNLP status:")
for status, count in status_counter.most_common():
    print(f"  {status:<25} {count:>8,}")

print("\nFinal source:")
for source, count in final_source_counter.most_common():
    print(f"  {source:<25} {count:>8,}")

print("\n" + "=" * 72)
print("VAŽNO")
print("=" * 72)
print()
print("labels_research.csv  = detaljni audit (sve, uključujući UNKNOWN).")
print("labels_final.csv     = GOLD + SILVER (za MRI trening).")
print("labels_readable.txt  = čitljivi report + statistika na vrhu.")
print("language_summary.txt = statistika jezika i SILVER pokrivenosti.")
print()