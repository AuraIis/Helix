"""Rigorous fact-recall probe for Helix checkpoints — knowledge PROFILE edition.

Measures fact recall as a contrastive margin, but now disentangles two variables that
were previously mixed (knowledge quality vs. distractor hardness):

    margin = NLL(best distractor at hardness H) - NLL(correct)   (per-token mean)
    ok@H   = margin > 0  →  correct beats the strongest distractor at hardness H

Each fact carries distractors at up to three hardness levels:
    easy  = wrong domain / obviously wrong   (Au vs Berlin)
    med   = same domain, distinguishable     (Au vs Fe)
    hard  = same domain, very confusable      (Au vs Ag)

So you get accuracy per CATEGORY and per HARDNESS (e.g. easy 95% / med 72% / hard 35%),
not one mushed number. Also a top-k-after-prompt check (recall proximity).

IMPORTANT category note: "tech" = TECHNICAL CONCEPTS (print(), #, key-value...), which
leak into prose/tutorials. This run trained 0% code, so a high tech score is NOT a
coding benchmark — it is concept familiarity. Do not read it as "Helix can code".

Read-only. Run on the Blackwell (needs the kernels).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

if torch.cuda.is_available():
    try:
        import mamba_ssm  # noqa: F401
        os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
        os.environ.setdefault("AURALIS_USE_GLA_KERNEL", "1")
        os.environ.setdefault("AURALIS_USE_FLASH_ATTN", "1")
    except Exception:
        pass

from auralis.model import build_model  # noqa: E402

# (id, lang, category, prompt, correct, [(distractor, hardness), ...])
# hardness in {"easy","med","hard"}.  tech = CONCEPTS only (0% code trained this run).
FACT_PROBES = [
    # ---------- Geografie ----------
    ("de_cap_de", "de", "geo", "Die Hauptstadt von Deutschland ist", " Berlin.", [(" Hund.","easy"),(" Paris.","med"),(" München.","hard"),(" Hamburg.","hard")]),
    ("de_cap_fr", "de", "geo", "Die Hauptstadt von Frankreich ist", " Paris.", [(" Tisch.","easy"),(" Rom.","med"),(" Lyon.","hard"),(" Marseille.","hard")]),
    ("de_cap_it", "de", "geo", "Die Hauptstadt von Italien ist", " Rom.", [(" Auto.","easy"),(" Madrid.","med"),(" Mailand.","hard"),(" Neapel.","hard")]),
    ("de_cap_es", "de", "geo", "Die Hauptstadt von Spanien ist", " Madrid.", [(" Baum.","easy"),(" Berlin.","med"),(" Barcelona.","hard")]),
    ("de_cap_at", "de", "geo", "Die Hauptstadt von Österreich ist", " Wien.", [(" Wasser.","easy"),(" Berlin.","med"),(" Graz.","hard"),(" Salzburg.","hard")]),
    ("de_cap_uk", "de", "geo", "Die Hauptstadt des Vereinigten Königreichs ist", " London.", [(" Brot.","easy"),(" Berlin.","med"),(" Manchester.","hard")]),
    ("en_cap_de", "en", "geo", "The capital of Germany is", " Berlin.", [(" dog.","easy"),(" Paris.","med"),(" Munich.","hard"),(" Hamburg.","hard")]),
    ("en_cap_jp", "en", "geo", "The capital of Japan is", " Tokyo.", [(" table.","easy"),(" Berlin.","med"),(" Osaka.","hard"),(" Kyoto.","hard")]),
    ("en_cap_uk", "en", "geo", "The capital of the United Kingdom is", " London.", [(" apple.","easy"),(" Berlin.","med"),(" Manchester.","hard")]),
    ("en_cap_fr", "en", "geo", "The capital of France is", " Paris.", [(" car.","easy"),(" Tokyo.","med"),(" Lyon.","hard"),(" Marseille.","hard")]),
    ("en_cap_us", "en", "geo", "The capital of the United States is", " Washington.", [(" tree.","easy"),(" London.","med"),(" New York.","hard")]),
    ("en_cap_ru", "en", "geo", "The capital of Russia is", " Moscow.", [(" bread.","easy"),(" Paris.","med"),(" Saint Petersburg.","hard")]),
    # ---------- Wissenschaft ----------
    ("de_gold",   "de", "sci", "Das chemische Symbol für Gold ist", " Au.", [(" Berlin.","easy"),(" Fe.","med"),(" Ag.","hard")]),
    ("de_silver", "de", "sci", "Das chemische Symbol für Silber ist", " Ag.", [(" Haus.","easy"),(" Fe.","med"),(" Au.","hard")]),
    ("de_boil",   "de", "sci", "Wasser siedet bei Normaldruck bei", " 100 Grad Celsius.", [(" blau.","easy"),(" 0 Grad Celsius.","med"),(" 90 Grad Celsius.","hard")]),
    ("de_freeze", "de", "sci", "Wasser gefriert bei", " 0 Grad Celsius.", [(" Montag.","easy"),(" 100 Grad Celsius.","med"),(" 4 Grad Celsius.","hard")]),
    ("de_sun",    "de", "sci", "Die Sonne ist ein", " Stern.", [(" Tisch.","easy"),(" Komet.","med"),(" Planet.","hard")]),
    ("de_center", "de", "sci", "Der Mittelpunkt unseres Sonnensystems ist die", " Sonne.", [(" Tür.","easy"),(" Milchstraße.","med"),(" Erde.","hard")]),
    ("de_h2o",    "de", "sci", "Die chemische Formel von Wasser ist", " H2O.", [(" XY.","easy"),(" O2.","med"),(" CO2.","hard")]),
    ("en_gold",   "en", "sci", "The chemical symbol for gold is", " Au.", [(" dog.","easy"),(" Fe.","med"),(" Ag.","hard")]),
    ("en_jupiter","en", "sci", "The largest planet in our solar system is", " Jupiter.", [(" apple.","easy"),(" Earth.","med"),(" Saturn.","hard"),(" Mars.","hard")]),
    ("en_boil",   "en", "sci", "Water boils at normal pressure at", " 100 degrees Celsius.", [(" blue.","easy"),(" 0 degrees Celsius.","med"),(" 90 degrees Celsius.","hard")]),
    ("en_h2o",    "en", "sci", "The chemical formula for water is", " H2O.", [(" ZZ.","easy"),(" O2.","med"),(" CO2.","hard")]),
    ("en_sun",    "en", "sci", "The Sun is a", " star.", [(" table.","easy"),(" comet.","med"),(" planet.","hard")]),
    # ---------- Geschichte ----------
    ("de_faust",  "de", "hist", "Das Drama Faust wurde geschrieben von", " Goethe.", [(" Auto.","easy"),(" Brecht.","med"),(" Schiller.","hard")]),
    ("de_relativ","de", "hist", "Die Relativitätstheorie stammt von Albert", " Einstein.", [(" Banane.","easy"),(" Darwin.","med"),(" Newton.","hard")]),
    ("de_brd",    "de", "hist", "Die Bundesrepublik Deutschland wurde gegründet im Jahr", " 1949.", [(" Apfel.","easy"),(" 1789.","med"),(" 1939.","hard")]),
    ("de_mauer",  "de", "hist", "Die Berliner Mauer fiel im Jahr", " 1989.", [(" Hund.","easy"),(" 1789.","med"),(" 1979.","hard")]),
    ("de_ww2",    "de", "hist", "Der Zweite Weltkrieg endete im Jahr", " 1945.", [(" blau.","easy"),(" 1815.","med"),(" 1944.","hard")]),
    ("de_columbus","de","hist", "Amerika wurde im Jahr 1492 erreicht von", " Kolumbus.", [(" Tisch.","easy"),(" Goethe.","med"),(" Magellan.","hard")]),
    ("en_romeo",  "en", "hist", "Romeo and Juliet was written by", " Shakespeare.", [(" car.","easy"),(" Chaucer.","med"),(" Dickens.","hard")]),
    ("en_relativ","en", "hist", "The theory of relativity was developed by Albert", " Einstein.", [(" banana.","easy"),(" Darwin.","med"),(" Newton.","hard")]),
    ("en_moon",   "en", "hist", "The first moon landing happened in the year", " 1969.", [(" dog.","easy"),(" 1869.","med"),(" 1979.","hard")]),
    ("en_ww2",    "en", "hist", "World War II ended in the year", " 1945.", [(" blue.","easy"),(" 1815.","med"),(" 1944.","hard")]),
    ("en_indep",  "en", "hist", "The United States declared independence in the year", " 1776.", [(" apple.","easy"),(" 1492.","med"),(" 1789.","hard")]),
    ("en_gravity","en", "hist", "The law of gravitation is associated with Isaac", " Newton.", [(" table.","easy"),(" Darwin.","med"),(" Einstein.","hard")]),
    # ---------- Sprache / Übersetzung ----------
    ("de2en_dog", "de", "lang", "Das englische Wort für Hund ist", " dog.", [(" Tisch.","easy"),(" bird.","med"),(" cat.","hard")]),
    ("de2en_cat", "de", "lang", "Das englische Wort für Katze ist", " cat.", [(" blau.","easy"),(" fish.","med"),(" dog.","hard")]),
    ("de2en_house","de","lang", "Das englische Wort für Haus ist", " house.", [(" Montag.","easy"),(" car.","med"),(" home.","hard")]),
    ("de2en_water","de","lang", "Das englische Wort für Wasser ist", " water.", [(" Auto.","easy"),(" bread.","med"),(" fire.","hard")]),
    ("de2en_red", "de", "lang", "Das englische Wort für rot ist", " red.", [(" Stuhl.","easy"),(" green.","med"),(" blue.","hard")]),
    ("de2en_sun", "de", "lang", "Das englische Wort für Sonne ist", " sun.", [(" Apfel.","easy"),(" star.","med"),(" moon.","hard")]),
    ("en2de_water","en","lang", "The German word for water is", " Wasser.", [(" dog.","easy"),(" Brot.","med"),(" Feuer.","hard")]),
    ("en2de_house","en","lang", "The German word for house is", " Haus.", [(" blue.","easy"),(" Baum.","med"),(" Wohnung.","hard")]),
    ("en2de_dog", "en", "lang", "The German word for dog is", " Hund.", [(" table.","easy"),(" Vogel.","med"),(" Katze.","hard")]),
    ("en2de_red", "en", "lang", "The German word for red is", " rot.", [(" car.","easy"),(" grün.","med"),(" blau.","hard")]),
    ("en2de_book","en","lang", "The German word for book is", " Buch.", [(" apple.","easy"),(" Tisch.","med"),(" Heft.","hard")]),
    # ---------- Tech Concepts (NOT trained code — concept familiarity only) ----------
    ("tech_print","en","tech", "In Python, to print text to the console you use the function", " print().", [(" banana.","easy"),(" open().","med"),(" println().","hard")]),
    ("tech_comment","en","tech","In Python, a single-line comment starts with the symbol", " #.", [(" @.","easy"),(" //.","med"),(" --.","hard")]),
    ("tech_json", "en","tech", "JSON stores data as collections of key-value", " pairs.", [(" apples.","easy"),(" rows.","med"),(" tables.","hard")]),
    ("tech_list", "en","tech", "In Python, an empty list is written as", " [].", [(" XX.","easy"),(" ().","med"),(" {}.","hard")]),
    ("tech_html", "en","tech", "Web pages are written in the markup language", " HTML.", [(" banana.","easy"),(" Python.","med"),(" XML.","hard")]),
    ("tech_https","en","tech", "A secure website uses the protocol", " HTTPS.", [(" dog.","easy"),(" FTP.","med"),(" HTTP.","hard")]),
    ("tech_var",  "en","tech", "In programming, a named value that can change is called a", " variable.", [(" table.","easy"),(" color.","med"),(" constant.","hard")]),
    ("tech_bit",  "en","tech", "The smallest unit of digital information is the", " bit.", [(" apple.","easy"),(" meter.","med"),(" byte.","hard")]),
    ("tech_loop", "en","tech", "In programming, code that repeats is called a", " loop.", [(" banana.","easy"),(" door.","med"),(" function.","hard")]),
    ("tech_sql",  "en","tech", "Relational databases are queried with the language", " SQL.", [(" car.","easy"),(" English.","med"),(" Python.","hard")]),
]

HARDNESS = ["easy", "med", "hard"]


def continuation_nll(model, sp, device, prompt, continuation):
    prompt_ids = sp.EncodeAsIds(prompt)
    full_ids = sp.EncodeAsIds(prompt + continuation)
    cont_start = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        cont_start += 1
    if cont_start >= len(full_ids):
        return float("inf"), None
    first_tok = full_ids[cont_start]
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x)["logits"][0].float()
    logp = torch.log_softmax(logits, dim=-1)
    losses = [float(-logp[pos - 1, int(y[pos - 1])].item()) for pos in range(max(1, cont_start), len(full_ids))]
    return sum(losses) / max(1, len(losses)), first_tok


def topk_hit(model, sp, device, prompt, first_tok, k=10):
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x)["logits"][0, -1].float()
    order = torch.argsort(logits, descending=True)
    pos = (order == first_tok).nonzero(as_tuple=True)[0]
    rank = int(pos.item()) if pos.numel() else 10**9
    return rank, rank < k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model = build_model(args.model_config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise RuntimeError(f"state mismatch missing={len(missing)} extra={len(extra)}")
    model.eval()
    step = (payload.get("state") or {}).get("step")

    rows = []
    for pid, lang, cat, prompt, correct, dists in FACT_PROBES:
        nll_c, first_c = continuation_nll(model, sp, device, prompt, correct)
        per_h = {}
        for h in HARDNESS:
            ds = [d for d, hh in dists if hh == h]
            if not ds:
                continue
            best = min(continuation_nll(model, sp, device, prompt, d)[0] for d in ds)
            per_h[h] = {"margin": round(best - nll_c, 4), "ok": bool(best - nll_c > 0)}
        rank, hit = topk_hit(model, sp, device, prompt, first_c, args.topk) if first_c is not None else (10**9, False)
        hardest = next((h for h in reversed(HARDNESS) if h in per_h), None)
        rows.append({"id": pid, "lang": lang, "cat": cat, "correct": correct.strip(),
                     "per_hardness": per_h, "strict_ok": per_h[hardest]["ok"] if hardest else False,
                     "strict_level": hardest, "topk_rank": rank, "topk_hit": bool(hit)})

    def acc_at(sub, h):
        vals = [r["per_hardness"][h]["ok"] for r in sub if h in r["per_hardness"]]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def strict_acc(sub):
        return sum(r["strict_ok"] for r in sub) / max(1, len(sub))

    cats = ["geo", "sci", "hist", "lang", "tech"]
    print(f"=== KNOWLEDGE PROFILE — step {step}  (n={len(rows)} probes) ===")
    print(f"{'category':10s}{'strict':>8s}{'easy':>8s}{'med':>8s}{'hard':>8s}{'top'+str(args.topk):>8s}")
    cat_out = {}
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        if not sub:
            continue
        e = acc_at(sub, "easy"); m = acc_at(sub, "med"); hd = acc_at(sub, "hard")
        sa = strict_acc(sub)
        tk = sum(r["topk_hit"] for r in sub) / len(sub)
        fmt = lambda a: f"{a[0]*100:5.0f}%" if a[0] is not None else "   — "
        print(f"{c:10s}{sa*100:7.0f}%{fmt(e):>8s}{fmt(m):>8s}{fmt(hd):>8s}{tk*100:7.0f}%")
        cat_out[c] = {"n": len(sub), "strict_acc": sa, "easy": e[0], "med": m[0], "hard": hd[0], "topk_hit": tk}
    print("-" * 50)
    ov_e = acc_at(rows, "easy"); ov_m = acc_at(rows, "med"); ov_h = acc_at(rows, "hard")
    print(f"{'OVERALL':10s}{strict_acc(rows)*100:7.0f}%{ov_e[0]*100:7.0f}%{ov_m[0]*100:7.0f}%{ov_h[0]*100:7.0f}%"
          f"{sum(r['topk_hit'] for r in rows)/len(rows)*100:7.0f}%")
    print("strict = beats the HARDEST distractor present.  tech = CONCEPTS only (0% code trained).")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "step": step, "checkpoint": str(args.checkpoint), "n_probes": len(rows),
            "overall": {"strict_acc": strict_acc(rows), "easy": ov_e[0], "med": ov_m[0], "hard": ov_h[0],
                        "topk_hit": sum(r["topk_hit"] for r in rows) / len(rows)},
            "by_category": cat_out,
            "note": "tech = technical concepts (print/#/JSON), NOT trained code; this run trained 0% code.",
            "probes": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
