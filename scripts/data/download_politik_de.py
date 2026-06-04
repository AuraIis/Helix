"""Phase-3+ politik corpus downloader.

Pulls every public German + EU political-data source we can hit without an
API key. Output: structured JSONL files in /staging/politik_de/raw/<source>/
plus a flat .txt mirror suitable for the Auralis cleaning pipeline.

Usage:
    python download_politik_de.py --source <name>

Sources (each callable independently):

    bundestag_mdb        — Bundestag MdB-Stammdaten (XML zip, all MPs since 1949)
    bundestag_protokolle — Plenarprotokolle XML (one per session, current term)
    lobbyregister_de     — Bundestag-Lobbyregister full JSON export
    abgeordnetenwatch    — abgeordnetenwatch.de API v2 (questions + answers)
    europarl_meps        — Members of the European Parliament (open data)

Each source writes a manifest.json with counts + elapsed time.
The flat .txt mirror has one record per blank-line-separated block,
ready to feed into assemble_for_filter.py with `--mode text`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "/staging/politik_de/raw"
ROOT = Path(os.environ.get("POLITIK_RAW_ROOT", DEFAULT_ROOT))


def _ua():
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Auralis/2.0 Politik-Crawler"
        )
    }


def _fetch(url: str, *, timeout: int = 60, decode: str | None = "utf-8"):
    req = urllib.request.Request(url, headers=_ua())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
    if decode is None:
        return raw
    return raw.decode(decode, errors="replace")


def _write_manifest(out_dir: Path, **kwargs):
    info = {
        "source": kwargs.pop("source"),
        "started_at": kwargs.pop("started_at"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **kwargs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def _strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
           .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'"))
    return s.strip()


# ============================================================================
# 1. Bundestag MdB-Stammdaten
# ============================================================================
def dl_bundestag_mdb():
    out_dir = ROOT / "bundestag_mdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "MdB-Stammdaten.zip"
    out_jsonl = out_dir / "bundestag_mdb.jsonl"
    out_text = out_dir / "bundestag_mdb.txt"
    print(f"=== bundestag-mdb -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    url = "https://www.bundestag.de/resource/blob/472878/12c5d2b5d8aa90ba5bce5e3f88f25dba/MdB-Stammdaten-data.zip"
    print(f"  fetching {url}", flush=True)
    try:
        raw = _fetch(url, decode=None)
    except Exception as e:
        # Codex P3: surface fatal fetch errors to the chain script via a
        # non-zero exit. Returning silently let chain_politik_de.sh mark
        # this stage "successful" and proceed to clean stage on no data.
        sys.exit(f"FATAL: cannot fetch MdB stammdaten: {e}")
    out_zip.write_bytes(raw)
    print(f"  zip: {len(raw)/1e6:.1f} MB", flush=True)

    import xml.etree.ElementTree as ET
    n_mps = 0
    with zipfile.ZipFile(out_zip) as zf, \
         out_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         out_text.open("w", encoding="utf-8") as f_txt:
        xml_name = max(zf.namelist(), key=lambda n: zf.getinfo(n).file_size)
        with zf.open(xml_name) as fh:
            tree = ET.parse(fh)
        root = tree.getroot()
        for mdb in root.findall(".//MDB"):
            rec: dict[str, Any] = {}
            rec["id"] = (mdb.findtext("ID", "") or "").strip()
            name = mdb.find("NAMEN/NAME")
            if name is not None:
                for tag in ("NACHNAME", "VORNAME", "ADEL", "PRAEFIX", "ANREDE_TITEL", "AKAD_TITEL"):
                    rec[tag.lower()] = (name.findtext(tag, "") or "").strip()
            bio = mdb.find("BIOGRAFISCHE_ANGABEN")
            if bio is not None:
                for tag in ("GEBURTSDATUM", "GEBURTSORT", "GEBURTSLAND", "STERBEDATUM",
                            "GESCHLECHT", "FAMILIENSTAND", "RELIGION", "BERUF",
                            "PARTEI_KURZ", "VITA_KURZ"):
                    rec[tag.lower()] = (bio.findtext(tag, "") or "").strip()
            mandates = []
            for wp in mdb.findall(".//WAHLPERIODE"):
                m = {
                    "wp": (wp.findtext("WP", "") or "").strip(),
                    "von": (wp.findtext("MDBWP_VON", "") or "").strip(),
                    "bis": (wp.findtext("MDBWP_BIS", "") or "").strip(),
                    "wkr_nr": (wp.findtext("WKR_NUMMER", "") or "").strip(),
                    "wkr_name": (wp.findtext("WKR_NAME", "") or "").strip(),
                    "land": (wp.findtext("WKR_LAND", "") or "").strip(),
                    "liste": (wp.findtext("LISTE", "") or "").strip(),
                    "mandatsart": (wp.findtext("MANDATSART", "") or "").strip(),
                }
                mandates.append(m)
            rec["mandate"] = mandates

            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

            full = " ".join(p for p in [
                rec.get("anrede_titel"), rec.get("akad_titel"),
                rec.get("vorname"), rec.get("adel"), rec.get("praefix"), rec.get("nachname"),
            ] if p)
            parts = [f"Politiker: {full}"]
            if rec.get("partei_kurz"):
                parts.append(f"Partei: {rec['partei_kurz']}")
            if rec.get("geburtsdatum"):
                geb = f"Geboren: {rec['geburtsdatum']}"
                if rec.get("geburtsort"):
                    geb += f" in {rec['geburtsort']}"
                parts.append(geb)
            if rec.get("beruf"):
                parts.append(f"Beruf: {rec['beruf']}")
            if mandates:
                wp_list = ", ".join(sorted({m["wp"] for m in mandates if m.get("wp")}))
                parts.append(f"Wahlperioden: {wp_list}")
            if rec.get("vita_kurz"):
                parts.append(_strip_html(rec["vita_kurz"]))
            f_txt.write("\n".join(parts) + "\n\n")
            n_mps += 1

    _write_manifest(out_dir, source="bundestag/MdB-Stammdaten", started_at=started,
                    politicians=n_mps, elapsed_seconds=time.time() - t0)
    print(f"  done: {n_mps:,} MdBs in {(time.time()-t0)/60:.1f} min", flush=True)


# ============================================================================
# 2. Bundestag Plenarprotokolle (XML, current term)
# ============================================================================
def dl_bundestag_protokolle():
    out_dir = ROOT / "bundestag_protokolle"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "bundestag_protokolle.jsonl"
    out_text = out_dir / "bundestag_protokolle.txt"
    print(f"=== bundestag-protokolle -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Bundestag publishes Plenarprotokolle via an AJAX filterlist endpoint.
    # Each call returns up to 20 records and includes the next-offset.
    base = "https://www.bundestag.de/ajax/filterlist/de/services/opendata/866354-866354"
    seen: set[str] = set()
    unique: list[str] = []
    offset = 0
    limit = 20
    print(f"  fetching plenarprotokolle index via filterlist", flush=True)
    while True:
        url = f"{base}?limit={limit}&noFilterSet=true&offset={offset}"
        try:
            html = _fetch(url, timeout=30)
        except Exception as e:
            print(f"  FATAL: cannot fetch index page offset={offset}: {e}", flush=True)
            break
        new_hrefs = re.findall(r'href="(https://www\.bundestag\.de/resource/blob/[^"]+\.xml)"', html)
        if not new_hrefs:
            break
        added = 0
        for h in new_hrefs:
            if h not in seen:
                seen.add(h)
                unique.append(h)
                added += 1
        if added == 0:
            break
        offset += limit
        time.sleep(0.4)
        if offset > 1000:  # safety cap
            break
    print(f"  found {len(unique)} unique XML protocols", flush=True)
    if not unique:
        # Codex P3: empty index = the upstream changed format or is offline.
        # We must not silently produce a 0-protocol manifest; the chain
        # script needs a non-zero exit to skip this stage.
        sys.exit("FATAL: no protokolle XML hrefs found via filterlist — upstream format may have changed")

    import xml.etree.ElementTree as ET
    n_proto = 0
    n_speeches = 0
    with out_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         out_text.open("w", encoding="utf-8") as f_txt:
        for i, url in enumerate(unique[:300], 1):
            try:
                xml_text = _fetch(url, timeout=120)
            except Exception as e:
                print(f"  [{i}/{len(unique)}] skip {url}: {e}", flush=True)
                continue
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as e:
                print(f"  [{i}] parse fail {url}: {e}", flush=True)
                continue
            sitzung_nr = root.attrib.get("sitzung-nr", "?")
            wp = root.attrib.get("wahlperiode", "?")
            for rede in root.iter("rede"):
                redner = rede.find(".//redner/name")
                if redner is not None:
                    spn_titel = redner.findtext("titel", "") or ""
                    spn_vor = redner.findtext("vorname", "") or ""
                    spn_nach = redner.findtext("nachname", "") or ""
                    spn_fraktion = redner.findtext("fraktion", "") or ""
                    speaker = " ".join(p for p in (spn_titel, spn_vor, spn_nach) if p).strip()
                else:
                    speaker = "Unbekannt"
                    spn_fraktion = ""
                text_parts = []
                for p in rede.iter("p"):
                    txt = "".join(p.itertext()).strip()
                    if txt:
                        text_parts.append(txt)
                full_text = "\n".join(text_parts)
                if not full_text or len(full_text) < 50:
                    continue
                rec = {
                    "wp": wp, "sitzung": sitzung_nr,
                    "redner": speaker, "fraktion": spn_fraktion,
                    "text": full_text, "source_url": url,
                }
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_txt.write(f"[Plenarprotokoll {wp}/{sitzung_nr}] {speaker} ({spn_fraktion}):\n")
                f_txt.write(full_text + "\n\n")
                n_speeches += 1
            n_proto += 1
            if i % 10 == 0:
                print(f"  [{i}/{len(unique)}] protocols={n_proto}, speeches={n_speeches}", flush=True)

    _write_manifest(out_dir, source="bundestag/plenarprotokolle", started_at=started,
                    protocols=n_proto, speeches=n_speeches,
                    elapsed_seconds=time.time() - t0)
    print(f"  done: {n_proto} protocols, {n_speeches:,} speeches in {(time.time()-t0)/60:.1f} min",
          flush=True)


# ============================================================================
# 3. Lobbyregister Bundestag — full JSON export
# ============================================================================
def dl_lobbyregister_de():
    out_dir = ROOT / "lobbyregister_de"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "lobbyregister_de.jsonl"
    out_text = out_dir / "lobbyregister_de.txt"
    print(f"=== lobbyregister-de -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    base = "https://www.lobbyregister.bundestag.de/lobbyregister-api/public/searchOpensearchResults"
    page = 0
    page_size = 100
    n_entries = 0
    with out_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         out_text.open("w", encoding="utf-8") as f_txt:
        while True:
            url = f"{base}?from={page * page_size}&size={page_size}&sort=registrierdatum-asc"
            try:
                body = _fetch(url, timeout=60)
                data = json.loads(body)
            except Exception as e:
                print(f"  page {page} fail: {e}", flush=True)
                break
            results = data.get("results") or data.get("hits", {}).get("hits", []) or []
            if not results:
                break
            for entry in results:
                src = entry.get("_source") or entry
                rec = {
                    "register_id": src.get("registernummer") or src.get("id"),
                    "name": src.get("name") or src.get("organisation"),
                    "rechtsform": src.get("rechtsform"),
                    "sitz": src.get("sitz"),
                    "interessenbereiche": src.get("interessenbereiche") or [],
                    "auftraggeber": src.get("auftraggeber") or [],
                    "jahresangaben": src.get("jahresangaben") or [],
                    "url": f"https://www.lobbyregister.bundestag.de/suche/{src.get('registernummer','')}",
                    "raw": src,
                }
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                txt_parts = [f"Lobbyeintrag: {rec.get('name','?')} ({rec.get('register_id','?')})"]
                if rec.get("rechtsform"):
                    txt_parts.append(f"Rechtsform: {rec['rechtsform']}")
                if rec.get("sitz"):
                    txt_parts.append(f"Sitz: {rec['sitz']}")
                if rec.get("interessenbereiche"):
                    txt_parts.append("Interessenbereiche: " + ", ".join(map(str, rec["interessenbereiche"])))
                if rec.get("auftraggeber"):
                    ag = [str(a.get("name", a)) if isinstance(a, dict) else str(a) for a in rec["auftraggeber"]]
                    if ag:
                        txt_parts.append("Auftraggeber: " + ", ".join(ag))
                f_txt.write("\n".join(txt_parts) + "\n\n")
                n_entries += 1
            print(f"  page {page}: {len(results)} entries (total {n_entries})", flush=True)
            if len(results) < page_size:
                break
            page += 1
            time.sleep(0.5)

    _write_manifest(out_dir, source="bundestag/lobbyregister", started_at=started,
                    entries=n_entries, elapsed_seconds=time.time() - t0)
    print(f"  done: {n_entries:,} lobby entries in {(time.time()-t0)/60:.1f} min", flush=True)


# ============================================================================
# 4. abgeordnetenwatch.de — Q&A
# ============================================================================
def dl_abgeordnetenwatch():
    """abgeordnetenwatch.de — politicians (35k+ records). The /questions
    endpoint returned 404 in our smoke test; politicians is the canonical
    profile resource and what we actually want for the politik corpus.
    """
    out_dir = ROOT / "abgeordnetenwatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "abgeordnetenwatch.jsonl"
    out_text = out_dir / "abgeordnetenwatch.txt"
    print(f"=== abgeordnetenwatch politicians -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    base = "https://www.abgeordnetenwatch.de/api/v2"
    n_p = 0

    with out_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         out_text.open("w", encoding="utf-8") as f_txt:
        page = 0
        per_page = 200
        max_pages = 200  # 40k politicians cap = full corpus (~35k actual)
        while page < max_pages:
            url = f"{base}/politicians?range_start={page * per_page}&range_end={(page + 1) * per_page}"
            try:
                body = _fetch(url, timeout=60)
                data = json.loads(body)
            except Exception as e:
                print(f"  page {page} fail: {e}", flush=True)
                break
            results = data.get("data") or []
            if not results:
                break
            for p in results:
                party = p.get("party") or {}
                rec = {
                    "id": p.get("id"),
                    "label": p.get("label"),
                    "first_name": p.get("first_name"),
                    "last_name": p.get("last_name"),
                    "year_of_birth": p.get("year_of_birth"),
                    "education": p.get("education"),
                    "occupation": p.get("occupation"),
                    "residence": p.get("residence"),
                    "sex": p.get("sex"),
                    "party_label": party.get("label") if isinstance(party, dict) else None,
                    "ext_url": p.get("abgeordnetenwatch_url"),
                    "api_url": p.get("api_url"),
                }
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # Flat text record
                parts = [f"Politiker: {rec.get('label','?')}"]
                if rec.get("party_label"):
                    parts.append(f"Partei: {rec['party_label']}")
                if rec.get("year_of_birth"):
                    parts.append(f"Jahrgang: {rec['year_of_birth']}")
                if rec.get("occupation"):
                    parts.append(f"Beruf: {rec['occupation']}")
                if rec.get("education"):
                    parts.append(f"Bildung: {_strip_html(rec['education'])}")
                if rec.get("residence"):
                    parts.append(f"Wohnort: {rec['residence']}")
                f_txt.write("\n".join(parts) + "\n\n")
                n_p += 1
            print(f"  page {page}: {len(results)} politicians (total {n_p:,})", flush=True)
            if len(results) < per_page:
                break
            page += 1
            time.sleep(0.3)

    _write_manifest(out_dir, source="abgeordnetenwatch.de/politicians", started_at=started,
                    politicians=n_p, elapsed_seconds=time.time() - t0)
    print(f"  done: {n_p:,} politicians in {(time.time()-t0)/60:.1f} min", flush=True)


# ============================================================================
# 5. EU Parliament — MEPs (open data, XML)
# ============================================================================
def dl_europarl_meps():
    out_dir = ROOT / "europarl_meps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "europarl_meps.jsonl"
    out_text = out_dir / "europarl_meps.txt"
    print(f"=== europarl-meps -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    url = "https://www.europarl.europa.eu/meps/en/full-list/xml"
    try:
        xml_text = _fetch(url, timeout=120)
    except Exception as e:
        sys.exit(f"FATAL: cannot fetch EuroParl MEPs: {e}")

    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         out_text.open("w", encoding="utf-8") as f_txt:
        for mep in root.iter("mep"):
            rec = {
                "id": mep.findtext("id", ""),
                "fullName": mep.findtext("fullName", ""),
                "country": mep.findtext("country", ""),
                "politicalGroup": mep.findtext("politicalGroup", ""),
                "nationalPoliticalGroup": mep.findtext("nationalPoliticalGroup", ""),
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_txt.write(
                f"MEP: {rec['fullName']} — {rec['country']} — "
                f"{rec['politicalGroup']} ({rec['nationalPoliticalGroup']})\n\n"
            )
            n += 1
    _write_manifest(out_dir, source="europarl/meps", started_at=started,
                    meps=n, elapsed_seconds=time.time() - t0)
    print(f"  done: {n:,} MEPs in {(time.time()-t0):.1f}s", flush=True)


# ============================================================================
# 6/7. Rechtsinformationen Bund — Bundesgesetze + Gerichtsentscheidungen
#     https://testphase.rechtsinformationen.bund.de  (DigitalService GmbH)
#     OpenAPI: https://docs.rechtsinformationen.bund.de/v3/api-docs
# ============================================================================
RIS_BASE = "https://testphase.rechtsinformationen.bund.de"


def _ris_strip_html(html: str) -> str:
    """RIS HTML responses are reasonably clean; drop tags, preserve newlines
    around block elements."""
    if not html:
        return ""
    # Block-element line breaks first
    html = re.sub(r"</(p|div|h[1-6]|li|tr|br)\s*>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", "", html)
    # Entity decode (light)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
                 .replace("&gt;", ">").replace("&quot;", '"')
                 .replace("&#39;", "'").replace("&apos;", "'")
                 .replace("&shy;", "").replace("&ndash;", "–").replace("&mdash;", "—"))
    # Collapse runs of blank lines
    html = re.sub(r"\n\s*\n+", "\n\n", html).strip()
    return html


def _ris_paginate(list_url_base: str):
    """Yield items page-by-page from a RIS list endpoint.
    Stops when the page returns no items or the totalItems count is exceeded.
    """
    page = 0
    page_size = 100
    seen = 0
    while True:
        url = f"{list_url_base}?size={page_size}&pageIndex={page}"
        try:
            body = _fetch(url, timeout=60)
            data = json.loads(body)
        except Exception as e:
            # Codex P3: page-0 failure must propagate as a fatal exit so the
            # chain script knows. A failure on a later page is recoverable
            # (we have records already) — log + stop yielding.
            if page == 0 and seen == 0:
                sys.exit(f"FATAL: RIS first-page fetch failed: {e}")
            print(f"  page {page} fail: {e} — yielding what we have ({seen} records)", flush=True)
            return
        members = data.get("member") or []
        total = data.get("totalItems") or None
        if not members:
            return
        for m in members:
            yield m.get("item") or m
        seen += len(members)
        print(f"  page {page}: {len(members)} ({seen}/{total or '?'})", flush=True)
        if len(members) < page_size or (total and seen >= total):
            return
        page += 1
        time.sleep(0.2)


def dl_rechtsinfo_caselaw():
    """All 82k+ federal court decisions, anonymized full text via RIS API."""
    out_dir = ROOT / "rechtsinfo_caselaw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "rechtsinfo_caselaw.jsonl"
    out_text = out_dir / "rechtsinfo_caselaw.txt"
    progress_file = out_dir / "_progress.json"
    print(f"=== rechtsinfo-caselaw -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Resume: skip docs we've already pulled
    done_ids: set[str] = set()
    if progress_file.exists():
        try:
            done_ids = set(json.loads(progress_file.read_text()))
        except Exception:
            done_ids = set()

    n_records = 0
    n_skipped = 0
    n_errors = 0

    list_url = f"{RIS_BASE}/v1/case-law"
    # Append mode — we may resume across runs
    with out_jsonl.open("a", encoding="utf-8") as f_jsonl, \
         out_text.open("a", encoding="utf-8") as f_txt:
        for item in _ris_paginate(list_url):
            doc_no = item.get("documentNumber")
            if not doc_no:
                continue
            if doc_no in done_ids:
                n_skipped += 1
                continue
            # Pull HTML body
            html_url = f"{RIS_BASE}/v1/case-law/{doc_no}.html"
            try:
                html = _fetch(html_url, timeout=30)
            except Exception as e:
                n_errors += 1
                if n_errors % 20 == 0:
                    print(f"  errors so far: {n_errors}", flush=True)
                continue
            body = _ris_strip_html(html)
            rec = {
                "documentNumber": doc_no,
                "ecli": item.get("ecli"),
                "headline": item.get("headline"),
                "decisionDate": item.get("decisionDate"),
                "fileNumbers": item.get("fileNumbers"),
                "courtType": item.get("courtType"),
                "courtName": item.get("courtName"),
                "judicialBody": item.get("judicialBody"),
                "documentType": item.get("documentType"),
                "source_url": f"{RIS_BASE}/v1/case-law/{doc_no}",
                "body": body,
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Flat text for the cleaning pipeline
            header = (
                f"[{rec.get('courtType','?')}] {rec.get('headline','?')} "
                f"({rec.get('documentType','?')}, {rec.get('decisionDate','?')})"
            )
            f_txt.write(header + "\n")
            if rec.get("ecli"):
                f_txt.write(f"ECLI: {rec['ecli']}\n")
            if body:
                f_txt.write(body + "\n")
            f_txt.write("\n")
            done_ids.add(doc_no)
            n_records += 1
            if n_records % 200 == 0:
                # Persist progress periodically
                progress_file.write_text(json.dumps(sorted(done_ids)))
                rate = n_records / max(time.time() - t0, 0.01)
                print(f"  {n_records:,} records, {rate:.1f}/s, errors {n_errors}", flush=True)

    progress_file.write_text(json.dumps(sorted(done_ids)))
    _write_manifest(out_dir, source="rechtsinformationen.bund.de/case-law", started_at=started,
                    decisions=n_records, skipped=n_skipped, errors=n_errors,
                    elapsed_seconds=time.time() - t0)
    print(f"  done: {n_records:,} new, {n_skipped:,} skipped, {n_errors} errors "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


def dl_rechtsinfo_legislation():
    """All 2.4k federal laws + decrees with current point-in-time HTML."""
    out_dir = ROOT / "rechtsinfo_legislation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "rechtsinfo_legislation.jsonl"
    out_text = out_dir / "rechtsinfo_legislation.txt"
    print(f"=== rechtsinfo-legislation -> {out_dir} ===", flush=True)
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    n_records = 0
    n_errors = 0
    list_url = f"{RIS_BASE}/v1/legislation"
    with out_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         out_text.open("w", encoding="utf-8") as f_txt:
        for item in _ris_paginate(list_url):
            # Legislation items reveal their HTML manifest URL in `encoding`
            encodings = item.get("encoding") or []
            html_url = next(
                (e.get("contentUrl") for e in encodings
                 if isinstance(e, dict) and e.get("encodingFormat") == "text/html"),
                None,
            )
            if not html_url:
                continue
            full_url = html_url if html_url.startswith("http") else f"{RIS_BASE}{html_url}"
            try:
                html = _fetch(full_url, timeout=60)
            except Exception as e:
                n_errors += 1
                continue
            body = _ris_strip_html(html)
            rec = {
                "id": item.get("@id"),
                "name": item.get("name"),
                "shortTitle": item.get("shortTitle") or item.get("alternativeTitles"),
                "officialTitle": item.get("officialTitle"),
                "abbreviation": item.get("abbreviation"),
                "datePublished": item.get("datePublished"),
                "agent": item.get("agent"),
                "source_url": full_url,
                "body": body,
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            header_bits = [rec.get("name") or rec.get("shortTitle") or rec.get("officialTitle") or "?"]
            if rec.get("abbreviation"):
                header_bits.append(f"({rec['abbreviation']})")
            if rec.get("datePublished"):
                header_bits.append(f"– {rec['datePublished']}")
            f_txt.write(" ".join(header_bits) + "\n")
            if body:
                f_txt.write(body + "\n")
            f_txt.write("\n")
            n_records += 1
            if n_records % 100 == 0:
                rate = n_records / max(time.time() - t0, 0.01)
                print(f"  {n_records}/{2425} laws, {rate:.1f}/s, errors {n_errors}", flush=True)

    _write_manifest(out_dir, source="rechtsinformationen.bund.de/legislation", started_at=started,
                    laws=n_records, errors=n_errors, elapsed_seconds=time.time() - t0)
    print(f"  done: {n_records:,} laws, {n_errors} errors in {(time.time()-t0)/60:.1f} min",
          flush=True)


SOURCES = {
    "bundestag_mdb": dl_bundestag_mdb,
    "bundestag_protokolle": dl_bundestag_protokolle,
    "lobbyregister_de": dl_lobbyregister_de,
    "abgeordnetenwatch": dl_abgeordnetenwatch,
    "europarl_meps": dl_europarl_meps,
    "rechtsinfo_caselaw": dl_rechtsinfo_caselaw,
    "rechtsinfo_legislation": dl_rechtsinfo_legislation,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=sorted(SOURCES.keys()), required=True)
    args = p.parse_args()
    print(f"POLITIK_RAW_ROOT = {ROOT}", flush=True)
    SOURCES[args.source]()


if __name__ == "__main__":
    main()
