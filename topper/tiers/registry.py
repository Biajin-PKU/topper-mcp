"""Load bundled CCF/CAS snapshots and resolve venue strings."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from topper.models import TierLabels
from topper.normalize import compact, fold, normalize_venue

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(name: str) -> dict[str, Any]:
    path = DATA_DIR / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class TierRegistry:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        from topper import config

        self.data_dir = Path(data_dir) if data_dir else config.data_dir()
        self.meta = self._read("meta.json")
        self.ccf_doc = self._read("ccf_venues.json")
        # CAS / FMS catalogues carry their own licences and are not bundled;
        # `stats()` and `topper doctor` report which tables are loaded.
        self.cas_doc = self._read_optional("cas_journals.json")
        self.fms_doc = self._read_optional("fms_journals.json")
        self.as_of = str(
            self.meta.get("as_of")
            or self.ccf_doc.get("as_of")
            or self.cas_doc.get("as_of")
            or ""
        )
        self._ccf_index: dict[str, dict[str, Any]] = {}
        self._cas_index: dict[str, dict[str, Any]] = {}
        self._fms_index: dict[str, dict[str, Any]] = {}
        self._build_index(self.ccf_doc.get("venues") or [], self._ccf_index, kind="ccf")
        self._build_index(self.cas_doc.get("journals") or [], self._cas_index, kind="cas")
        self._build_index(
            self.fms_doc.get("journals") or [], self._fms_index, kind="fms"
        )

    def _read(self, name: str) -> dict[str, Any]:
        path = self.data_dir / name
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _read_optional(self, name: str) -> dict[str, Any]:
        path = self.data_dir / name
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _build_index(
        self,
        rows: list[dict[str, Any]],
        index: dict[str, dict[str, Any]],
        *,
        kind: str,
    ) -> None:
        for row in rows:
            keys = {fold(row.get("key") or ""), fold(row.get("name") or "")}
            for a in row.get("aliases") or []:
                keys.add(fold(a))
                keys.add(normalize_venue(a))
                keys.add(compact(a))
            keys.add(normalize_venue(row.get("name") or ""))
            keys.add(compact(row.get("name") or ""))
            keys.add(compact(row.get("key") or ""))
            # Index ISSN / EISSN so a bare ISSN venue string can resolve too.
            for issn in row.get("issn") or []:
                keys.add(fold(issn))
                keys.add(compact(issn))
            payload = dict(row)
            payload["_kind"] = kind
            for k in keys:
                if not k:
                    continue
                # Prefer exact catalog rows; first writer wins for stability
                index.setdefault(k, payload)

    def _find(
        self, venue: str
    ) -> tuple[
        Optional[dict[str, Any]], Optional[dict[str, Any]], Optional[dict[str, Any]]
    ]:
        if not venue:
            return None, None, None
        candidates = [
            fold(venue),
            normalize_venue(venue),
            compact(venue),
        ]
        ccf = cas = fms = None
        for c in candidates:
            if not c:
                continue
            if ccf is None and c in self._ccf_index:
                ccf = self._ccf_index[c]
            if cas is None and c in self._cas_index:
                cas = self._cas_index[c]
            if fms is None and c in self._fms_index:
                fms = self._fms_index[c]
        return ccf, cas, fms

    def lookup(self, venue: Optional[str]) -> TierLabels:
        if not venue:
            return TierLabels(source_as_of=self.as_of or None)
        ccf_row, cas_row, fms_row = self._find(venue)
        ccf = None
        cas_zone = None
        key = None
        if ccf_row:
            ccf = ccf_row.get("ccf")
            key = ccf_row.get("key")
        cas_major = cas_top = None
        sci = ssci = ahci = None
        jcr_quartile = impact_factor = None
        if cas_row:
            cas_zone = cas_row.get("cas_zone")
            key = key or cas_row.get("key")
            cas_major = cas_row.get("cas_major")
            cas_top = cas_row.get("cas_top")
            sci = cas_row.get("sci")
            ssci = cas_row.get("ssci")
            ahci = cas_row.get("ahci")
            jcr_quartile = cas_row.get("jcr_quartile")
            impact_factor = cas_row.get("impact_factor")
        fms_tier = fms_discipline = None
        if fms_row:
            fms_tier = fms_row.get("fms_tier")
            fms_discipline = fms_row.get("fms_discipline")
            key = key or fms_row.get("key")
        return TierLabels(
            ccf=str(ccf) if ccf else None,
            cas_zone=int(cas_zone) if cas_zone is not None else None,
            cas_major=str(cas_major) if cas_major else None,
            cas_top=bool(cas_top) if cas_top is not None else None,
            fms_tier=str(fms_tier).upper() if fms_tier else None,
            fms_discipline=str(fms_discipline) if fms_discipline else None,
            sci=bool(sci) if sci is not None else None,
            ssci=bool(ssci) if ssci is not None else None,
            ahci=bool(ahci) if ahci is not None else None,
            jcr_quartile=str(jcr_quartile) if jcr_quartile else None,
            impact_factor=float(impact_factor) if impact_factor is not None else None,
            source_as_of=self.as_of or None,
        )

    def resolve_key(self, venue: Optional[str]) -> Optional[str]:
        if not venue:
            return None
        ccf_row, cas_row, fms_row = self._find(venue)
        if ccf_row:
            return str(ccf_row.get("key") or "") or None
        if cas_row:
            return str(cas_row.get("key") or "") or None
        if fms_row:
            return str(fms_row.get("key") or "") or None
        return None

    def stats(self) -> dict[str, Any]:
        journals = self.cas_doc.get("journals") or []
        fms = self.fms_doc.get("journals") or []
        return {
            "as_of": self.as_of,
            "ccf_venues": len(self.ccf_doc.get("venues") or []),
            "cas_journals": len(journals),
            "fms_journals": len(fms),
            "sci_journals": sum(1 for j in journals if j.get("sci")),
            "ssci_journals": sum(1 for j in journals if j.get("ssci")),
            "ahci_journals": sum(1 for j in journals if j.get("ahci")),
            "jcr_quartiled_journals": sum(1 for j in journals if j.get("jcr_quartile")),
            "data_dir": str(self.data_dir),
        }


@lru_cache(maxsize=1)
def get_default_registry() -> TierRegistry:
    from topper import config

    return TierRegistry(config.data_dir())


def lookup_venue(venue: str) -> TierLabels:
    return get_default_registry().lookup(venue)
