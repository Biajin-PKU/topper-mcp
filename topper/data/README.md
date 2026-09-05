# Tier tables

The gate is only as broad as the tables installed here.

## Bundled

| File | What it is |
|---|---|
| `ccf_venues.json` | CCF recommended catalog (conferences + journals, A/B/C). Published openly by the China Computer Federation. |
| `flagship_venues.json` | Hand-curated list of venues treated as top-tier regardless of tables. Names only — no zone, quartile or impact factor is asserted. Edit it for your field. |
| `flagship_authors.json`, `flagship_orgs.json` | Small authority seeds used as a soft rank boost. |

With just these, coverage is CCF venues plus the flagship list. Add the
tables below for journal-ranked fields.

## Not bundled — add your own

`cas_journals.json` (中科院分区表) and `fms_journals.json` (FMS 经管期刊评级)
are not included: both catalogs carry their own terms of use. `topper doctor`
reports them as missing; the gate then uses CCF plus the flagship list.

If you have a licensed copy, drop it in this directory (or point
`TOPPER_DATA_DIR` elsewhere) in this shape:

```json
{
  "as_of": "2026-08-27",
  "journals": [
    {
      "key": "journal-of-econometrics",
      "name": "Journal of Econometrics",
      "cas_zone": 1,
      "cas_major": "经济学",
      "cas_top": true,
      "sci": true, "ssci": true, "ahci": false,
      "jcr_quartile": "Q1",
      "impact_factor": 9.9
    }
  ]
}
```

`key` is the normalized name (lowercase, non-alphanumerics collapsed to `-`);
the loader also matches on `name`, so `key` is optional. Every field except
`name` is optional — a row with only `cas_zone` still works.

`fms_journals.json` uses the same envelope with `fms_tier` (`A`/`B`/`C`/`D`/
`T1`/`T2`) and `fms_discipline`.

## Known limitation

Working-paper series (NBER, SSRN, RePEc) are not journals and appear in no
journal ranking. Fields that circulate their main results through them are
under-covered by any venue-tier gate.
