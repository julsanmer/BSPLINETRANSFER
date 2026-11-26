# SPICE kernels

Only the meta-kernel text files (`*.tm`) are tracked. The binary kernels they
load are ignored by Git (`kernels/*.bsp`, `kernels/*.tls`) and must be
downloaded into this directory before any `--ephemeris spice` run. The unit
tests and both notebooks use the built-in Keplerian ephemeris and do not need
them.

| File | Used by | Size | Source |
| --- | --- | --- | --- |
| `naif0012.tls` | all meta-kernels | 5 KB | NAIF generic leap-seconds kernel |
| `de440s.bsp` | all meta-kernels | 31 MB | NAIF generic planetary ephemeris (DE440, short span) |
| `1989ml_2028_2039.bsp` | `fasttransfer_1989ml.tm` | 275 KB | JPL Horizons SPK for asteroid 1989 ML, 2028-01-01 to 2039-01-01 |
| `tempel1_2028_2049.bsp` | `fasttransfer_tempel1.tm` | 432 KB | JPL Horizons SPK for comet 9P/Tempel 1, 2028-01-01 to 2049-01-01 |

Generic NAIF kernels (run from the repository root):

```bash
curl -L -o kernels/naif0012.tls https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
```

```bash
curl -L -o kernels/de440s.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
```

Small-body SPKs come from the JPL Horizons API, which returns the SPK as a
base64 field of a JSON response (`EPHEM_TYPE=SPK`). Example for 1989 ML:

```bash
curl -s "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='1989%20ML'&OBJ_DATA=NO&MAKE_EPHEM=YES&EPHEM_TYPE=SPK&START_TIME='2028-01-01'&STOP_TIME='2039-01-01'" | python3 -c "import json,sys,base64; d=json.load(sys.stdin); open('kernels/1989ml_2028_2039.bsp','wb').write(base64.b64decode(d['spk']))"
```

Use `COMMAND='9P'` (or the Horizons record number for Tempel 1) with
`STOP_TIME='2049-01-01'` and the output name `kernels/tempel1_2028_2049.bsp`
for the comet. Horizons may ask you to pick a specific apparition record for a
comet; if the response has no `spk` field, inspect its `result` text and add
the record number to `COMMAND`.

The meta-kernels reference these files by paths relative to the repository
root, so run SPICE commands from there.
