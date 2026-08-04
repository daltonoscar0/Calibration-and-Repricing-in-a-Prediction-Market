# Decision log

- Processed match table written as CSV, not Parquet: Parquet requires `pyarrow`, outside the four permitted dependencies.
- `Makefile` uses `PYTHON ?= python3`, so the pipeline runs against any 3.11+ interpreter with `requirements.txt` installed.
- Pinnacle closing quotes (PSCH/PSCD/PSCA) are present from 2012/13, earlier than expected; they cover 65% of the sample and market average (BbAv*/Avg*) covers 2005/06–2011/12.
- Overround stripped both ways: proportional normalisation is the headline series, Shin (1993) is reported alongside. Shin's insider share z is solved per match by Brent root-finding on the constraint that the adjusted probabilities sum to one; degenerate books with no sign change fall back to proportional (none occurred).
- Probability bins are fixed edges rather than empirical deciles — deciles between 0.1 and 0.9, finer bins below 0.10 and above 0.90, so the favourite–longshot tails are resolved. Bins with no observations are dropped from the table.
- Favourite–longshot figure reports every bin including the thin tails (n=47 in the lowest populated bin) and labels n on each point rather than suppressing low-count bins.
