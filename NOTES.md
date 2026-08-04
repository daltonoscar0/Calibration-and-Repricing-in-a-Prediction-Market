# Decision log

- Processed match table written as CSV, not Parquet: Parquet requires `pyarrow`, outside the four permitted dependencies.
- `Makefile` uses `PYTHON ?= python3`, so the pipeline runs against any 3.11+ interpreter with `requirements.txt` installed.
- Pinnacle closing quotes (PSCH/PSCD/PSCA) are present from 2012/13, earlier than expected; they cover 65% of the sample and market average (BbAv*/Avg*) covers 2005/06–2011/12.
- Overround stripped both ways: proportional normalisation is the headline series, Shin (1993) is reported alongside. Shin's insider share z is solved per match by Brent root-finding on the constraint that the adjusted probabilities sum to one; degenerate books with no sign change fall back to proportional (none occurred).
- Probability bins are fixed edges rather than empirical deciles — deciles between 0.1 and 0.9, finer bins below 0.10 and above 0.90, so the favourite–longshot tails are resolved. Bins with no observations are dropped from the table.
- Favourite–longshot figure reports every bin including the thin tails (n=47 in the lowest populated bin) and labels n on each point rather than suppressing low-count bins.
- Event study uses proportional probabilities, not Shin: the two series are near-identical over the mid-probability range where club win probabilities live, and proportional keeps the benchmark regression interpretable.
- Event windows are confined to the club's own season, so no window straddles a season boundary; events are required to have at least 8 matches inside k = -5..+5. All 70 candidate events clear that bar (minimum window length 9).
- Benchmark is estimated per club-season on that club's matches outside every event window in the season, so a second sacking in the same season cannot contaminate the first one's benchmark (Aston Villa 2015/16, Chelsea 2022/23 and Southampton 2022/23 each carry two events).
- Opponent strength is the opponent's season-average implied win probability, computed over the full season. This is in-sample by construction; it is a market-based quality proxy rather than a forecast, so the look-ahead is accepted and noted as a limitation.
- Bootstrap resamples whole events with replacement (5,000 draws, fixed seed), so inference is clustered at the event level.
- Event dates flagged as confident to within a day or two in `src/sackings.csv`: Alan Irvine (West Brom 2014), Mark Hughes (Stoke 2018), David Wagner (Huddersfield 2019), Quique Sanchez Flores (Watford 2019), Chris Wilder (Sheffield United 2021), Nathan Jones (Southampton 2023), Gary O'Neil (Wolves 2024). At match frequency a one-day error moves an event across a fixture only if the announcement sits on a matchday.
