"""
Predict the outcome of the 2025/26 Premier League season using a random
forest classifier.

This script takes a series of historical Premier League seasons in CSV
format, builds summary statistics for each club (points, wins, draws,
losses, goals for/against and goal difference) and then trains a
RandomForestClassifier from scikit‑learn to predict the final league
position of each team in a subsequent season.
The 2025/26 Premier League will include 20 clubs – the 17 sides that remained in the
division in 2024/25 and three promoted clubs (Leeds United, Burnley
and Sunderland)

Each CSV file (e.g., `PL_2018-19.csv`, `PL_2019-20.csv`, etc.)
lists every Premier League match in the given season with columns for
the date, home side (Team 1), final score (FT), half‑time score (HT)
and away side (Team 2). An example of the first few rows of the
2019/20 file is shown below:

```
              Date          Team 1   FT   HT            Team 2
0   Fri Aug 9 2019       Liverpool  4-1  4-0           Norwich
1  Sat Aug 10 2019        West Ham  0-5  0-1          Man City
2  Sat Aug 10 2019     Bournemouth  1-1  0-0  Sheffield United
3  Sat Aug 10 2019         Burnley  3-0  0-0       Southampton
4  Sat Aug 10 2019  Crystal Palace  0-0  0-0           Everton
```
