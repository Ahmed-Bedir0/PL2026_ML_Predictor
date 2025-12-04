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
