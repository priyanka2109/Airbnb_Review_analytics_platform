# Airbnb Review Analytics Platform

An end-to-end data analytics and sentiment analysis platform designed to help users and hosts understand customer feedback from Airbnb reviews.

## Overview

This project analyzes Airbnb property reviews using Natural Language Processing (NLP) techniques to extract sentiment and uncover actionable insights.

The platform helps answer questions like:

- What are the most common positive and negative themes in reviews?
- How does sentiment vary across listings or locations?
- Which listings have the best customer satisfaction?

## Features

- Sentiment analysis using :contentReference[oaicite:0]{index=0} and :contentReference[oaicite:1]{index=1}
- Review preprocessing and text cleaning
- Exploratory Data Analysis (EDA)
- Visualization of sentiment distribution
- Listing-level sentiment insights
- Interactive web app using :contentReference[oaicite:2]{index=2}

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- TextBlob
- VADER
- Flask

## Project Structure

```text
airbnb-review-analytics/
│
├── data/
├── notebooks/
├── app.py
├── templates/
├── static/
└── README.md
