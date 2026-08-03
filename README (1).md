# Perceptual map agent pipeline

A brand-and-product-agnostic Streamlit app. Either upload any perceptual-map-
style survey export (repeated attribute blocks, one block per brand - the
standard shape a Google Forms multi-section survey exports to), OR generate
realistic sample data to test-drive the tool first - then get:

1. An interactive map (pick any two measured attributes as axes)
2. Auto-detected brand clusters and empty (whitespace) quadrants
3. An AI-generated strategic brief, rendered as marketer-ready cards (not
   raw text) - gap classification, consumer behavior link, and a concrete
   4P recommendation with impact/effort tags for each finding

Nothing is hardcoded to a category - attribute names, number of brands, and
brand names are all read from your file (or typed in for sample data
generation) at runtime.

## Two ways to get data in

- **Upload my own survey CSV** - use this with real research data.
- **Generate sample data** - type in any category, brand names, and
  attributes you want, and the app builds a realistic-looking synthetic
  dataset with actual spread/clusters/whitespace to explore. This is purely
  for testing the pipeline - it is NOT real market research and shouldn't be
  presented to stakeholders as such.

## Run locally (free)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", point it at your repo and `app.py`.
4. Deploy - no credit card, no cost.

## Using the AI brief feature

The map + clustering + whitespace detection work with zero API cost (pure
pandas/scikit-learn). The strategic brief step calls Claude, so you'll need
an Anthropic API key:

1. Get one free at https://console.anthropic.com (new accounts get starter
   credits).
2. Paste it into the app's password field when prompted - it's used only
   for that session and never saved or logged anywhere in this app.

## Expected CSV format

- First column: respondent ID (any name)
- Remaining columns: attribute questions, repeated once per brand, in the
  same order for every brand (this is what a standard Google Forms export
  with one "section" per brand looks like)
- Brand names are NOT expected in the header row (Google Forms doesn't
  export them there) - you'll enter them once in the app, in the same order
  as your form's sections
