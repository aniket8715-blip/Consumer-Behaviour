"""
Perceptual Map AI Agent Pipeline
--------------------------------
A brand-and-product-agnostic tool. Either upload a real perceptual-map
survey export, OR generate sample data to test-drive the tool - then get:
  1. An interactive perceptual map (pick any two attributes as axes)
  2. Auto-detected clusters and whitespace zones
  3. An AI-generated strategic brief (gap -> behavior -> recommendation),
     rendered as marketer-ready cards, not raw text

Nothing here is hardcoded to any category. Attribute names, brand count,
and brand names are all read from the uploaded file / user input at runtime.
The sample-data generator is equally generic: it builds data around
whatever category/brand/attribute names the user types in.
"""

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Perceptual Map Agent Pipeline", layout="wide", page_icon="\U0001F5FA")

EARTH_COLORS = [
    "#2E5339", "#B0413E", "#C08552", "#3D5A80", "#7A5C3E",
    "#5B6C5D", "#8A6552", "#4C3B4D", "#6B8F71", "#A45C40",
]

# ----------------------------------------------------------------------
# Styling - warm, earthy, editorial. No default Streamlit look.
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: #3D2B1F; }

.stApp { background-color: #FAF6EF; }

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E4DCCB;
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] { color: #8A7A63; }

.gap-card {
    background: #FFFFFF;
    border: 1px solid #E4DCCB;
    border-left: 5px solid var(--accent, #B0413E);
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.gap-card h4 { margin: 0 0 6px 0; font-family: 'Fraunces', serif; color: #3D2B1F; }
.gap-tag {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 8px;
}
.tag-opportunity { background: #E1EEE1; color: #2E5339; }
.tag-crowded { background: #F3E3D8; color: #8A6552; }
.tag-avoid { background: #F5DEDD; color: #B0413E; }
.tag-impact-high { background: #2E5339; color: white; }
.tag-impact-medium { background: #C08552; color: white; }
.tag-impact-low { background: #C7BFAE; color: #3D2B1F; }

.section-label {
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 12px;
    font-weight: 600;
    color: #8A7A63;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Generic parsing (works for any category)
# ----------------------------------------------------------------------
def detect_block_structure(df: pd.DataFrame):
    raw_headers = list(df.columns[1:])
    normalized = [re.sub(r"\.\d+$", "", h) for h in raw_headers]
    first = normalized[0]
    block_len = None
    for i in range(1, len(normalized)):
        if normalized[i] == first:
            block_len = i
            break
    if block_len is None:
        block_len = len(normalized)
    n_brands = len(normalized) // block_len
    attrs = normalized[:block_len]
    return attrs, n_brands, block_len


def compute_brand_means(df, attrs, n_brands, block_len, brand_names):
    data = df.iloc[:, 1:]
    rows = []
    for i in range(n_brands):
        block = data.iloc[:, i * block_len:(i + 1) * block_len]
        block.columns = attrs
        rows.append(block.mean())
    result = pd.DataFrame(rows, columns=attrs)
    result.index = brand_names
    return result.round(2)


# ----------------------------------------------------------------------
# Sample / dummy data generator - generic, driven entirely by whatever
# category, brand names, and attribute names the user provides.
# Clearly synthetic - for testing the pipeline, not real market research.
# ----------------------------------------------------------------------
def generate_dummy_csv(brand_names, attribute_names, n_respondents, seed=42):
    rng = np.random.default_rng(seed)
    n_brands = len(brand_names)
    n_attrs = len(attribute_names)

    # give each brand a distinct random "profile" per attribute so the
    # generated map actually has spread, clusters, and whitespace to find
    brand_profiles = rng.uniform(1.8, 4.6, size=(n_brands, n_attrs))

    columns = {"ID": range(1, n_respondents + 1)}
    for b in range(n_brands):
        for a, attr in enumerate(attribute_names):
            vals = rng.normal(loc=brand_profiles[b, a], scale=0.9, size=n_respondents)
            vals = np.clip(np.round(vals), 1, 5).astype(int)
            col_name = attr if b == 0 else f"{attr}.{b}"
            columns[col_name] = vals

    return pd.DataFrame(columns)


# ----------------------------------------------------------------------
# Map interpreter logic
# ----------------------------------------------------------------------
def find_clusters(means, x_attr, y_attr, distance_threshold):
    coords = means[[x_attr, y_attr]].values
    if len(coords) < 2:
        return {means.index[0]: 0} if len(coords) == 1 else {}
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage="average")
    labels = model.fit_predict(coords)
    return dict(zip(means.index, labels))


def find_whitespace(means, x_attr, y_attr):
    x_vals, y_vals = means[x_attr], means[y_attr]
    x_mid = (x_vals.max() + x_vals.min()) / 2 if len(x_vals) > 1 else x_vals.mean()
    y_mid = (y_vals.max() + y_vals.min()) / 2 if len(y_vals) > 1 else y_vals.mean()
    quadrants = {
        f"high {x_attr} / high {y_attr}": int(((x_vals > x_mid) & (y_vals > y_mid)).sum()),
        f"high {x_attr} / low {y_attr}": int(((x_vals > x_mid) & (y_vals <= y_mid)).sum()),
        f"low {x_attr} / high {y_attr}": int(((x_vals <= x_mid) & (y_vals > y_mid)).sum()),
        f"low {x_attr} / low {y_attr}": int(((x_vals <= x_mid) & (y_vals <= y_mid)).sum()),
    }
    empty = [q for q, count in quadrants.items() if count == 0]
    return quadrants, empty, x_mid, y_mid


def plot_map(means, x_attr, y_attr, clusters, x_mid, y_mid):
    fig = go.Figure()
    for i, brand in enumerate(means.index):
        color = EARTH_COLORS[clusters.get(brand, i) % len(EARTH_COLORS)] if clusters else EARTH_COLORS[i % len(EARTH_COLORS)]
        fig.add_trace(go.Scatter(
            x=[means.loc[brand, x_attr]], y=[means.loc[brand, y_attr]],
            mode="markers+text", text=[brand], textposition="top center",
            textfont=dict(size=13, family="Fraunces, serif", color="#3D2B1F"),
            marker=dict(size=18, color=color, line=dict(width=1.5, color="#FAF6EF")),
            name=brand, showlegend=False,
        ))
    fig.add_vline(x=x_mid, line_dash="dash", line_color="#B0413E", opacity=0.4)
    fig.add_hline(y=y_mid, line_dash="dash", line_color="#B0413E", opacity=0.4)
    fig.update_layout(
        xaxis_title=x_attr, yaxis_title=y_attr,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FAF6EF",
        font=dict(family="Inter, sans-serif", color="#3D2B1F", size=13),
        height=540, margin=dict(l=50, r=50, t=30, b=50),
        xaxis=dict(gridcolor="#EDE6D6"), yaxis=dict(gridcolor="#EDE6D6"),
    )
    return fig


# ----------------------------------------------------------------------
# LLM reasoning agents - generic, structured JSON output for clean
# rendering as marketer-ready cards
# ----------------------------------------------------------------------
AGENT_PROMPT = """You are a market strategy analyst. You will be given brand \
positioning data from a perceptual map survey. The attributes and brands \
are specific to whatever category this data comes from - reason generically \
from the actual attribute names and numbers given, do not assume any category.

Attributes measured: {attrs}
Brand mean scores:
{table}

Selected axes for the 2D map: x = {x_attr}, y = {y_attr}
Detected clusters: {clusters}
Empty (whitespace) quadrants: {whitespace}

Return ONLY valid JSON (no markdown fences, no preamble) matching this exact schema:
{{
  "gaps": [
    {{
      "title": "short title for this gap or cluster",
      "classification": "opportunity" | "crowded" | "avoid",
      "rationale": "1-2 sentences, grounded only in the attributes/numbers given",
      "behavior_driver": "1 sentence on likely consumer behavior behind this",
      "recommendation": "one concrete 4P action (product/price/place/promotion)",
      "impact": "high" | "medium" | "low",
      "effort": "high" | "medium" | "low"
    }}
  ]
}}
Cover every empty quadrant and every detected cluster. 3-6 items total.
"""


def call_gemini(api_key, prompt):
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = response.text
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def render_gap_cards(gaps):
    tag_class = {"opportunity": "tag-opportunity", "crowded": "tag-crowded", "avoid": "tag-avoid"}
    accent = {"opportunity": "#2E5339", "crowded": "#C08552", "avoid": "#B0413E"}
    for g in gaps:
        cls = g.get("classification", "opportunity")
        impact = g.get("impact", "medium")
        effort = g.get("effort", "medium")
        st.markdown(f"""
        <div class="gap-card" style="--accent: {accent.get(cls, '#B0413E')}">
            <span class="gap-tag {tag_class.get(cls, 'tag-opportunity')}">{cls.upper()}</span>
            <span class="gap-tag tag-impact-{impact}">IMPACT: {impact.upper()}</span>
            <span class="gap-tag tag-impact-{effort}" style="opacity:0.7">EFFORT: {effort.upper()}</span>
            <h4>{g.get('title', '')}</h4>
            <p style="color:#5C4A38; margin:6px 0;">{g.get('rationale', '')}</p>
            <p class="section-label">Consumer behavior</p>
            <p style="margin:2px 0 10px 0;">{g.get('behavior_driver', '')}</p>
            <p class="section-label">Recommended action</p>
            <p style="margin:2px 0;">{g.get('recommendation', '')}</p>
        </div>
        """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("Perceptual map agent pipeline")
st.caption("Works with any brand or product category. Bring your own survey data, or generate sample data to test the tool first.")

mode = st.radio(
    "Data source", ["Upload my own survey CSV", "Generate sample data (to test the tool)"],
    horizontal=True, label_visibility="collapsed",
)

df = None
if mode == "Upload my own survey CSV":
    uploaded = st.file_uploader("Upload perceptual map survey CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
else:
    with st.form("dummy_data_form"):
        st.markdown("**Describe the category you want to simulate.** This generates realistic-looking synthetic survey data so you can test the whole pipeline before running it on real research.")
        category = st.text_input("Category (for your reference only, e.g. 'sportswear', 'banking apps')", value="Sample category")
        brands_raw = st.text_input("Brand names (comma-separated)", value="Brand A, Brand B, Brand C, Brand D")
        attrs_raw = st.text_input("Attributes to measure (comma-separated)", value="Premium feel, Trustworthiness, Innovation, Value for money")
        n_resp = st.slider("Number of simulated respondents", 20, 300, 90)
        submitted = st.form_submit_button("Generate sample data")
    if submitted:
        brand_names_dummy = [b.strip() for b in brands_raw.split(",") if b.strip()]
        attrs_dummy = [a.strip() for a in attrs_raw.split(",") if a.strip()]
        df = generate_dummy_csv(brand_names_dummy, attrs_dummy, n_resp)
        st.session_state["dummy_df"] = df
        st.session_state["dummy_brand_names"] = brand_names_dummy
    elif "dummy_df" in st.session_state:
        df = st.session_state["dummy_df"]

if df is not None:
    attrs, n_brands, block_len = detect_block_structure(df)
    st.success(f"Detected {n_brands} brands x {block_len} attributes: {', '.join(attrs)}")

    if mode != "Upload my own survey CSV" and "dummy_brand_names" in st.session_state:
        default_names = ", ".join(st.session_state["dummy_brand_names"])
    else:
        default_names = ", ".join([f"Brand {i+1}" for i in range(n_brands)])

    brand_input = st.text_input(
        f"Brand names, in order (comma-separated, {n_brands} expected)", value=default_names,
    )
    brand_names = [b.strip() for b in brand_input.split(",")]

    if len(brand_names) != n_brands:
        st.error(f"Expected {n_brands} brand names, got {len(brand_names)}. Adjust the list above.")
    else:
        means = compute_brand_means(df, attrs, n_brands, block_len, brand_names)

        m1, m2, m3 = st.columns(3)
        m1.metric("Brands", n_brands)
        m2.metric("Attributes measured", len(attrs))
        m3.metric("Respondents", len(df))

        with st.expander("Brand mean scores"):
            st.dataframe(means, use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        x_attr = col1.selectbox("X-axis attribute", attrs, index=0)
        y_attr = col2.selectbox("Y-axis attribute", attrs, index=min(1, len(attrs) - 1))
        threshold = col3.slider("Cluster sensitivity", 0.1, 2.0, 0.6, 0.1)

        clusters = find_clusters(means, x_attr, y_attr, threshold)
        quadrants, empty, x_mid, y_mid = find_whitespace(means, x_attr, y_attr)

        st.plotly_chart(plot_map(means, x_attr, y_attr, clusters, x_mid, y_mid), use_container_width=True)

        with st.expander("Detected quadrant occupancy"):
            st.write(quadrants)
            st.write("**Empty (whitespace) quadrants:**", empty if empty else "None")

        st.divider()
        st.subheader("Strategic brief")
        st.caption("Requires your own free Google API key (aistudio.google.com/apikey). Used only for this session, never stored.")
        api_key = st.text_input("Google API key", type="password")

        if st.button("Generate brief", disabled=not api_key, type="primary"):
            prompt = AGENT_PROMPT.format(
                attrs=", ".join(attrs), table=means.to_string(),
                x_attr=x_attr, y_attr=y_attr, clusters=clusters, whitespace=empty,
            )
            with st.spinner("Running gap, behavior, and strategy agents..."):
                try:
                    result = call_gemini(api_key, prompt)
                    st.session_state["brief"] = result
                except Exception as e:
                    st.error(f"Couldn't parse the response - try again. ({e})")

        if "brief" in st.session_state:
            render_gap_cards(st.session_state["brief"].get("gaps", []))
            st.download_button(
                "Download brief (JSON)",
                json.dumps(st.session_state["brief"], indent=2),
                file_name="strategic_brief.json",
            )
else:
    st.info("Upload a CSV, or switch to sample data generation above, to get started.")
