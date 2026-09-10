# ============================================================
# BIG BAZAAR — CUSTOMER INTELLIGENCE DASHBOARD
# GOOGLE COLAB — COMPLETE FIXED VERSION
# ============================================================

# IMPORTANT:
# Do NOT upgrade pandas / numpy / sklearn in Colab.
# This code is designed to use the packages already available.
#
# If Gradio is missing, uncomment ONLY this:
#
# !pip install -q "gradio==6.24.0" "gradio-client==2.6.1"


# ============================================================
# IMPORTS
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. CREATE CUSTOMER DATASET
# ============================================================

np.random.seed(42)

n_per_group = 125


# ------------------------------------------------------------
# GROUP 1 — HIGH VALUE CUSTOMERS
# ------------------------------------------------------------

high_value = pd.DataFrame({
    "Age": np.random.normal(40, 5, n_per_group),
    "Annual_Income": np.random.normal(110000, 15000, n_per_group),
    "Spending_Score": np.random.normal(88, 7, n_per_group),
    "Visit_Frequency": np.random.normal(17, 3, n_per_group),
    "Average_Transaction_Value": np.random.normal(4200, 600, n_per_group)
})


# ------------------------------------------------------------
# GROUP 2 — DEAL HUNTERS
# ------------------------------------------------------------

deal_hunters = pd.DataFrame({
    "Age": np.random.normal(38, 6, n_per_group),
    "Annual_Income": np.random.normal(30000, 7000, n_per_group),
    "Spending_Score": np.random.normal(25, 8, n_per_group),
    "Visit_Frequency": np.random.normal(4, 1.5, n_per_group),
    "Average_Transaction_Value": np.random.normal(900, 180, n_per_group)
})


# ------------------------------------------------------------
# GROUP 3 — HIGH INCOME / LOW ENGAGEMENT
# ------------------------------------------------------------

low_engagement = pd.DataFrame({
    "Age": np.random.normal(48, 6, n_per_group),
    "Annual_Income": np.random.normal(125000, 18000, n_per_group),
    "Spending_Score": np.random.normal(35, 8, n_per_group),
    "Visit_Frequency": np.random.normal(3, 1, n_per_group),
    "Average_Transaction_Value": np.random.normal(2500, 450, n_per_group)
})


# ------------------------------------------------------------
# GROUP 4 — FREQUENT SHOPPERS
# ------------------------------------------------------------

frequent_shoppers = pd.DataFrame({
    "Age": np.random.normal(32, 5, n_per_group),
    "Annual_Income": np.random.normal(60000, 10000, n_per_group),
    "Spending_Score": np.random.normal(70, 8, n_per_group),
    "Visit_Frequency": np.random.normal(14, 3, n_per_group),
    "Average_Transaction_Value": np.random.normal(2300, 400, n_per_group)
})


df = pd.concat(
    [
        high_value,
        deal_hunters,
        low_engagement,
        frequent_shoppers
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# CLEAN VALUES
# ------------------------------------------------------------

df["Age"] = df["Age"].clip(18, 75)

df["Annual_Income"] = df["Annual_Income"].clip(
    10000,
    200000
)

df["Spending_Score"] = df["Spending_Score"].clip(
    1,
    100
)

df["Visit_Frequency"] = df["Visit_Frequency"].clip(
    1,
    30
)

df["Average_Transaction_Value"] = df[
    "Average_Transaction_Value"
].clip(
    200,
    8000
)

df = df.round(2)


# ------------------------------------------------------------
# CUSTOMER ID
# ------------------------------------------------------------

df.insert(
    0,
    "Customer_ID",
    [f"BB-{1001+i}" for i in range(len(df))]
)


# ============================================================
# 2. PREPROCESSING
# ============================================================

features = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Visit_Frequency",
    "Average_Transaction_Value"
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[features]
)


# ============================================================
# 3. MODEL SELECTION
# ============================================================

k_values = list(range(2, 9))

inertias = []
silhouettes = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(
        model.inertia_
    )

    silhouettes.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )


best_k = k_values[
    np.argmax(silhouettes)
]


# ============================================================
# 4. FINAL K-MEANS MODEL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 5. SEGMENT NAMING
# ============================================================

cluster_stats = df.groupby(
    "Cluster"
)[features].mean()


segment_names = {}

income_median = cluster_stats[
    "Annual_Income"
].median()

spending_median = cluster_stats[
    "Spending_Score"
].median()

visit_median = cluster_stats[
    "Visit_Frequency"
].median()


for cluster in cluster_stats.index:

    income = cluster_stats.loc[
        cluster,
        "Annual_Income"
    ]

    spending = cluster_stats.loc[
        cluster,
        "Spending_Score"
    ]

    visits = cluster_stats.loc[
        cluster,
        "Visit_Frequency"
    ]

    if (
        spending > spending_median
        and visits > visit_median
        and income > income_median
    ):

        segment_names[
            cluster
        ] = "The Inner Circle"

    elif (
        income < income_median
        and spending < spending_median
    ):

        segment_names[
            cluster
        ] = "The Deal Hunters"

    elif (
        income > income_median
        and visits < visit_median
    ):

        segment_names[
            cluster
        ] = "The Untapped"

    else:

        segment_names[
            cluster
        ] = "The Regulars"


# ------------------------------------------------------------
# SAFETY FALLBACK
# ------------------------------------------------------------

used = set()

available_segments = [
    "The Inner Circle",
    "The Untapped",
    "The Regulars",
    "The Deal Hunters"
]

for cluster in cluster_stats.index:

    proposed = segment_names[
        cluster
    ]

    if proposed in used:

        for option in available_segments:

            if option not in used:

                segment_names[
                    cluster
                ] = option

                break

    used.add(
        segment_names[cluster]
    )


df["Segment"] = df[
    "Cluster"
].map(segment_names)


# ============================================================
# 6. MARKETING STRATEGIES
# ============================================================

marketing_strategy = {

    "The Inner Circle":
        "Premium rewards, early access, exclusive launches and VIP experiences.",

    "The Untapped":
        "Personalised recommendations, high-value bundles and re-engagement campaigns.",

    "The Regulars":
        "Loyalty rewards, subscription-style offers and frequency-based incentives.",

    "The Deal Hunters":
        "Flash deals, coupons, value bundles and price-sensitive promotions."
}


# ============================================================
# 7. EXPORT CSV
# ============================================================

output_file = (
    "/content/Big_Bazaar_Customer_Segmentation_Final.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 8. COLOUR PALETTE
# ============================================================

WHITE = "#FFFFFF"
CREAM = "#FBFAF5"

TEXT = "#151A16"
TEXT_SECONDARY = "#59615B"
TEXT_MUTED = "#7A817B"

BORDER = "#E5E7E1"

GREEN = "#B9D7A5"
GREEN_SOFT = "#EAF3E4"

YELLOW = "#F6E5A6"
YELLOW_SOFT = "#FCF7DD"

PINK = "#F3C9D2"
PINK_SOFT = "#FCEEF1"

BLUE = "#C5DDF0"
BLUE_SOFT = "#EDF6FB"

TEAL = "#2F6F67"
DEEP_GREEN = "#183D35"


# ============================================================
# 9. MATPLOTLIB CHART STYLE
# ============================================================

def style_chart(ax, title, ylabel=None):

    ax.set_title(
        title,
        fontsize=15,
        fontweight="600",
        pad=15,
        color=TEXT
    )

    if ylabel:
        ax.set_ylabel(
            ylabel,
            color=TEXT_SECONDARY
        )

    ax.tick_params(
        colors=TEXT_SECONDARY
    )

    ax.grid(
        axis="y",
        alpha=0.18
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor(
        "none"
    )


# ============================================================
# 10. CHARTS
# ============================================================

segment_order = [
    "The Inner Circle",
    "The Untapped",
    "The Regulars",
    "The Deal Hunters"
]


def segment_distribution():

    counts = (
        df["Segment"]
        .value_counts()
        .reindex(segment_order)
        .fillna(0)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.barh(
        counts.index,
        counts.values,
        color=[
            GREEN,
            BLUE,
            PINK,
            YELLOW
        ]
    )

    ax.invert_yaxis()

    for i, value in enumerate(
        counts.values
    ):
        ax.text(
            value + 3,
            i,
            str(int(value)),
            va="center",
            fontsize=10,
            color=TEXT
        )

    style_chart(
        ax,
        "Customer Distribution",
        "Customers"
    )

    fig.tight_layout()

    return fig


def spending_chart():

    values = (
        df.groupby("Segment")
        ["Spending_Score"]
        .mean()
        .reindex(segment_order)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        values.index,
        values.values,
        color=[
            GREEN,
            BLUE,
            PINK,
            YELLOW
        ]
    )

    ax.set_ylabel(
        "Spending Score",
        color=TEXT_SECONDARY
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for i, value in enumerate(
        values.values
    ):
        ax.text(
            i,
            value + 2,
            f"{value:.1f}",
            ha="center",
            color=TEXT
        )

    style_chart(
        ax,
        "Average Spending Score"
    )

    fig.tight_layout()

    return fig


def frequency_chart():

    values = (
        df.groupby("Segment")
        ["Visit_Frequency"]
        .mean()
        .reindex(segment_order)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        values.index,
        values.values,
        color=[
            BLUE,
            GREEN,
            YELLOW,
            PINK
        ]
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for i, value in enumerate(
        values.values
    ):
        ax.text(
            i,
            value + 0.4,
            f"{value:.1f}",
            ha="center",
            color=TEXT
        )

    style_chart(
        ax,
        "Average Visit Frequency"
    )

    fig.tight_layout()

    return fig


def income_chart():

    values = (
        df.groupby("Segment")
        ["Annual_Income"]
        .mean()
        .reindex(segment_order)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        values.index,
        values.values / 1000,
        color=[
            GREEN,
            BLUE,
            PINK,
            YELLOW
        ]
    )

    ax.set_ylabel(
        "Income (₹ Thousands)",
        color=TEXT_SECONDARY
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for i, value in enumerate(
        values.values / 1000
    ):
        ax.text(
            i,
            value + 2,
            f"₹{value:.1f}K",
            ha="center",
            color=TEXT
        )

    style_chart(
        ax,
        "Average Annual Income"
    )

    fig.tight_layout()

    return fig


def transaction_chart():

    values = (
        df.groupby("Segment")
        ["Average_Transaction_Value"]
        .mean()
        .reindex(segment_order)
    )

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        values.index,
        values.values,
        color=[
            YELLOW,
            GREEN,
            BLUE,
            PINK
        ]
    )

    ax.set_ylabel(
        "₹",
        color=TEXT_SECONDARY
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for i, value in enumerate(
        values.values
    ):
        ax.text(
            i,
            value + 100,
            f"₹{value:,.0f}",
            ha="center",
            color=TEXT
        )

    style_chart(
        ax,
        "Average Transaction Value"
    )

    fig.tight_layout()

    return fig


def elbow_chart():

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        k_values,
        inertias,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel(
        "Number of Clusters (K)",
        color=TEXT_SECONDARY
    )

    ax.set_ylabel(
        "Inertia",
        color=TEXT_SECONDARY
    )

    style_chart(
        ax,
        "Elbow Analysis"
    )

    fig.tight_layout()

    return fig


def silhouette_chart():

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        k_values,
        silhouettes,
        color=BLUE
    )

    ax.set_xlabel(
        "Number of Clusters (K)",
        color=TEXT_SECONDARY
    )

    ax.set_ylabel(
        "Silhouette Score",
        color=TEXT_SECONDARY
    )

    for i, value in enumerate(
        silhouettes
    ):
        ax.text(
            k_values[i],
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            color=TEXT
        )

    style_chart(
        ax,
        "Silhouette Score by K"
    )

    fig.tight_layout()

    return fig


# ============================================================
# 11. SVG ICONS
# ============================================================

def icon_svg(kind="spark"):

    icons = {

        "home": """
        <svg viewBox="0 0 24 24">
        <path d="M3 11.5 12 4l9 7.5"/>
        <path d="M5 10.5V20h14v-9.5"/>
        <path d="M9 20v-5h6v5"/>
        </svg>
        """,

        "users": """
        <svg viewBox="0 0 24 24">
        <circle cx="9" cy="8" r="3"/>
        <circle cx="17" cy="9" r="2.5"/>
        <path d="M3 20c.5-4 2.5-6 6-6s5.5 2 6 6"/>
        <path d="M15 14c3 0 5 1.7 6 5"/>
        </svg>
        """,

        "spark": """
        <svg viewBox="0 0 24 24">
        <path d="M12 2 14 9l7 3-7 2-2 8-2-8-7-2 7-3 2-7Z"/>
        </svg>
        """,

        "chart": """
        <svg viewBox="0 0 24 24">
        <path d="M4 19V5"/>
        <path d="M4 19h17"/>
        <rect x="7" y="12" width="3" height="5"/>
        <rect x="12" y="9" width="3" height="8"/>
        <rect x="17" y="6" width="3" height="11"/>
        </svg>
        """,

        "plus": """
        <svg viewBox="0 0 24 24">
        <circle cx="12" cy="12" r="9"/>
        <path d="M12 8v8M8 12h8"/>
        </svg>
        """,

        "search": """
        <svg viewBox="0 0 24 24">
        <circle cx="10.5" cy="10.5" r="6.5"/>
        <path d="m16 16 5 5"/>
        </svg>
        """,

        "database": """
        <svg viewBox="0 0 24 24">
        <ellipse cx="12" cy="5" rx="7" ry="3"/>
        <path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/>
        <path d="M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7"/>
        </svg>
        """,

        "arrow": """
        <svg viewBox="0 0 24 24">
        <path d="M5 12h13"/>
        <path d="m13 6 6 6-6 6"/>
        </svg>
        """
    }

    return icons.get(
        kind,
        icons["spark"]
    )


# ============================================================
# 12. HTML HELPERS
# ============================================================

def pastel_icon(icon, bg):

    return f"""
    <div class="pastel-icon"
         style="background:{bg};">
        {icon_svg(icon)}
    </div>
    """


def segment_card(
    title,
    subtitle,
    description,
    icon,
    bg
):

    return f"""
    <div class="segment-card">

        <div class="segment-top">

            {pastel_icon(icon, bg)}

            <div>

                <div class="segment-label">
                    CUSTOMER GROUP
                </div>

                <h3>
                    {title}
                </h3>

            </div>

        </div>

        <div class="segment-subtitle">
            {subtitle}
        </div>

        <p>
            {description}
        </p>

        <div class="segment-arrow">
            Explore segment
            {icon_svg("arrow")}
        </div>

    </div>
    """


# ============================================================
# 13. CUSTOMER PREDICTION
# ============================================================

def predict_customer(
    age,
    income,
    spending,
    visits,
    transaction
):

    values = np.array([
        [
            age,
            income,
            spending,
            visits,
            transaction
        ]
    ])

    scaled = scaler.transform(
        values
    )

    cluster = kmeans.predict(
        scaled
    )[0]

    segment = segment_names[
        cluster
    ]

    description = marketing_strategy[
        segment
    ]

    return f"""
    <div class="prediction-card">

        <div class="prediction-label">
            PREDICTED CUSTOMER SEGMENT
        </div>

        <div class="prediction-name">
            {segment}
        </div>

        <div class="prediction-description">

            Based on the customer's income,
            spending behaviour, visit frequency
            and transaction value, the clustering
            model places this customer in
            <strong>{segment}</strong>.

        </div>

        <div class="badge-row">

            <span class="badge green">
                K-MEANS
            </span>

            <span class="badge yellow">
                PERSONALIZED
            </span>

            <span class="badge blue">
                DATA-DRIVEN
            </span>

        </div>

        <br>

        <strong>
            Recommended approach
        </strong>

        <p>
            {description}
        </p>

    </div>
    """


# ============================================================
# 14. CUSTOMER PROFILE
# ============================================================

def customer_profile(
    customer_id
):

    row = df[
        df["Customer_ID"] == customer_id
    ].iloc[0]

    segment = row["Segment"]

    return f"""
    <div class="prediction-card">

        <div class="prediction-label">
            CUSTOMER PROFILE
        </div>

        <div class="prediction-name">
            {customer_id}
        </div>

        <div class="prediction-description">

            <strong>Segment:</strong>
            {segment}

            <br><br>

            <strong>Age:</strong>
            {row["Age"]:.0f}

            <br>

            <strong>Annual Income:</strong>
            ₹{row["Annual_Income"]:,.0f}

            <br>

            <strong>Spending Score:</strong>
            {row["Spending_Score"]:.1f}

            <br>

            <strong>Visit Frequency:</strong>
            {row["Visit_Frequency"]:.1f}

            <br>

            <strong>Average Transaction:</strong>
            ₹{row["Average_Transaction_Value"]:,.0f}

            <br><br>

            <strong>
                Suggested Strategy:
            </strong>

            <br>

            {marketing_strategy[segment]}

        </div>

    </div>
    """


# ============================================================
# 15. MARKETING PLAYBOOK
# ============================================================

campaign_titles = {

    "The Inner Circle":
        "FIRST DIBS",

    "The Untapped":
        "COME BACK DIFFERENT",

    "The Regulars":
        "MAKE IT A HABIT",

    "The Deal Hunters":
        "DEAL WORTH SAVING"
}


campaign_copies = {

    "The Inner Circle":
        "Early access, premium bundles and loyalty rewards that make customers feel seen.",

    "The Untapped":
        "Personalised recommendations and comeback nudges built around dormant potential.",

    "The Regulars":
        "Habit-building bundles, streak rewards and smart cross-sells to increase basket size.",

    "The Deal Hunters":
        "Sharp, visible offers with genuine value — designed for quick conversion."
}


def campaign_cards():

    cards = ""

    for i, segment in enumerate(
        segment_order,
        1
    ):

        cards += f"""

        <div class="campaign-card">

            <div class="campaign-number">
                CAMPAIGN 0{i}
            </div>

            <div class="campaign-segment">
                {segment.upper()}
            </div>

            <div class="campaign-title">
                {campaign_titles[segment]}
            </div>

            <div class="campaign-copy">
                {campaign_copies[segment]}
            </div>

            <div class="campaign-hook">

                <span>
                    CREATIVE LINE
                </span>

                <b>
                    "{marketing_strategy[segment]}"
                </b>

            </div>

        </div>

        """

    return f"""
    <div class="campaign-grid">
        {cards}
    </div>
    """


# ============================================================
# 16. CSS
# ============================================================

CSS = """

/* ============================================================
   GLOBAL
   ============================================================ */

:root {

    --bg: #FFFFFF;
    --surface: #FBFAF5;
    --surface-2: #FFFFFF;

    --text: #151A16;
    --text-secondary: #59615B;
    --text-muted: #7A817B;

    --border: #E5E7E1;

    --green: #B9D7A5;
    --green-soft: #EAF3E4;

    --yellow: #F6E5A6;
    --yellow-soft: #FCF7DD;

    --pink: #F3C9D2;
    --pink-soft: #FCEEF1;

    --blue: #C5DDF0;
    --blue-soft: #EDF6FB;

    --teal: #2F6F67;
    --deep-green: #183D35;
}


/* ============================================================
   DARK MODE
   ============================================================ */

body.dark-mode {

    --bg: #101412;
    --surface: #171C19;
    --surface-2: #1D2420;

    --text: #FFFFFF;
    --text-secondary: #FFFFFF;
    --text-muted: #FFFFFF;

    --border: #39423C;

    --green-soft: #263326;
    --yellow-soft: #373322;
    --pink-soft: #35272B;
    --blue-soft: #27343D;

    --teal: #8CCDC3;
    --deep-green: #D9EEDB;
}


/* ============================================================
   MAIN APP
   ============================================================ */

html,
body,
.gradio-container {

    background:
        var(--bg) !important;

    color:
        var(--text) !important;

    font-family:
        "DM Sans",
        Arial,
        sans-serif !important;
}


/* ============================================================
   DARK MODE — FORCE READABLE TEXT
   ============================================================ */

body.dark-mode,
body.dark-mode .gradio-container,
body.dark-mode h1,
body.dark-mode h2,
body.dark-mode h3,
body.dark-mode h4,
body.dark-mode h5,
body.dark-mode h6,
body.dark-mode p,
body.dark-mode span,
body.dark-mode div,
body.dark-mode label,
body.dark-mode td,
body.dark-mode th,
body.dark-mode strong,
body.dark-mode small {

    color:
        #FFFFFF !important;
}


/* Keep green accent text visible */
body.dark-mode .segment-subtitle {

    color:
        #8CCDC3 !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

#main-sidebar {

    background:
        var(--surface) !important;

    border-right:
        1px solid var(--border) !important;

    color:
        var(--text) !important;
}


#main-sidebar * {

    color:
        var(--text) !important;
}


body.dark-mode #main-sidebar {

    background:
        #121714 !important;

    border-right:
        1px solid #39423C !important;
}


body.dark-mode #main-sidebar * {

    color:
        #FFFFFF !important;
}


/* ============================================================
   BRAND
   ============================================================ */

.brand-block {

    padding:
        12px 8px 25px 8px;

    border-bottom:
        1px solid var(--border);

    margin-bottom:
        20px;
}


.brand-title {

    font-family:
        Georgia,
        serif;

    font-size:
        31px;

    line-height:
        1;

    color:
        var(--text) !important;

    letter-spacing:
        -1px;
}


.brand-subtitle {

    margin-top:
        8px;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        2px;

    color:
        var(--text-secondary) !important;
}


.brand-est {

    margin-top:
        8px;

    font-size:
        10px;

    color:
        var(--text-muted) !important;

    letter-spacing:
        1px;
}


/* ============================================================
   NAVIGATION
   ============================================================ */

.nav-btn {

    width:
        100% !important;

    border:
        1px solid transparent !important;

    background:
        transparent !important;

    color:
        var(--text) !important;

    border-radius:
        10px !important;

    margin:
        3px 0 !important;

    text-align:
        left !important;

    transition:
        all .18s ease !important;
}


.nav-btn:hover {

    background:
        var(--green-soft) !important;

    border-color:
        var(--border) !important;

    transform:
        translateX(2px);
}


.nav-btn *,
.nav-btn span {

    color:
        var(--text) !important;
}


body.dark-mode .nav-btn,
body.dark-mode .nav-btn *,
body.dark-mode .nav-btn span {

    color:
        #FFFFFF !important;
}


body.dark-mode .nav-btn:hover {

    background:
        #243027 !important;

    border-color:
        #465249 !important;
}


/* ============================================================
   SIDEBAR FOOTER
   ============================================================ */

.sidebar-footer {

    margin-top:
        20px;

    padding:
        15px 12px;

    border:
        1px solid var(--border);

    border-radius:
        12px;

    background:
        var(--surface-2);
}


.sidebar-footer-title {

    font-size:
        11px;

    font-weight:
        700;

    letter-spacing:
        1.5px;

    color:
        var(--text) !important;
}


.sidebar-footer-text {

    font-size:
        11px;

    margin-top:
        6px;

    line-height:
        1.5;

    color:
        var(--text-secondary) !important;
}


/* ============================================================
   PAGE
   ============================================================ */

.page {

    padding:
        45px 55px 70px 55px;

    max-width:
        1500px;

    margin:
        0 auto;
}


/* ============================================================
   TYPOGRAPHY
   ============================================================ */

h1,
h2,
h3 {

    font-family:
        Georgia,
        serif !important;

    color:
        var(--text) !important;

    font-weight:
        400 !important;
}


.hero h1 {

    font-size:
        clamp(48px, 6vw, 82px);

    line-height:
        .94;

    letter-spacing:
        -2.5px;

    max-width:
        900px;

    margin:
        22px 0 20px;
}


.hero h1 em {

    color:
        var(--teal) !important;

    font-style:
        normal;
}


.hero-copy {

    max-width:
        680px;

    color:
        var(--text-secondary) !important;

    font-size:
        16px;

    line-height:
        1.7;
}


/* ============================================================
   EYEBROW
   ============================================================ */

.eyebrow {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        7px;

    padding:
        7px 11px;

    border:
        1px solid var(--border);

    border-radius:
        30px;

    background:
        var(--surface);

    color:
        var(--text-secondary) !important;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        1.3px;
}


.eyebrow-dot {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        var(--green);
}


/* ============================================================
   BADGES
   ============================================================ */

.badge-row {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        8px;

    margin-top:
        20px;
}


.badge {

    padding:
        7px 10px;

    border-radius:
        30px;

    font-size:
        10px;

    font-weight:
        700;

    border:
        1px solid var(--border);

    color:
        var(--text) !important;
}


.badge.green {
    background:
        var(--green-soft);
}

.badge.yellow {
    background:
        var(--yellow-soft);
}

.badge.pink {
    background:
        var(--pink-soft);
}

.badge.blue {
    background:
        var(--blue-soft);
}


/* ============================================================
   VISUAL STRIP
   ============================================================ */

.visual-strip {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        14px;

    margin:
        25px 0 38px;
}


.visual-card {

    min-height:
        115px;

    padding:
        20px;

    border:
        1px solid var(--border);

    border-radius:
        18px;

    position:
        relative;

    overflow:
        hidden;

    transition:
        transform .2s ease;
}


.visual-card:hover {

    transform:
        translateY(-3px);
}


.visual-card.green {
    background:
        var(--green-soft);
}

.visual-card.yellow {
    background:
        var(--yellow-soft);
}

.visual-card.pink {
    background:
        var(--pink-soft);
}

.visual-card.blue {
    background:
        var(--blue-soft);
}


.visual-card .icon {

    width:
        35px;

    height:
        35px;

    margin-bottom:
        14px;
}


.visual-card .icon svg {

    width:
        100%;

    height:
        100%;

    fill:
        none;

    stroke:
        var(--text);

    stroke-width:
        1.6;
}


.visual-card-title {

    font-family:
        Georgia,
        serif;

    font-size:
        19px;

    color:
        var(--text) !important;
}


.visual-card-caption {

    margin-top:
        4px;

    font-size:
        11px;

    color:
        var(--text-secondary) !important;
}


/* ============================================================
   KPI
   ============================================================ */

.kpi-grid {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        14px;

    margin:
        20px 0 45px;
}


.kpi {

    padding:
        22px;

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        16px;

    position:
        relative;
}


.kpi-number {

    font-family:
        Georgia,
        serif;

    font-size:
        34px;

    color:
        var(--text) !important;
}


.kpi-label {

    margin-top:
        4px;

    font-size:
        11px;

    font-weight:
        700;

    letter-spacing:
        1px;

    text-transform:
        uppercase;

    color:
        var(--text-muted) !important;
}


/* ============================================================
   SECTION HEADER
   ============================================================ */

.section-header {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        end;

    margin:
        45px 0 18px;
}


.section-title {

    font-family:
        Georgia,
        serif;

    font-size:
        38px;

    color:
        var(--text) !important;
}


.section-description {

    max-width:
        500px;

    font-size:
        13px;

    line-height:
        1.6;

    color:
        var(--text-secondary) !important;
}


/* ============================================================
   CHART CARD
   ============================================================ */

.chart-card {

    padding:
        10px 14px 5px;

    background:
        var(--surface);

    border:
        1px solid var(--border);

    border-radius:
        18px;
}


/* ============================================================
   SEGMENT CARDS
   ============================================================ */

.segment-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        16px;
}


.segment-card {

    padding:
        24px;

    border:
        1px solid var(--border);

    border-radius:
        18px;

    background:
        var(--surface);

    transition:
        transform .2s ease;
}


.segment-card:hover {

    transform:
        translateY(-4px);
}


.segment-top {

    display:
        flex;

    align-items:
        center;

    gap:
        15px;
}


.pastel-icon {

    width:
        52px;

    height:
        52px;

    border-radius:
        15px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    flex-shrink:
        0;
}


.pastel-icon svg {

    width:
        26px;

    height:
        26px;

    fill:
        none;

    stroke:
        var(--text);

    stroke-width:
        1.5;
}


.segment-label {

    font-size:
        9px;

    letter-spacing:
        1.5px;

    font-weight:
        700;

    color:
        var(--text-muted) !important;
}


.segment-card h3 {

    font-size:
        27px;

    margin:
        2px 0 0;

    color:
        var(--text) !important;
}


.segment-subtitle {

    margin:
        22px 0 8px;

    font-weight:
        600;

    color:
        var(--teal) !important;
}


.segment-card p {

    color:
        var(--text-secondary) !important;

    font-size:
        13px;

    line-height:
        1.65;
}


.segment-arrow {

    display:
        flex;

    align-items:
        center;

    gap:
        7px;

    margin-top:
        22px;

    font-size:
        11px;

    font-weight:
        700;

    color:
        var(--text) !important;
}


.segment-arrow svg {

    width:
        15px;

    height:
        15px;

    fill:
        none;

    stroke:
        currentColor;
}


/* ============================================================
   NEW CUSTOMER
   ============================================================ */

.prediction-card {

    padding:
        30px;

    border:
        1px solid var(--border);

    border-radius:
        20px;

    background:
        linear-gradient(
            135deg,
            var(--green-soft),
            var(--surface)
        );
}


.prediction-label {

    font-size:
        10px;

    letter-spacing:
        1.5px;

    font-weight:
        700;

    color:
        var(--text-muted) !important;
}


.prediction-name {

    font-family:
        Georgia,
        serif;

    font-size:
        44px;

    line-height:
        1;

    margin:
        12px 0;

    color:
        var(--text) !important;
}


.prediction-description {

    color:
        var(--text-secondary) !important;

    line-height:
        1.65;
}


/* ============================================================
   INPUTS
   ============================================================ */

.input-group {

    background:
        var(--surface) !important;

    border:
        1px solid var(--border) !important;

    border-radius:
        13px !important;
}


body.dark-mode input,
body.dark-mode textarea,
body.dark-mode select {

    background:
        #1D2420 !important;

    color:
        #FFFFFF !important;

    border-color:
        #465249 !important;
}


body.dark-mode input::placeholder,
body.dark-mode textarea::placeholder {

    color:
        #FFFFFF !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.primary-btn {

    background:
        var(--deep-green) !important;

    color:
        #FFFFFF !important;

    border:
        none !important;

    border-radius:
        11px !important;

    font-weight:
        700 !important;
}


.primary-btn * {

    color:
        #FFFFFF !important;
}


/* ============================================================
   DARK MODE BUTTON
   ============================================================ */

#theme-toggle {

    margin-top:
        8px !important;

    border:
        1px solid var(--border) !important;

    background:
        var(--surface-2) !important;

    color:
        var(--text) !important;
}


#theme-toggle * {

    color:
        var(--text) !important;
}


body.dark-mode #theme-toggle {

    background:
        #FFFFFF !important;

    color:
        #101412 !important;

    border-color:
        #FFFFFF !important;
}


body.dark-mode #theme-toggle * {

    color:
        #101412 !important;
}


/* ============================================================
   MARKETING TABLE
   ============================================================ */

.playbook-table {

    width:
        100%;

    border-collapse:
        separate;

    border-spacing:
        0;

    overflow:
        hidden;

    border:
        1px solid var(--border);

    border-radius:
        16px;

    background:
        var(--surface);
}


.playbook-table th {

    text-align:
        left;

    padding:
        16px;

    font-size:
        10px;

    letter-spacing:
        1px;

    text-transform:
        uppercase;

    color:
        var(--text-muted) !important;

    border-bottom:
        1px solid var(--border);
}


.playbook-table td {

    padding:
        17px 16px;

    font-size:
        12px;

    line-height:
        1.55;

    color:
        var(--text-secondary) !important;

    border-bottom:
        1px solid var(--border);
}


.playbook-table td:first-child {

    font-family:
        Georgia,
        serif;

    font-size:
        17px;

    color:
        var(--text) !important;
}


/* ============================================================
   CAMPAIGNS
   ============================================================ */

.campaign-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        16px;
}


.campaign-card {

    padding:
        25px;

    border:
        1px solid var(--border);

    border-radius:
        18px;

    background:
        var(--surface);
}


.campaign-number {

    font-size:
        10px;

    letter-spacing:
        1.5px;

    font-weight:
        700;

    color:
        var(--text-muted) !important;
}


.campaign-segment {

    margin-top:
        12px;

    font-size:
        10px;

    font-weight:
        700;

    letter-spacing:
        1px;

    color:
        var(--teal) !important;
}


.campaign-title {

    margin-top:
        7px;

    font-family:
        Georgia,
        serif;

    font-size:
        30px;

    color:
        var(--text) !important;
}


.campaign-copy {

    margin-top:
        10px;

    font-size:
        13px;

    line-height:
        1.6;

    color:
        var(--text-secondary) !important;
}


.campaign-hook {

    margin-top:
        20px;

    padding-top:
        15px;

    border-top:
        1px solid var(--border);
}


.campaign-hook span {

    display:
        block;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        1px;

    color:
        var(--text-muted) !important;
}


.campaign-hook b {

    display:
        block;

    margin-top:
        7px;

    color:
        var(--text) !important;

    line-height:
        1.5;
}


/* ============================================================
   DARK DATAFRAME
   ============================================================ */

body.dark-mode .dataframe {

    background:
        #171C19 !important;

    color:
        #000000 !important;
}


body.dark-mode .dataframe * {

    color:
        #000000 !important;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 900px) {

    .page {

        padding:
            30px 22px 50px;
    }

    .visual-strip {

        grid-template-columns:
            repeat(2, 1fr);
    }

    .kpi-grid {

        grid-template-columns:
            repeat(2, 1fr);
    }

    .segment-grid {

        grid-template-columns:
            1fr;
    }

    .campaign-grid {

        grid-template-columns:
            1fr;
    }

    .hero h1 {

        font-size:
            52px;
    }
}


@media (max-width: 600px) {

    .visual-strip,
    .kpi-grid {

        grid-template-columns:
            1fr;
    }

    .hero h1 {

        font-size:
            45px;
    }
}

"""


# ============================================================
# 17. BUILD GRADIO APP
# ============================================================

with gr.Blocks(
    title="Big Bazaar — Customer Intelligence",
    theme=gr.themes.Base(
        font=[
            "DM Sans",
            "sans-serif"
        ]
    ),
    css=CSS
) as demo:


    # ========================================================
    # SIDEBAR
    # ========================================================

    with gr.Sidebar(
        position="left",
        open=True,
        width=285,
        elem_id="main-sidebar"
    ):

        gr.HTML(
            """
            <div class="brand-block">

                <div class="brand-title">
                    BIG BAZAAR
                </div>

                <div class="brand-subtitle">
                    CUSTOMER INTELLIGENCE
                </div>

                <div class="brand-est">
                    RETAIL ANALYTICS · EST. 2001
                </div>

            </div>
            """
        )


        gr.HTML(
            """
            <div style="
                font-size:10px;
                font-weight:700;
                letter-spacing:1.5px;
                margin-bottom:10px;
            ">
                EXPLORE
            </div>
            """
        )


        # ----------------------------------------------------
        # NAVIGATION BUTTONS
        # ----------------------------------------------------

        nav_overview = gr.Button(
            "⌂   Overview",
            elem_classes="nav-btn"
        )

        nav_segments = gr.Button(
            "◈   Customer Segments",
            elem_classes="nav-btn"
        )

        nav_new = gr.Button(
            "＋   New Customer",
            elem_classes="nav-btn"
        )

        nav_explorer = gr.Button(
            "⌕   Customer Explorer",
            elem_classes="nav-btn"
        )

        nav_model = gr.Button(
            "◫   Model Intelligence",
            elem_classes="nav-btn"
        )

        nav_marketing = gr.Button(
            "↗   Marketing Playbook",
            elem_classes="nav-btn"
        )

        nav_data = gr.Button(
            "▦   Customer Data",
            elem_classes="nav-btn"
        )


        gr.HTML(
            "<div style='height:10px'></div>"
        )


        theme_btn = gr.Button(
            "☾   Dark Mode",
            elem_id="theme-toggle",
            elem_classes="nav-btn"
        )


        gr.HTML(
            """
            <div class="sidebar-footer">

                <div class="sidebar-footer-title">
                    LEGACY × NEXT
                </div>

                <div class="sidebar-footer-text">
                    Turning customer behaviour into
                    sharper retail decisions.
                </div>

            </div>
            """
        )


    # ========================================================
    # PAGE 1 — OVERVIEW
    # ========================================================

    with gr.Column(
        visible=True,
        elem_id="overview-page"
    ) as overview_page:

        gr.HTML(
            f"""
            <div class="page">

                <div class="hero">

                    <div class="eyebrow">
                        <span class="eyebrow-dot"></span>
                        CUSTOMER SEGMENTATION · RETAIL INTELLIGENCE
                    </div>

                    <h1>
                        Know your customer.<br>
                        <em>Grow the relationship.</em>
                    </h1>

                    <div class="hero-copy">

                        Big Bazaar customer behaviour translated into
                        actionable segments using K-Means clustering.
                        The model looks beyond individual purchases to
                        understand patterns in income, spending,
                        frequency and transaction value.

                    </div>

                    <div class="badge-row">

                        <span class="badge green">
                            K-MEANS
                        </span>

                        <span class="badge yellow">
                            500 CUSTOMERS
                        </span>

                        <span class="badge pink">
                            5 FEATURES
                        </span>

                        <span class="badge blue">
                            BEHAVIOUR-LED
                        </span>

                    </div>

                </div>


                <div class="visual-strip">

                    <div class="visual-card green">

                        <div class="icon">
                            {icon_svg("users")}
                        </div>

                        <div class="visual-card-title">
                            People
                        </div>

                        <div class="visual-card-caption">
                            Who shops with us
                        </div>

                    </div>


                    <div class="visual-card yellow">

                        <div class="icon">
                            {icon_svg("chart")}
                        </div>

                        <div class="visual-card-title">
                            Behaviour
                        </div>

                        <div class="visual-card-caption">
                            How they shop
                        </div>

                    </div>


                    <div class="visual-card pink">

                        <div class="icon">
                            {icon_svg("spark")}
                        </div>

                        <div class="visual-card-title">
                            Opportunity
                        </div>

                        <div class="visual-card-caption">
                            Where value exists
                        </div>

                    </div>


                    <div class="visual-card blue">

                        <div class="icon">
                            {icon_svg("arrow")}
                        </div>

                        <div class="visual-card-title">
                            Action
                        </div>

                        <div class="visual-card-caption">
                            What to do next
                        </div>

                    </div>

                </div>

            </div>
            """
        )


        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        with gr.Row(
            elem_classes="kpi-grid"
        ):

            gr.HTML(
                f"""
                <div class="kpi">

                    <div class="kpi-number">
                        {len(df)}
                    </div>

                    <div class="kpi-label">
                        Customers
                    </div>

                </div>
                """
            )


            gr.HTML(
                f"""
                <div class="kpi">

                    <div class="kpi-number">
                        {best_k}
                    </div>

                    <div class="kpi-label">
                        Segments
                    </div>

                </div>
                """
            )


            gr.HTML(
                f"""
                <div class="kpi">

                    <div class="kpi-number">
                        {max(silhouettes):.2f}
                    </div>

                    <div class="kpi-label">
                        Silhouette Score
                    </div>

                </div>
                """
            )


            gr.HTML(
                f"""
                <div class="kpi">

                    <div class="kpi-number">
                        {len(features)}
                    </div>

                    <div class="kpi-label">
                        Features
                    </div>

                </div>
                """
            )


        gr.HTML(
            """
            <div class="page">

                <div class="section-header">

                    <div>

                        <div class="eyebrow">
                            <span class="eyebrow-dot"></span>
                            CUSTOMER MIX
                        </div>

                        <div class="section-title">
                            The shape of the customer base
                        </div>

                    </div>

                    <div class="section-description">
                        Each group represents a distinct behavioural
                        pattern discovered by the clustering model.
                    </div>

                </div>

            </div>
            """
        )


        with gr.Row():

            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=segment_distribution(),
                    show_label=False
                )


            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=spending_chart(),
                    show_label=False
                )


        with gr.Row():

            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=frequency_chart(),
                    show_label=False
                )


            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=transaction_chart(),
                    show_label=False
                )


    # ========================================================
    # PAGE 2 — CUSTOMER SEGMENTS
    # ========================================================

    with gr.Column(
        visible=False
    ) as segments_page:

        gr.HTML(
            """
            <div class="page">

                <div class="eyebrow">
                    <span class="eyebrow-dot"></span>
                    CUSTOMER ARCHITECTURE
                </div>

                <div class="section-title">
                    Meet the four customer groups.
                </div>

                <div class="section-description"
                     style="margin-top:12px;">

                    These are not simply demographic categories.
                    They are behavioural profiles designed to help
                    marketers understand what motivates each group.

                </div>

                <br>

                <div class="segment-grid">

            """
            +

            segment_card(
                "The Inner Circle",
                "High value · High engagement",
                "Customers with strong income, high spending and frequent visits. They represent the most valuable relationship opportunity.",
                "spark",
                GREEN_SOFT
            )

            +

            segment_card(
                "The Untapped",
                "High income · Low engagement",
                "Customers with significant purchasing potential who are not yet engaging with the store frequently enough.",
                "chart",
                BLUE_SOFT
            )

            +

            segment_card(
                "The Regulars",
                "Consistent · Reliable",
                "Customers who maintain a healthy relationship with the brand and visit frequently enough to support loyalty strategies.",
                "users",
                PINK_SOFT
            )

            +

            segment_card(
                "The Deal Hunters",
                "Value driven · Price sensitive",
                "Customers whose behaviour suggests a stronger response to discounts, promotions and value-oriented offers.",
                "plus",
                YELLOW_SOFT
            )

            +

            """
                </div>
            </div>
            """
        )


        with gr.Row():

            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=income_chart(),
                    show_label=False
                )


            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=transaction_chart(),
                    show_label=False
                )


    # ========================================================
    # PAGE 3 — NEW CUSTOMER
    # ========================================================

    with gr.Column(
        visible=False
    ) as new_customer_page:

        gr.HTML(
            """
            <div class="page">

                <div class="eyebrow">
                    <span class="eyebrow-dot"></span>
                    PREDICTIVE CUSTOMER PROFILING
                </div>

                <div class="section-title">
                    Where does a new customer belong?
                </div>

                <div class="section-description"
                     style="margin-top:12px;">

                    Enter the customer's behavioural profile.
                    The trained K-Means model will assign the customer
                    to the closest discovered segment.

                </div>

                <br>

            </div>
            """
        )


        with gr.Row():

            with gr.Column(
                scale=1
            ):

                age_input = gr.Number(
                    label="Age",
                    value=35,
                    elem_classes="input-group"
                )

                income_input = gr.Number(
                    label="Annual Income (₹)",
                    value=60000,
                    elem_classes="input-group"
                )

                spending_input = gr.Number(
                    label="Spending Score",
                    value=60,
                    elem_classes="input-group"
                )

                visits_input = gr.Number(
                    label="Visit Frequency",
                    value=10,
                    elem_classes="input-group"
                )

                transaction_input = gr.Number(
                    label="Average Transaction Value (₹)",
                    value=2000,
                    elem_classes="input-group"
                )

                predict_btn = gr.Button(
                    "Assign Customer Segment  →",
                    variant="primary",
                    elem_classes="primary-btn"
                )


            with gr.Column(
                scale=1
            ):

                prediction_output = gr.HTML(
                    value="""
                    <div class="prediction-card">

                        <div class="prediction-label">
                            READY FOR ANALYSIS
                        </div>

                        <div class="prediction-name">
                            Your customer.
                        </div>

                        <div class="prediction-description">

                            Enter the customer information on the left
                            and let the clustering model identify the
                            closest behavioural segment.

                        </div>

                    </div>
                    """
                )


    # ========================================================
    # PAGE 4 — CUSTOMER EXPLORER
    # ========================================================

    with gr.Column(
        visible=False
    ) as explorer_page:

        gr.HTML(
            """
            <div class="page">

                <div class="eyebrow">
                    <span class="eyebrow-dot"></span>
                    CUSTOMER EXPLORER
                </div>

                <div class="section-title">
                    Explore an individual customer.
                </div>

                <div class="section-description"
                     style="margin-top:12px;">

                    Select any customer to inspect their profile,
                    segment and recommended marketing strategy.

                </div>

                <br>

            </div>
            """
        )


        customer_dropdown = gr.Dropdown(
            choices=df[
                "Customer_ID"
            ].tolist(),

            value=df[
                "Customer_ID"
            ].iloc[0],

            label="Select Customer",

            elem_classes="input-group"
        )


        profile_output = gr.HTML(
            value=customer_profile(
                df[
                    "Customer_ID"
                ].iloc[0]
            )
        )


        gr.Dataframe(
            value=df[
                [
                    "Customer_ID",
                    "Age",
                    "Annual_Income",
                    "Spending_Score",
                    "Visit_Frequency",
                    "Average_Transaction_Value",
                    "Segment"
                ]
            ].head(25),

            interactive=False,

            wrap=True
        )


    # ========================================================
    # PAGE 5 — MODEL INTELLIGENCE
    # ========================================================

    with gr.Column(
        visible=False
    ) as model_page:

        gr.HTML(
            f"""
            <div class="page">

                <div class="eyebrow">
                    <span class="eyebrow-dot"></span>
                    MODEL INTELLIGENCE
                </div>

                <div class="section-title">
                    Why {best_k} segments?
                </div>

                <div class="section-description"
                     style="margin-top:12px;">

                    K-Means clustering was evaluated across multiple
                    values of K. The silhouette score was used alongside
                    the elbow method to identify a useful segmentation.

                </div>

                <br>

                <div class="badge-row">

                    <span class="badge green">
                        STANDARD SCALER
                    </span>

                    <span class="badge yellow">
                        K-MEANS
                    </span>

                    <span class="badge pink">
                        K = 2 → 8
                    </span>

                    <span class="badge blue">
                        RANDOM STATE = 42
                    </span>

                </div>

            </div>
            """
        )


        with gr.Row():

            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=elbow_chart(),
                    show_label=False
                )


            with gr.Column(
                elem_classes="chart-card"
            ):

                gr.Plot(
                    value=silhouette_chart(),
                    show_label=False
                )


        gr.HTML(
            f"""
            <div class="page">

                <div class="section-header">

                    <div class="section-title">
                        Evaluation summary
                    </div>

                </div>


                <div class="segment-grid">

                    <div class="segment-card">

                        <div class="segment-top">

                            {pastel_icon(
                                "chart",
                                GREEN_SOFT
                            )}

                            <div>

                                <div class="segment-label">
                                    SILHOUETTE
                                </div>

                                <h3>
                                    {max(silhouettes):.3f}
                                </h3>

                            </div>

                        </div>

                        <p>
                            Higher silhouette values indicate that
                            customers are reasonably separated from
                            neighbouring clusters.
                        </p>

                    </div>


                    <div class="segment-card">

                        <div class="segment-top">

                            {pastel_icon(
                                "spark",
                                BLUE_SOFT
                            )}

                            <div>

                                <div class="segment-label">
                                    SELECTED K
                                </div>

                                <h3>
                                    {best_k}
                                </h3>

                            </div>

                        </div>

                        <p>
                            The selected number of clusters provides
                            the strongest silhouette score among the
                            evaluated K values.
                        </p>

                    </div>

                </div>

            </div>
            """
        )


    # ========================================================
    # PAGE 6 — MARKETING PLAYBOOK
    # ========================================================

    with gr.Column(
        visible=False
    ) as marketing_page:

        gr.HTML(
            """
            <div class="page">

                <div class="eyebrow">
                    <span class="eyebrow-dot"></span>
                    FROM SEGMENTS TO ACTION
                </div>

                <div class="section-title">
                    The marketing playbook.
                </div>

                <div class="section-description"
                     style="margin-top:12px;">

                    Segmentation only becomes useful when it changes
                    the decision being made. Each group therefore
                    receives a different strategic response.

                </div>

                <br>

            """
            +

            campaign_cards()

            +

            """
            </div>
            """
        )


        gr.HTML(
            """
            <div class="page">

                <table class="playbook-table">

                    <thead>

                        <tr>

                            <th>
                                Customer Group
                            </th>

                            <th>
                                Primary Behaviour
                            </th>

                            <th>
                                Recommended Strategy
                            </th>

                        </tr>

                    </thead>


                    <tbody>

                        <tr>

                            <td>
                                The Inner Circle
                            </td>

                            <td>
                                High income, high spending,
                                high engagement.
                            </td>

                            <td>
                                Premium rewards, early access,
                                exclusive launches and VIP experiences.
                            </td>

                        </tr>


                        <tr>

                            <td>
                                The Untapped
                            </td>

                            <td>
                                High income but comparatively
                                low engagement.
                            </td>

                            <td>
                                Personalised recommendations,
                                high-value bundles and re-engagement.
                            </td>

                        </tr>


                        <tr>

                            <td>
                                The Regulars
                            </td>

                            <td>
                                Consistent visits and healthy
                                spending behaviour.
                            </td>

                            <td>
                                Loyalty rewards, frequency incentives
                                and subscription-style offers.
                            </td>

                        </tr>


                        <tr>

                            <td>
                                The Deal Hunters
                            </td>

                            <td>
                                Lower spending and high
                                price sensitivity.
                            </td>

                            <td>
                                Flash deals, coupons, value bundles
                                and promotional campaigns.
                            </td>

                        </tr>

                    </tbody>

                </table>

            </div>
            """
        )


    # ========================================================
    # PAGE 7 — CUSTOMER DATA
    # ========================================================

    with gr.Column(
        visible=False
    ) as data_page:

        gr.HTML(
            """
            <div class="page">

                <div class="eyebrow">
                    <span class="eyebrow-dot"></span>
                    DATA FOUNDATION
                </div>

                <div class="section-title">
                    Customer dataset.
                </div>

                <div class="section-description"
                     style="margin-top:12px;">

                    The modelling dataset used for preprocessing,
                    clustering and customer profiling.

                </div>

                <br>

            </div>
            """
        )


        gr.Dataframe(
            value=df,
            interactive=False,
            wrap=True
        )


        gr.DownloadButton(
            "Download Customer Dataset ↓",
            value=output_file,
            variant="primary",
            elem_classes="primary-btn"
        )


    # ========================================================
    # IMPORTANT — PAGE LIST
    #
    # This is the FIX for sidebar navigation.
    #
    # We directly update the visibility of the seven
    # actual Gradio Column components.
    # ========================================================

    pages = [
        overview_page,
        segments_page,
        new_customer_page,
        explorer_page,
        model_page,
        marketing_page,
        data_page
    ]


    # ========================================================
    # NAVIGATION FUNCTION
    # ========================================================

    def show_page(index):

        return [
            gr.Column(
                visible=(i == index)
            )

            for i in range(
                len(pages)
            )
        ]


    # ========================================================
    # SIDEBAR NAVIGATION EVENTS
    # ========================================================

    nav_overview.click(
        fn=lambda: show_page(0),
        inputs=None,
        outputs=pages
    )


    nav_segments.click(
        fn=lambda: show_page(1),
        inputs=None,
        outputs=pages
    )


    nav_new.click(
        fn=lambda: show_page(2),
        inputs=None,
        outputs=pages
    )


    nav_explorer.click(
        fn=lambda: show_page(3),
        inputs=None,
        outputs=pages
    )


    nav_model.click(
        fn=lambda: show_page(4),
        inputs=None,
        outputs=pages
    )


    nav_marketing.click(
        fn=lambda: show_page(5),
        inputs=None,
        outputs=pages
    )


    nav_data.click(
        fn=lambda: show_page(6),
        inputs=None,
        outputs=pages
    )


    # ========================================================
    # NEW CUSTOMER PREDICTION EVENT
    # ========================================================

    predict_btn.click(
        fn=predict_customer,

        inputs=[
            age_input,
            income_input,
            spending_input,
            visits_input,
            transaction_input
        ],

        outputs=prediction_output
    )


    # ========================================================
    # CUSTOMER EXPLORER EVENT
    # ========================================================

    customer_dropdown.change(
        fn=customer_profile,

        inputs=customer_dropdown,

        outputs=profile_output
    )


    # ========================================================
    # DARK MODE
    #
    # ONLY changes the body class.
    # It does NOT control page navigation.
    # ========================================================

    theme_btn.click(
        fn=None,
        inputs=None,
        outputs=None,

        js="""
        () => {

            document.body.classList.toggle(
                "dark-mode"
            );

            const button =
                document.querySelector(
                    "#theme-toggle"
                );

            if (!button) {
                return;
            }

            if (
                document.body.classList.contains(
                    "dark-mode"
                )
            ) {

                button.innerText =
                    "☀   Light Mode";

            } else {

                button.innerText =
                    "☾   Dark Mode";
            }
        }
        """
    )


# ============================================================
# 18. LAUNCH
# ============================================================

demo.launch(
    share=True,
    debug=True
)
