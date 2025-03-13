import dash
from dash import html, dcc, callback, Output, Input, State, ctx
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
from scipy.stats import gaussian_kde, mannwhitneyu, chi2_contingency, ttest_ind
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, f_oneway

# Load your datasets
raw_df = pd.read_csv("exoplanet_dataset.csv")
cleaned_df = pd.read_csv("cleaned_dataset.csv")  

# =================================================== HOME SECTION ===============================================================
# Calculate the counts
total_habitable_dataset = cleaned_df[cleaned_df["P_HABITABLE"] == 1].shape[0]

filtered_habitable = cleaned_df[
    (cleaned_df["P_TYPE"].isin(["Terran", "Superterran"])) &
    (cleaned_df["P_SEMI_MAJOR_AXIS"].between(0.5, 2)) &
    (cleaned_df["P_FLUX"] < 3) &
    (cleaned_df["S_METALLICITY"].between(-0.2, 0.2)) &
    (cleaned_df["P_ECCENTRICITY"] < 0.53)
].shape[0]

# Create the bar chart
fig_count = go.Figure()

fig_count.add_trace(go.Bar(
    x=["Dataset Habitability", "Filtered Habitability"],
    y=[total_habitable_dataset, filtered_habitable],
    marker=dict(color=["blue", "green"]),
    text=[total_habitable_dataset, filtered_habitable],
    textposition="auto"
))

# Update layout
fig_count.update_layout(
    title="Comparison of Habitability Classification",
    xaxis_title="Classification Type",
    yaxis_title="Number of Planets",
    template="plotly_white"
)

import plotly.graph_objects as go

# Calculate the counts
total_exoplanets = cleaned_df.shape[0]  # Total exoplanets in dataset
total_habitable_dataset = cleaned_df[cleaned_df["P_HABITABLE"] == 1].shape[0]

filtered_habitable = cleaned_df[
    (cleaned_df["P_TYPE"].isin(["Terran", "Superterran"])) &
    (cleaned_df["P_SEMI_MAJOR_AXIS"].between(0.5, 2)) &
    (cleaned_df["P_FLUX"] < 3) &
    (cleaned_df["S_METALLICITY"].between(-0.2, 0.2)) &
    (cleaned_df["P_ECCENTRICITY"] < 0.53)
].shape[0]

# Data for pie chart
labels = ["Dataset Habitable", "Filtered Habitable", "Non-Habitable"]
values = [total_habitable_dataset, filtered_habitable, total_exoplanets - total_habitable_dataset]

# Create Pie Chart with better readability
fig_pie = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values, 
    hole=0.3, 
    marker=dict(colors=["blue", "green", "gray"]),
    textinfo="label+percent",
    textposition="inside",  # Moves labels outside
    pull=[0.05, 0.05, 0]  # Slightly "explode" first two slices for clarity
)])

# Update layout to prevent overlap
fig_pie.update_layout(
    title=dict(text="Habitability Classification: Dataset vs. Filtered Criteria", 
               font=dict(size=16), 
               y=0.95, x=0.5, xanchor="center", yanchor="top"),  # Move title up
    legend=dict(
        x=1, y=0.5,  # Move legend to the right
        xanchor="left", yanchor="middle",
        bgcolor="rgba(255,255,255,0.7)"
    ),
    margin=dict(t=80, b=40, l=40, r=40),  # Adjust margins
    template="plotly_white"
)
# =================================================== DATA SUMMARY SECTION ===============================================================
# Count missing values per column
raw_missing = raw_df.isnull().sum()
cleaned_missing = cleaned_df.isnull().sum()

# Compute number of filled values (previously missing but now present)
filled_values = raw_missing - cleaned_missing
filled_values[filled_values < 0] = 0  # Avoid negative values

# Columns removed in cleaned data
removed_columns = set(raw_df.columns) - set(cleaned_df.columns)

# Figure 1: Raw Data Missing Values (Vertical Bars)
fig_raw = go.Figure()
fig_raw.add_trace(go.Bar(
    x=raw_df.columns,
    y=raw_missing,
    name="Missing (Raw)",
    marker=dict(color='red', opacity=0.7)
))
fig_raw.update_layout(
    title=dict(text="Raw Data: Missing Values", x=0.5),
    xaxis_title="Columns",
    yaxis_title="Number of Missing Values",
    template="plotly_white"
)

# Figure 2: Cleaned Data - Filled & Removed Columns (Vertical Bars)
fig_cleaned = go.Figure()
fig_cleaned.add_trace(go.Bar(
    x=cleaned_df.columns,
    y=filled_values,
    name="Filled Values",
    marker=dict(color='green', opacity=0.7)
))
if removed_columns:
    fig_cleaned.add_trace(go.Scatter(
        x=list(removed_columns),
        y=[0] * len(removed_columns),
        mode='markers',
        marker=dict(color='black', size=10, symbol='x'),
        name="Removed Columns"
    ))
fig_cleaned.update_layout(
    title=dict(text="Cleaned Data: Filled & Removed Columns", x=0.5), 
    xaxis_title="Columns",
    yaxis_title="Number of Filled Values",
    template="plotly_white"
)

# Descriptions for each data column (add description here if you add any new column)
descriptions = {
    "P_YEAR": "Year of discovery of exoplanet. The largest number of discoveries were made in 2016,due to the release of NASAs Kelpler scape telescope data.Instead of validating using each exoplanet separately,they used new statistical validation method using VSEPAQ (validaton of exoplanet signals using prababiistic approximation) to analyse all the candidaes at once. So 1,284 new planets were discovered at one go.",
    "P_MASS": "The mass of exoplanet. Detected using two main methods: Radial and Transit Timing Variation (TTV). In the radia method the thug the planet gives it host star due to its gravity,casues the light from the star gets blue shifted or red shifted and this is used to detect its mass. In TTV we discover hidden planets too. The know planets gravity affects the transiting planets time to vary. This variation is used to detect its mass.",
    "P_RADIUS": "Radius of the exoplanet. Radius is measured using the transit method,where the dip in the host stars luminosity depends on the relative size of the transiting exoplanet. Bigger planet,more dip and vice versa.",
    "P_PERIOD": "Orbital period of the exoplanet, representing the time it takes to complete one orbit around its host star. Usually detected using the transit method measuring the difference between the two dips(min two transits required). Used to calculate distance of planet from its star using Kepler's third law(habitability and conditions for water).",
    "P_SEMI_MAJOR_AXIS": "Semi-major axis of the exoplanet’s orbit, representing its average distance from the host star. Calculated sing Kepler's third law (already know period and mass of star).Important o calculate the habitability of an exoplanet.",
    "P_ECCENTRICITY": "Orbital eccentricity of the exoplanet, describing the deviation of its orbit from a perfect circle. The most common detection method is by radial velocity. If the path is circular,the RV graph is smooth and symetric. If it is elliplticalR curve is asymmmtric becuse sarmoves faster when lanet is near and slower when planet s farther. Annalysing this RV curve, we ge the ecentricity.",
    "P_INCLINATION": "Inclination of the exoplanet’s orbital plane relative to the line of sight from Earth. Face on (i=0) and edge on(i=90).How centred the transition is the RV depends on mass of planet and inclination. So it plays a major role in analysis.",
    "S_RA": "Right ascension of the host star, indicating its position in the sky.Gives the \"longitude\" of star in the sky. Measured in hours,minutes,seconds (equatorial coordinate system).Measured using star catalogue(gaia).",
    "S_DEC": "Declination of the host star, indicating its position in the sky.Gives the \"latitude\" of star in the sky. Measured in degrees (equatorial coordinate system).Measured sing star catalogue(gaia).",
    "S_MAG": "Apparent magnitude of the host star, indicating its brightness as observed from Earth. The telescope measures the brightness in different wavelengths, passed through filters. Then it s converted to apparent magnitude(-2.5 log(F))",
    "S_DISTANCE": "Distance of the host star from Earth.Stellar parallax measures a star’s apparent shift against distant background stars as Earth orbits the Sun. NASA's Gaia spacecraft precisely tracks this tiny angular displacement and calculates the stars distance.",
    "S_TEMPERATURE": "Surface temperature of the host star, typically measured in Kelvin.Spectroscopy analyzes a star’s light by splitting it into a spectrum of colors.By measuring the peak wavelength, astronomers calculate the star’s temperature (Wien’s Law). If luminosity and radius are known we can also use Stphan-Boltzmans Law.",
    "S_MASS": "Mass of the host star relative to the Sun's mass.Using Stellar Models & Spectroscopy, NASA analyzes a star’s spectrum to determine its spectral type, luminosity, and temperature. Using the mass-luminosity relation (L proportional to M^3.5), the star’s mass is estimated. Alternatively,Exoplanet Orbital Motion ,If an exoplanet is detected via radial velocity, the star’s mass is inferred from the gravitational \"wobble\" using Kepler’s third law and the planet’s orbital period.",
    "S_RADIUS": "Radius of the host star relative to the Sun's radius.When an exoplanet transits (passes in front of) its host star, it temporarily blocks some of the star’s light. The dip in brightness is related to the star and planet’s radii.",
    "S_METALLICITY": "Metal content of the host star, indicating the abundance of elements heavier than hydrogen and helium.Spectral line analysis studies the dark absorption lines in a star’s spectrum, which correspond to specific elements in its atmosphere. NASA’s telescopes split starlight into a spectrum.This reveals the star’s chemical composition and metallicity.",
    "S_AGE": "Estimated age of the host star.NASA estimates a star’s age by comparing its luminosity, temperature, and spectral type to theoretical stellar evolution models. These models track how stars change over time on the Hertzsprung-Russell (H-R) diagram and determines its age.",
    "S_LOG_LUM": "Logarithmic luminosity of the host star relative to the Sun.NASA calculates a star’s luminosity by measuring its apparent brightness and distance (via parallax). Using the Inverse Square Law, the absolute magnitude is determined and converted to luminosity.",
    "S_LOG_G": "Logarithm of the surface gravity of the host star.NASA determines a star’s surface gravity using its mass and radius by the formula g=GM/R^2",
    "P_ESCAPE": "Escape velocity of the exoplanet, describing the speed required to overcome its gravitational pull.NASA calculates an exoplanet’s escape velocity using its mass and radius using the formula v=sqrt(2GM/R).",
    "P_POTENTIAL": "Gravitational potential on the surface of the exoplanet.NASA calculates an exoplanet’s gravitational potential at its surface using: -GM/R",
    "P_GRAVITY": "Surface gravity of the exoplanet.NASA determines an exoplanet’s surface gravity with the formula: GM/R^2.",
    "P_DENSITY": "Density of the exoplanet. Calculated using mass and radius(volume).Helps understand the internal composition.",
    "P_HILL_SPHERE": "Hill sphere radius of the exoplanet, representing the region of space where its gravity dominates over the host star's.Calculate using formula: a(Mp/3*Ms)^(1/3).Helps to analyse gravitational interactions and its ability to hold moons and rings.",
    "P_DISTANCE": "Distance of the exoplanet from Earth. Calculated indirectlyby calculating the distance to the host star and is considered approximately the same.",
    "P_PERIASTRON": "Closest distance of the exoplanet to its host star during its orbit.NASA calculates it using the semi-major axis an orbital eccentricity with: a(1-e). Affects the temperature variations on the exoplanet.",
    "P_APASTRON": "Farthest distance of the exoplanet from its host star during its orbit.NASA calculates it using the semi-major axis an orbital eccentricity with: a(1+e). Affects the tidalforces ad orbital stability on the exoplanet.",
    "P_DISTANCE_EFF": "Effective Parallax Distance, a refined stellar distance estimate incorporating corrections for parallax biases and uncertainties. It provides a more reliable distance than a simple inverse parallax, often used in Gaia-based catalogs.",
    "P_FLUX": "Stellar flux received by the exoplanet.NASA calculates it using the luminosity of the star and the distance of the exoplanet from the star with: L/4*pi*d^2.",
    "P_TEMP_EQUIL": "Temperature of the exoplanet, assuming no atmosphere. Calculated using temperature ,radius ,semi major axis and bond albedo.Provides clues about atmospheric composition.",
    "S_LUMINOSITY": "Luminosity of the host star relative to the Sun.NASA calculates a star’s luminosity by measuring its apparent brightness and distance (via parallax). Using the Inverse Square Law, the absolute magnitude is determined and converted to luminosity.", 
    "S_SNOW_LINE": "Distance from the host star where temperatures allow volatile compounds to condense into solid ice grains.Estimated using : 2.7*(Lhs/Ls)^1/2.Determines where gas giants and icy bodies form.",
    "S_ABIO_ZONE": "Region around the host star where abiogenesis (origin of life) is possible.",
    "S_TIDAL_LOCK": "Indicator of whether the host star exerts tidal locking on its orbiting exoplanets.Tidal locking occurs when an exoplanet’s rotation period synchronizes with its orbital period, meaning the same side always faces its star.Key Factors Affecting: Distance from Star,Star’s Mass.Affects climate and habitability.",
    "P_HABZONE_OPT": "Optimal habitable zone distance for the exoplanet, where conditions for life are most favorable.",
    "P_HABZONE_CON": "Optimal habitable zone distance for the exoplanet, where conditions for life are most favorable.",
    "P_HABITABLE": "Binary indicator (0 or 1) of whether the exoplanet is considered potentially habitable."
}

# =================================================== DATA ANALYSIS SECTION ===============================================================
# Define hypothesis number and deescription accordingly here
hypothesis = pd.DataFrame({
    "Parameter": [
        "Hypothesis-1", "Hypothesis-2", "Hypothesis-3", "Hypothesis-4", "Hypothesis-5",
        "Hypothesis-6", "Hypothesis-7", "Hypothesis-8", "Hypothesis-9", "Hypothesis-10",
        "Hypothesis-11", "Hypothesis-12",
    ],
    "Value": [
        "The relationship between mass and radius of exoplanets can influence their potential habitability.",
        "The composition of habitable exoplanets plays a crucial role in determining their potential to support life.",
        "The presence of water on exoplanets depends on equilibrium temperature ranges suitable for habitability.",
        "Exoplanets with orbital periods between 50-500 days are most suitable for habitability due to optimal energy reception.",
        "The semi-major axis and Hill sphere influence habitability by determining orbital stability and gravitational dominance.",
        "Atmospheric retention, influenced by escape velocity, is crucial for maintaining surface conditions suitable for habitability.",
        "Orbital Distance and equilibrium temperature jointly affect habitability by regulating climate stability and surface conditions.",
        "Orbital period and tidal locking impact habitability by influencing temperature and atmospheric dynamics on exoplanets.",
        "The metallicity of host stars differs for terrestrial and non-terrestrial exoplanets, influencing their formation and composition.",
        "Planetary eccentricity affects surface temperature by causing variations in stellar energy received throughout the orbit.",
        "Stellar flux directly influences the surface temperature of exoplanets, affecting their potential habitability.",
        "Higher orbital eccentricity negatively impacts planetary habitability by causing extreme temperature variations."
    ],
})

# Function to create a clickable card (nothing to modify)
def create_card(param, value, idx):
    return html.Div(
        dbc.Card(
            dbc.CardBody([
                html.H5(param, className="card-title"),
                html.P(f"{value}", className="card-text"),
            ]),
            className="shadow-sm text-center",
            style={
                "width": "250px",
                "margin": "10px",
                "cursor": "pointer",
                "transition": "0.3s",
                "border": "1px solid #ddd",
            },
        ),
        id={"type": "open-modal", "index": idx},
        n_clicks=0
    )

# Layout of graph and description section (nothing to modify)
modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Parameter Details")),
        dbc.ModalBody(
            dbc.Row([
                dbc.Col(dcc.Graph(id="graph"), width=8),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Description", className="card-title"),
                            html.P(id="description", className="card-text"),
                        ]),
                        className="shadow-sm",
                        style={"backgroundColor": "#f8f9fa"},
                    ),
                    width=4
                ),
            ])
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ml-auto", n_clicks=0)
        ),
    ],
    id="modal",
    size="xl",
    is_open=False,
)

# ========================================== CONCLUSION SECTION ===============================================================
def generate_graph(step):
    if step == 0:
        # Count habitable planets per type
        habitable_counts = cleaned_df.groupby("P_TYPE")["P_HABITABLE"].mean().reset_index()

        # Plot with Plotly
        fig = px.bar(
            habitable_counts, 
            x="P_TYPE", 
            y="P_HABITABLE", 
            color_discrete_sequence=["teal"], 
            text_auto=True
        )

        # Customize layout
        fig.update_layout(
            title="Habitability by Planet Type",
            xaxis_title="Planet Type",
            yaxis_title="Fraction of Habitable Planets",
            xaxis_tickangle=45
        )

        text = """
            ## **1. Habitability by Planet Type**
            - The fraction of habitable planets varies by planet type, with **terrestrial planets** (rocky planets similar to Earth) having the highest likelihood of habitability.  
            - **Iron-rich planets** (dense rocky planets) are more favorable for habitability than water worlds, gas giants, or ice giants. This aligns with the fact that:
            - Rocky planets provide a **solid surface** for liquid water accumulation.
            - Iron-rich planets have **high core activity**, potentially generating a **magnetic field** that shields against stellar radiation.
            - **Super-Earths** (if categorized separately) may show a **high habitability fraction** due to their ability to **retain a thick atmosphere** and moderate climate conditions.  
            - Gas giants and sub-Neptunes have nearly **zero habitability** because they lack a **solid surface** and have **hydrogen-helium dominated atmospheres**, making them inhospitable.
        """

    elif step == 1:
        # Data for habitable and non-habitable planets
        habitable_sma = cleaned_df[cleaned_df["P_HABITABLE"] == 1]["P_SEMI_MAJOR_AXIS"].dropna()
        non_habitable_sma = cleaned_df[cleaned_df["P_HABITABLE"] == 0]["P_SEMI_MAJOR_AXIS"].dropna()

        # Create histogram with Plotly
        fig = ff.create_distplot(
            [habitable_sma, non_habitable_sma], 
            group_labels=["Habitable", "Non-Habitable"],
            colors=["green", "red"],
            show_hist=True, 
            show_rug=False
        )

        # Add shaded optimal range
        fig.add_vrect(
            x0=0.1, x1=2, 
            fillcolor="yellow", 
            opacity=0.3, 
            layer="below", 
            line_width=0,
            annotation_text="Optimal Range",
            annotation_position="top"
        )

        # Update layout
        fig.update_layout(
            title="Semi-Major Axis Distribution for Habitability",
            xaxis_title="Semi-Major Axis (AU)",
            yaxis_title="Density",
            legend_title="Planet Type"
        )

        text = """
            ## **2. Semi-Major Axis Distribution for Habitability**
            - The **habitable zone (HZ)**, often defined as **0.5 - 2 AU**, is where most habitable planets are concentrated.
            - Within this range, planets receive **moderate stellar radiation**, increasing the chance of sustaining liquid water.
            - The histogram likely peaks around **1 AU**, aligning with Earth's position in the Solar System.
            - Non-habitable planets have a **broader distribution**, spanning **hot Jupiters (very close-in planets)** to **cold gas giants**.
            - The **yellow-shaded optimal range** highlights the **conservative habitable zone**, beyond which planets become:
            - **Too hot** → leading to atmospheric loss and runaway greenhouse effects.
            - **Too cold** → resulting in frozen surface water.
            - **Outliers in the habitable category (beyond 2 AU) may indicate:**
            - **Thick atmospheres** trapping heat.
            - **Subsurface oceans** beneath ice layers (e.g., Europa-like planets).
        """

    elif step == 2:
        # Prepare data for box plot
        eccentricity_data = cleaned_df[['P_HABITABLE', 'P_ECCENTRICITY']].dropna()
        eccentricity_data['P_HABITABLE'] = eccentricity_data['P_HABITABLE'].map({1: 'Habitable', 0: 'Non-Habitable'})

        # Create box plot
        fig = px.box(
            eccentricity_data, 
            x='P_HABITABLE', 
            y='P_ECCENTRICITY', 
            color='P_HABITABLE',
            labels={"P_HABITABLE": "Habitability", "P_ECCENTRICITY": "Eccentricity"},
            title="Eccentricity vs Habitability",
            color_discrete_map={"Habitable": "green", "Non-Habitable": "red"}
        )

        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="lightgray")
        )

        text = """
            ## **3. Eccentricity vs Habitability**
            - **Lower eccentricities are preferred for habitability**, as high eccentricity introduces **large seasonal variations**, which can be **detrimental to climate stability**.
            - The box plot likely shows:
            - **Habitable planets** clustering around **lower eccentricity values** (close to circular orbits).
            - **Non-habitable planets** displaying **wider eccentricity ranges**, including extreme cases.
            - **Why is high eccentricity bad?**
            - High eccentricity planets experience **intense temperature fluctuations**.
            - At **perihelion** (closest approach to the star), extreme heating may **evaporate surface water**.
            - At **aphelion** (farthest distance), the planet might cool down significantly, **freezing liquid reservoirs**.
            - These **rapid climate shifts** reduce habitability by making conditions unstable.
            - **Exceptions:**
            - **Tidally heated exoplanets** may maintain liquid oceans despite eccentric orbits.
            - Planets with **massive atmospheres** can redistribute heat, reducing temperature swings.
        """

    elif step == 3:
        # Data for habitable and non-habitable planets
        habitable_density = cleaned_df[cleaned_df["P_HABITABLE"] == 1]["P_DENSITY"].dropna()
        non_habitable_density = cleaned_df[cleaned_df["P_HABITABLE"] == 0]["P_DENSITY"].dropna()

        # Create histogram with Plotly
        fig = ff.create_distplot(
            [habitable_density, non_habitable_density], 
            group_labels=["Habitable", "Non-Habitable"],
            colors=["green", "red"],
            show_hist=True, 
            show_rug=False
        )

        # Update layout
        fig.update_layout(
            title="Density Distribution for Habitability",
            xaxis_title="Planet Density (g/cm³)",
            yaxis_title="Density",
            legend_title="Planet Type"
        )

        text = """
            ## **4. Density Distribution for Habitability**
            - **Density is a strong indicator of planetary composition:**
            - **Rocky planets** → Moderate to high densities (~5-8 g/cm³).
            - **Gas & ice giants** → Low densities (~0.5-2 g/cm³).
            - The histogram suggests **most habitable planets have densities between 4-7 g/cm³**, consistent with an Earth-like rocky composition.
            - **Why is high density favorable?**
            - High-density planets retain a **metallic core**, which can generate a **protective magnetic field**.
            - Denser planets **hold onto their atmosphere**, preventing the loss of essential gases like oxygen and nitrogen.
            - **Why are low-density planets less habitable?**
            - Low-density planets are typically **mini-Neptunes or gas-rich worlds**.
            - Their thick hydrogen-helium atmospheres create **extreme pressures and temperatures**, making them inhospitable.
            - **Outliers:** Some **super-Earths** might be habitable despite **higher-than-Earth densities**, provided they have a suitable atmosphere.
        """

    elif step == 4:
        # Data preparation
        habitable_flux = cleaned_df.loc[cleaned_df["P_HABITABLE"] == 1, "P_FLUX"].dropna().values
        non_habitable_flux = cleaned_df.loc[cleaned_df["P_HABITABLE"] == 0, "P_FLUX"].dropna().values

        # Create figure
        fig = go.Figure()

        # Add histogram for habitable planets
        fig.add_trace(go.Histogram(
            x=habitable_flux, 
            name="Habitable", 
            opacity=0.6, 
            marker_color="blue"
        ))

        # Add histogram for non-habitable planets
        fig.add_trace(go.Histogram(
            x=non_habitable_flux, 
            name="Non-Habitable", 
            opacity=0.6, 
            marker_color="orange"
        ))

        # Add flux threshold line
        fig.add_vline(
            x=3, 
            line=dict(color="black", dash="dash"), 
            annotation_text="Flux Threshold"
        )

        # Update layout with visible axes
        fig.update_layout(
            barmode="overlay",  # Overlays both histograms
            title="Stellar Flux Distribution for Habitability",
            xaxis=dict(
                title="Stellar Flux",
                showline=True,
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                title="Number of Planets",
                showline=True,
                showgrid=True,
                zeroline=True
            ),
            legend_title="Planet Type",
            plot_bgcolor="white"
        )


        text = """
            ## **5. Stellar Flux Distribution for Habitability**
            - **Stellar flux (radiation received from the star) is critical for surface temperature and water retention.**
            - The histogram shows a **sharp decline in habitability beyond a flux level of ~3**, marked by the **black dashed threshold line**.
            - **Why does habitability drop beyond this point?**
            - Excessive stellar radiation leads to:
                - **Runaway greenhouse effects** (e.g., Venus-like conditions).
                - **Atmospheric evaporation** due to extreme heating.
                - **Increased UV exposure**, which can sterilize planetary surfaces.
            - The **majority of habitable planets fall below this flux threshold**, meaning they receive just enough radiation to sustain liquid water without triggering extreme climate effects.
            - Some **high-flux planets** may remain habitable if they have **thick reflective cloud cover** to regulate surface temperatures.
            - Some **low-flux planets** might rely on **internal heating** or **greenhouse effects** to maintain habitable conditions.
        """

    else:
        # Data preparation
        metallicity_data = cleaned_df[["P_HABITABLE", "S_METALLICITY"]].dropna()
        metallicity_data["P_HABITABLE"] = metallicity_data["P_HABITABLE"].map({1: "Habitable", 0: "Non-Habitable"})

        # Create figure
        fig = go.Figure()

        # Add box plot for habitable planets
        fig.add_trace(go.Box(
            y=metallicity_data.loc[metallicity_data["P_HABITABLE"] == "Habitable", "S_METALLICITY"],
            name="Habitable",
            marker_color="green",
            boxmean=True  # Show mean line
        ))

        # Add box plot for non-habitable planets
        fig.add_trace(go.Box(
            y=metallicity_data.loc[metallicity_data["P_HABITABLE"] == "Non-Habitable", "S_METALLICITY"],
            name="Non-Habitable",
            marker_color="red",
            boxmean=True
        ))

        # Update layout with visible axes
        fig.update_layout(
            title="Host Star Metallicity vs Habitability",
            xaxis=dict(
                title="Habitability",
                showline=True,
                showgrid=True,
                zeroline=True
            ),
            yaxis=dict(
                title="Stellar Metallicity",
                showline=True,
                showgrid=True,
                zeroline=True
            ),
            plot_bgcolor="white"
        )
                
        text = """
            ## **6. Host Star Metallicity vs Habitability**
            - **Host star metallicity influences planet formation.**
            - The **mean metallicity of terrestrial planet hosts is -0.028**, indicating that planets around **low-metallicity stars are not necessarily disadvantaged**.
            - **Negative correlation (-0.124) between host star metallicity and terrestrial planet fraction** suggests:
            - **Metal-poor stars may host more terrestrial planets** than metal-rich stars.
            - This could be because:
                - **High-metallicity environments** favor the formation of gas giants over small rocky planets.
                - **Low-metallicity stars** may still form Earth-like planets, but with fewer heavy elements.
            - **Why isn’t high metallicity strictly necessary?**
            - While metallicity aids planet formation, **too much metallicity favors gas giants**, reducing the number of small, rocky habitable planets.
            - Observations suggest even **sub-solar metallicity stars can host habitable planets**.
            - Some **high-metallicity stars still host habitable planets**, but they are rarer due to the preference for **gas giant formation** in such environments.
        """

    return fig, dcc.Markdown(text, dangerously_allow_html=True)
# =================================================== DASHBOARD STRUCTURE ===============================================================
# Initialize Dash
app = dash.Dash(__name__, external_stylesheets=["/assets/style.css", dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    html.Nav(className="navbar", children=[html.A(label, href=f"#{label.lower()}", className="nav-link") 
                                        for label in ["Home", "Summary", "Analysis", "Conclusion"]]),
    html.Div(id="progress-bar", className="progress-bar"),
    
    # Home Section
    html.Section(id="home", children=[
        html.Div(className="header-container", children=[
            html.H1("✨ Stellar Analysis", className="title"),
            html.Small("by Vibhashree Vasuki", className="subtitle"),
            html.P("Exploring the possibility of life on distant planets.", className="description")
        ]),
        html.Div(className="graph-section", children=[
            html.Div(className="text-section", children=[
                html.H2("📊 Visualizing Stellar Data", className="section-title"),
                html.P("Explore the relationship between stellar properties using interactive graphs.", className="section-description")
            ]),
            html.Div(className="graph-container", children=[
                dcc.Graph(id="stellar-bar-chart-1", figure=fig_count, className="graph"),

                html.Div(className="text-section", children=[
                    html.H3("Key Factors Influencing Exoplanet Habitability"),
                    
                    html.P("Our analysis identifies key planetary, orbital, and stellar properties affecting habitability. "
                        "Below is a summary of the most significant findings:"),

                    html.Ul(children=[
                        html.Li([
                            html.Strong("Planetary Composition: "),
                            "Earth-like rocky planets with moderate density are more habitable than gas-rich ones."
                        ]),
                        html.Li([
                            html.Strong("Orbital Position: "),
                            "Planets in the 0.5–2 AU range from their star have higher habitability potential."
                        ]),
                        html.Li([
                            html.Strong("Eccentricity & Climate Stability: "),
                            "Lower eccentricity ensures stable temperatures, favoring habitability."
                        ]),
                        html.Li([
                            html.Strong("Stellar Flux & Radiation: "),
                            "Habitability declines sharply beyond a stellar flux of ~3 due to excessive radiation."
                        ]),
                        html.Li([
                            html.Strong("Host Star Metallicity: "),
                            "Weak correlation suggests both metal-rich and metal-poor stars can host habitable planets."
                        ]),
                        html.Li([
                            html.Strong("System Architecture: "),
                            "Compact multi-planet systems tend to have more stable habitable planets."
                        ]),
                    ]),

                    html.P("These findings highlight the complex interplay between planetary structure, orbital dynamics, "
                        "and stellar environment in determining exoplanet habitability.")
                ]),

                dcc.Graph(id="stellar-pie-chart", figure=fig_pie, className="graph"),
            ])
        ])
    ]),

    html.Div(className="section-divider"),

    # Data Summary Section
    html.Section(id="summary", children=[
        html.Div(className="text-section", children=[
            html.H2("📊 Data Summary", className="section-title"),
        ]),

        html.Div(className="swap-container", children=[
            html.Div(id="text-section", className="swap-box text-box", children=[
                html.H3("🌌 Raw Exoplanetary Dataset Overview", className="box-title"),
                html.P("This dataset contains 5,599 entries and 57 columns, covering exoplanet properties, discovery details, and stellar characteristics.", className="box-content"),
                
                html.H4("📊 Data Composition"),
                html.Ul([
                    html.P("34 Numerical, 19 Categorical and 4 Integer"),
                ]),
                
                html.H4("🔍 Missing Data"),
                html.Ul([
                    html.Li("Eccentricity (~14%) and inclination (~23%) missing."),
                    html.Li("Stellar age (~21%) and metallicity (~8%) have gaps."),
                    html.Li("P_OMEGA has over 70% missing values."),
                ]),
            ]),
            
            html.Button("Processed Data", id="swap-btn", n_clicks=0, className="swap-btn"),

            html.Div(id="graph-section", className="swap-box graph-box", children=[
                dcc.Graph(id="graph-display", figure=fig_raw, className="graph")
            ])
        ]),

        html.Div(className="selector-box", children=[
            html.Label("Select Data Field:", className="selector-label"),
            dcc.Dropdown(
                id="data-field-selector",
                options = [{"label": col, "value": col} for col in cleaned_df.select_dtypes(include=['number']).columns],
                value="P_YEAR",
                clearable=False,
            ),
            html.Div(className="content-box", children=[
                dcc.Graph(id="data-graph", className="graph"),
                html.Div(id="data-description", className="data-description-box")
            ]),
        ]),
    ]), 

    html.Div(className="section-divider"),
    
    # Data Analysis Section
    html.Section(id="analysis", className="analysis-section", children=[
        html.Div(className="text-section", children=[
            html.H2("🔍 Data Analysis", className="section-title"),
        ]),
        
        html.Div([
            html.H3("Select a Hypothesis", className="analysis-text"),
            
            html.Div(
                [
                    html.Div(create_card(row["Parameter"], row["Value"], idx), className="analysis-card")
                    for idx, row in hypothesis.iterrows()
                ],
                className="analysis-container",
            ),
            modal
        ], className="analysis-box")
    ]),

    html.Div(className="section-divider"),

    # Conclusion Section
    html.Section(
        id="conclusion",
        children=[
            html.Div(className="text-section", children=[
                html.H2("Conclusion", className="section-title"),
            ]),

            html.Div(className="conclusion-section", children=[
                html.Div(className="conclusion-section", children=[
                    html.H2("📊 This analysis highlights key insights from the data.", className="section-title"),
                    html.P("Click 'Next' to explore different habitability conditions.", className="section-description")
                ]),

                html.Div(className="conclusion-container", children=[
                    html.Div(className="conclusion-box", children=[
                        dcc.Graph(id="conclusion-graph", className="graph"),
                        html.Div(id="conclusion-description", className="conclusion-description-box"),
                        html.Button("Next", id="next-slide", n_clicks=0, className="next-button")
                    ]),
                ])
            ]),
        ]
    ),

    html.Div(className="section-divider"),

    # Footer
    html.Footer(className="footer", children=[
        html.P("© 2025 Stellar Analysis Project"),
        html.A("View Me on GitHub", href="https://github.com/paaduka32", className="footer-link")
    ])
])

# ========================================== METHOD TO HANDLE RAW AND PROCESSED DATA SUMMARY IN DATA SUMMARY SECTION ===============================================================
@callback(
    Output("text-section", "children"),
    Output("graph-section", "children"),
    Output("swap-btn", "children"),  
    Input("swap-btn", "n_clicks")
)
def swap_content(n_clicks):
    # Update Text Sections
    text1 = html.Div([
        html.H3("🌌 Raw Exoplanetary Dataset Overview", className="box-title"),
        
        html.P("This dataset contains 5,599 entries and 57 columns, covering exoplanet properties, discovery details, and stellar characteristics.", className="box-content"),
        
        html.H4("📊 Data Composition"),
        html.Ul([
            html.P("34 Numerical, 19 Categorical and 4 Integer"),
        ]),
        
        html.H4("🔍 Missing Data"),
        html.Ul([
            html.Li("Eccentricity (~14%) and inclination (~23%) missing."),
            html.Li("Stellar age (~21%) and metallicity (~8%) have gaps."),
            html.Li("P_OMEGA has over 70% missing values."),
        ]),
    ])
    text2 = html.Div([
        html.H3("🛠️ Data Cleaning Process", className="box-title"),
        html.P("To ensure high-quality data for analysis, we applied the following cleaning steps:", className="box-content"),
        html.Ul([
            html.Li("Dropped columns with more than 50% missing values."),
            html.Li("Filled missing categorical values using mode."),
            html.Li("Applied interpolation for time-ordered variables."),
            html.Li("Used linear regression to estimate missing numerical values."),
            html.Li("Replaced zero values with NaN in critical columns."),
            html.Li("Applied KNN imputation for complex missing patterns."),
            html.Li("Removed duplicate rows to maintain data integrity."),
        ])
    ])

    # Update graphs
    graph1 = dcc.Graph(id="graph-display", figure=fig_raw, className="graph")
    graph2 = dcc.Graph(id="graph-display", figure=fig_cleaned, className="graph")

    # Button text toggle
    if n_clicks % 2 == 0:
        return text1, graph1, "Processed Data"
    else:
        return text2, graph2, "Raw Data"

# ========================================== METHOD TO HANDLE HISTOGRAMS FOR DATA COLUMNS' SUMMARY IN DATA SUMMARY SECTION ===============================================================
@app.callback(
    [Output("data-graph", "figure"),
    Output("data-description", "children")],
    [Input("data-field-selector", "value")]
)
def update_output(selected_field):
    valid_data = cleaned_df[selected_field].dropna().astype(float)

    if valid_data.empty:
        return go.Figure(), "No valid data available."

    # Generate histogram for various data columns
    fig = px.histogram(valid_data, x=selected_field, nbins=30, histnorm='probability density',
                       title=f"Distribution of {selected_field}", opacity=0.6)

    data_array = valid_data.values

    # Perform KDE fit for histogram 
    kde = gaussian_kde(data_array)
    x_vals = np.linspace(data_array.min(), data_array.max(), 200)
    y_vals = kde(x_vals)

    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Density Curve', 
                             line=dict(color='blue', width=2)))

    mean_val = np.mean(data_array)

    mean_density = kde(mean_val)[0]

    # Add mean marker and style the graph
    fig.add_trace(go.Scatter(
        x=[mean_val], 
        y=[mean_density * 1.1],
        mode='markers+text', 
        name="Mean",
        text=[f"Mean: {mean_val:.2f}"], 
        textposition="bottom center",
        marker=dict(color='red', size=8, symbol="x")
    ))
    fig.update_layout(
        xaxis_title=selected_field,
        yaxis_title="Density",
        template="plotly_white",
        showlegend=True,
        font=dict(size=12),
    )

    # Fetch respected description from above description pool
    description = descriptions.get(selected_field, "No description available.")

    return fig, description

# ========================================== METHOD TO DISPLAY GRAPH AND DESCRIPTION IN DATA ANALYSIS SECTION ===============================================================
# Callback to Show Graph & Description
@app.callback(
    [Output("modal", "is_open"),
    Output("graph", "figure"),
    Output("description", "children")],
    [Input({"type": "open-modal", "index": dash.ALL}, "n_clicks"),
    Input("close-modal", "n_clicks")],
    [State("modal", "is_open")]
)
def display_graph(open_clicks, close_click, is_open):
    triggered_id = ctx.triggered_id  # Get which component was clicked

    # If close button is clicked, close the modal
    if triggered_id == "close-modal":
        return False, dash.no_update, dash.no_update

    # If one of the parameter cards is clicked
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "open-modal":
        idx = triggered_id["index"]
        param = hypothesis.loc[idx, "Parameter"]
        
        # If hypothesis-1 is selected
        if param == "Hypothesis-1":
            # Log-transform mass and radius
            cleaned_df["log_M"] = np.log10(cleaned_df["P_MASS"])
            cleaned_df["log_R"] = np.log10(cleaned_df["P_RADIUS"])

            # Create subplot layout (2 rows, 1 column)
            fig = make_subplots(
                rows=2, cols=1, 
                subplot_titles=[
                    "Mass-Radius Relation (Log-Log Scale)", 
                    "Mass-Radius Relation by Planet Type"
                ],
                vertical_spacing=0.2
            )

            # First Plot: Scatter Plot for Mass vs. Radius (Log-Log)
            scatter_fig = px.scatter(
                cleaned_df, x="log_M", y="log_R", color="P_TEMP_EQUIL", 
                color_continuous_scale="viridis",
                labels={
                    "log_M": "log(Mass) [Earth Masses]", 
                    "log_R": "log(Radius) [Earth Radii]", 
                    "P_TEMP_EQUIL": "Equilibrium Temperature (K)"
                },
            )

            for trace in scatter_fig.data:
                fig.add_trace(trace, row=1, col=1)

            # Pearson Correlation
            corr, p_value = pearsonr(cleaned_df["log_M"], cleaned_df["log_R"])
            description = f"Pearson Correlation Coefficient: {corr:.3f}\nP-value: {p_value:.3f}\n"

            if p_value < 0.05:
                description += "=> There is a significant correlation between planetary mass and radius.\n"
            else:
                description += "=> No significant correlation between planetary mass and radius.\n"

            # Linear Regression Fit
            X = cleaned_df["log_M"].values.reshape(-1, 1)
            y = cleaned_df["log_R"].values.reshape(-1, 1)
            lin_reg = LinearRegression().fit(X, y)
            n_fit = lin_reg.coef_[0][0]
            k_fit = 10 ** lin_reg.intercept_[0]

            description += f"Best-fit power-law equation: log(R) = {np.log10(k_fit):.2f} + {n_fit:.2f} log(M)\n"

            # ANOVA Test
            temp_bins = pd.qcut(cleaned_df["P_TEMP_EQUIL"], q=4, labels=["Low", "Medium", "High", "Very High"])
            groups = [cleaned_df["log_R"][temp_bins == level] for level in temp_bins.cat.categories]
            anova_stat, anova_p = f_oneway(*groups)

            # Second Plot: Mass-Radius Relation by Planet Type
            def mass_radius_relation(M, k, n):
                return k * (M ** n)

            types = ["Terran", "Superterran", "Neptunian", "Jovian"]
            colors = ["blue", "green", "orange", "red"]
            fit_results = {}

            for planet_type, color in zip(types, colors):
                subset = cleaned_df[cleaned_df["P_TYPE"] == planet_type]
                if not subset.empty:
                    popt, _ = curve_fit(mass_radius_relation, subset["P_MASS"], subset["P_RADIUS"], p0=[1, 0.3])
                    k_fit, n_fit = popt
                    fit_results[planet_type] = (k_fit, n_fit)

                    # Scatter Plot for each type
                    fig.add_trace(
                        go.Scatter(
                            x=subset["log_M"], y=subset["log_R"], mode="markers",
                            name=f"{planet_type} (n={n_fit:.2f})", marker=dict(color=color),
                        ),
                        row=2, col=1
                    )

                    # Plot Fit Line (Non-interactive, separate from legend)
                    mass_range = np.linspace(subset["log_M"].min(), subset["log_M"].max(), 100)
                    fig.add_trace(
                        go.Scatter(
                            x=mass_range, 
                            y=np.log10(mass_radius_relation(10**mass_range, k_fit, n_fit)),
                            mode="lines", 
                            line=dict(color=color, dash="dash"),  
                            showlegend=False
                        ),
                        row=2, col=1
                    )

            fig.update_xaxes(title_text="log(Mass) [Earth Masses]", row=1, col=1)
            fig.update_yaxes(title_text="log(Radius) [Earth Radii]", row=1, col=1)
            fig.update_xaxes(title_text="log(Mass) [Earth Masses]", row=2, col=1)
            fig.update_yaxes(title_text="log(Radius) [Earth Radii]", row=2, col=1)

            fig.update_layout(
                title="Mass-Radius Relations",
                height=900,
                legend_tracegroupgap=500
            )

            description = f"""
            <b>Pearson Correlation Coefficient:</b> {corr:.3f} <br>
            <b>P-value:</b> {p_value:.3f} <br>
            {"<b>⇒ There is a significant correlation between planetary mass and radius.</b>" if p_value < 0.05 else "<b>⇒ No significant correlation between planetary mass and radius.</b>"} <br><br>

            <b>Best-fit power-law equation:</b> log(R) = {np.log10(k_fit):.2f} + {n_fit:.2f} log(M) <br><br>

            <b>ANOVA Test for Temperature Effect on Radius:</b> <br>
            F-statistic: {anova_stat:.3f}, p-value: {anova_p:.3f} <br>
            {"<b>⇒ Temperature has a significant effect on planetary radius.</b>" if anova_p < 0.05 else "<b>⇒ No significant effect of temperature on planetary radius.</b>"} <br><br>

            <b>===== Mass-Radius Relation by Planet Type =====</b><br>
            """  
            # Add planet type results
            for planet_type, (k_fit, n_fit) in fit_results.items():
                description += f"<b>{planet_type}:</b> k = {k_fit:.2f}, n = {n_fit:.2f} <br>"
                
                if n_fit < 0.2:
                    description += f"  → {planet_type} planets are likely gas giants with strong compression effects.<br>"
                elif 0.2 <= n_fit < 0.27:
                    description += f"  → {planet_type} planets are likely Neptune-like with thick atmospheres.<br>"
                elif 0.27 <= n_fit < 0.3:
                    description += f"  → {planet_type} planets are likely Earth-like (rocky, silicate composition).<br>"
                else:
                    description += f"  → {planet_type} planets may be water-rich or mixed composition.<br>"

            description += """
            <br><b>===== General Observations =====</b><br>
            1. Gas giants (low n) show strong compression effects, limiting radius growth at high mass.<br>
            2. Rocky planets (Earth-like) follow a steeper power-law (higher n), indicating a denser composition.<br>
            3. The transition from Superterran to Neptunian (~2-3 Earth radii) suggests an atmospheric threshold.<br>
            4. Some outliers may indicate exotic compositions, such as water worlds or dense iron planets.<br>
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-2 is selected
        elif param == "Hypothesis-2":
            # Define the power-law function
            def mass_radius_relation(M, k, n):
                return k * (M ** n)

            # Fit using curve_fit (nonlinear least squares)
            popt, pcov = curve_fit(mass_radius_relation, cleaned_df["P_MASS"], cleaned_df["P_RADIUS"], p0=[1, 0.3])
            k_fit, n_fit = popt

            # Log-transform mass and radius
            cleaned_df["log_M"] = np.log10(cleaned_df["P_MASS"])
            cleaned_df["log_R"] = np.log10(cleaned_df["P_RADIUS"])

            # Linear Regression on log-transformed data
            X = cleaned_df["log_M"].values.reshape(-1, 1)
            y = cleaned_df["log_R"].values.reshape(-1, 1)
            lin_reg = LinearRegression().fit(X, y)
            n_linear = lin_reg.coef_[0][0]
            k_linear = 10 ** lin_reg.intercept_[0]

            # Create interactive scatter plot
            fig = px.scatter(cleaned_df, x="log_M", y="log_R", 
                            labels={"log_M": "log(Mass) [Earth Masses]", "log_R": "log(Radius) [Earth Radii]"},
                            title="Linear Fit (Power-Law Model)")

            # Add regression fit line
            fig.add_trace(go.Scatter(
                x=cleaned_df["log_M"], 
                y=lin_reg.predict(X).flatten(),
                mode="lines", 
                name=f"Fit: log(R) = {np.log10(k_linear):.2f} + {n_linear:.2f} log(M)",
                line=dict(color="red")
            ))

            # Save results in markdown description
            description = f"""
            **Best-fit parameters (Nonlinear Curve Fit):**  
            - k = {k_fit:.2f}  
            - n = {n_fit:.2f}  

            **Best-fit parameters (Linear Regression on Log-Transformed Data):**  
            - k = {k_linear:.2f}  
            - n = {n_linear:.2f}  

            **Interpretation:**  
            """

            if n_fit < 0.2:
                description += "• Planets are likely **gas giants** (compressibility effects).  \n"
            elif 0.2 <= n_fit < 0.27:
                description += "• Planets are likely **iron-rich** (dense rocky composition).  \n"
            elif 0.27 <= n_fit < 0.3:
                description += "• Planets are likely **silicate (Earth-like rocky planets).**  \n"
            else:
                description += "• Planets may be **water-rich or mixed composition.**  \n"
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-3 is selected
        elif param == "Hypothesis-3":
            water_habitable = cleaned_df[(cleaned_df["P_TEMP_EQUIL"] >= 200) & (cleaned_df["P_TEMP_EQUIL"] <= 300)]

            # Extract temperature values and compute KDE
            temperature_values = cleaned_df["P_TEMP_EQUIL"].dropna().values
            kde = gaussian_kde(temperature_values)

            # Generate x values for KDE plot
            x_vals = np.linspace(temperature_values.min(), temperature_values.max(), 200)
            y_vals = kde(x_vals)

            # Create Plotly figure
            fig = go.Figure()

            # KDE Density Curve
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="lines", 
                name="Density Estimation", line=dict(color="gray")
            ))

            # Liquid Water Zone (200K - 300K)
            fig.add_trace(go.Scatter(
                x=[200, 200, 300, 300], y=[0, y_vals.max(), y_vals.max(), 0],
                fill="toself", mode="none",
                fillcolor="rgba(0, 0, 255, 0.3)",
                name="Liquid Water Zone"
            ))

            # Format axes and title
            fig.update_layout(
                title="KDE of Planetary Equilibrium Temperatures",
                xaxis_title="Equilibrium Temperature (K)",
                yaxis_title="Density",
                xaxis=dict(range=[0, temperature_values.max()]),
                template="plotly_white"
            )

            # Compute percentage of planets in habitable range
            water_habitable_percentage = len(water_habitable) / len(cleaned_df) * 100

            # Store results in Markdown format
            description = f"""
            **Percentage of planets in the liquid water temperature range:**  
            - {water_habitable_percentage:.2f}%

            **Conclusion:**  
            """

            if water_habitable_percentage > 10:
                description += "✔ A significant number of exoplanets may have conditions suitable for **liquid water**."
            else:
                description += "❌ Very few planets fall within the temperature range needed for **liquid water**."
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-4 is selected
        elif param == "Hypothesis-4":
            habitable = cleaned_df[cleaned_df["P_HABITABLE"] == 1]["P_PERIOD"].dropna()
            non_habitable = cleaned_df[cleaned_df["P_HABITABLE"] == 0]["P_PERIOD"].dropna()

            # KDE for habitable planets
            x_hab = np.linspace(habitable.min(), habitable.max(), 1000)
            kde_hab = gaussian_kde(habitable)

            # KDE for non-habitable planets
            x_non_hab = np.linspace(non_habitable.min(), non_habitable.max(), 1000)
            kde_non_hab = gaussian_kde(non_habitable)

            # Create Plotly figure
            fig = go.Figure()

            # KDE Curve - Habitable
            fig.add_trace(go.Scatter(
                x=x_hab, y=kde_hab(x_hab), mode="lines",
                name="Habitable", line=dict(color="green")
            ))

            # KDE Curve - Non-Habitable
            fig.add_trace(go.Scatter(
                x=x_non_hab, y=kde_non_hab(x_non_hab), mode="lines",
                name="Non-Habitable", line=dict(color="red")
            ))

            # Add vertical lines for hypothesis range (50-500 days)
            fig.add_trace(go.Scatter(
                x=[50, 50], y=[0, max(kde_hab(x_hab).max(), kde_non_hab(x_non_hab).max())],
                mode="lines", line=dict(color="blue", dash="dash"), name="50 Days (Lower Bound)"
            ))
            fig.add_trace(go.Scatter(
                x=[500, 500], y=[0, max(kde_hab(x_hab).max(), kde_non_hab(x_non_hab).max())],
                mode="lines", line=dict(color="blue", dash="dash"), name="500 Days (Upper Bound)"
            ))

            # Format layout
            fig.update_layout(
                title="KDE of Orbital Period for Habitable vs Non-Habitable Planets",
                xaxis_title="Orbital Period (Days)",
                yaxis_title="Density",
                xaxis=dict(range=[0, 550]),
                template="plotly_white"
            )

            # Statistical Test - Mann-Whitney U test
            stat, p_value = mannwhitneyu(habitable, non_habitable, alternative="two-sided")

            # Store results in Markdown format
            description = f"""
            ### **Conclusion:**
            1. The **Mann-Whitney U test** resulted in a p-value of **{p_value:.3f}**.
            2. Since **p > 0.05**, we **fail to reject** the null hypothesis.
            3. This means there is **no statistically significant difference** between the orbital periods of habitable and non-habitable planets.
            4. The hypothesis that planets with orbital periods between **50-500 days** are more likely to be habitable **is not strongly supported** by this dataset.
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-5 is selected
        elif param == "Hypothesis-5":
            cleaned_df["IN_HABITABLE_ZONE"] = (cleaned_df["P_SEMI_MAJOR_AXIS"] >= 0.5) & (cleaned_df["P_SEMI_MAJOR_AXIS"] <= 2)

            # Create contingency table
            contingency_table = pd.crosstab(cleaned_df["IN_HABITABLE_ZONE"], cleaned_df["P_HABITABLE"])

            # Perform chi-square test
            chi2_stat, p_value_sma, _, _ = chi2_contingency(contingency_table)

            # Hill Sphere vs. Habitability
            hill_habitable = cleaned_df[cleaned_df["P_HABITABLE"] == 1]["P_HILL_SPHERE"].dropna()
            hill_non_habitable = cleaned_df[cleaned_df["P_HABITABLE"] == 0]["P_HILL_SPHERE"].dropna()

            # Perform t-test
            t_stat, p_value_hill = ttest_ind(hill_habitable, hill_non_habitable, equal_var=False)

            # Semi-Major Axis Data
            sma_hab = cleaned_df[cleaned_df["P_HABITABLE"] == 1]["P_SEMI_MAJOR_AXIS"].dropna()
            sma_non_hab = cleaned_df[cleaned_df["P_HABITABLE"] == 0]["P_SEMI_MAJOR_AXIS"].dropna()

            # Hill Sphere KDE Data
            x_hill = np.linspace(cleaned_df["P_HILL_SPHERE"].min(), cleaned_df["P_HILL_SPHERE"].max(), 200)
            kde_hab = gaussian_kde(hill_habitable)
            kde_non_hab = gaussian_kde(hill_non_habitable)

            # Create subplot layout (2 rows, 1 column)
            fig = make_subplots(rows=2, cols=1, subplot_titles=[
                "Boxplot: Semi-Major Axis (Habitable vs Non-Habitable)",
                "KDE Density: Hill Sphere (Habitable vs Non-Habitable)"
            ])

            # Boxplot for Semi-Major Axis
            fig.add_trace(go.Box(
                y=sma_non_hab, name="Non-Habitable", marker_color="red"
            ), row=1, col=1)

            fig.add_trace(go.Box(
                y=sma_hab, name="Habitable", marker_color="green"
            ), row=1, col=1)

            # Habitability Zone Markers
            fig.add_hline(y=0.5, line_dash="dash", line_color="blue", annotation_text="0.5 AU (Lower Bound)", row=1, col=1)
            fig.add_hline(y=2.0, line_dash="dash", line_color="blue", annotation_text="2.0 AU (Upper Bound)", row=1, col=1)

            # KDE Density Plot for Hill Sphere
            fig.add_trace(go.Scatter(
                x=x_hill, y=kde_hab(x_hill), mode="lines",
                name="Habitable", line=dict(color="green")
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=x_hill, y=kde_non_hab(x_hill), mode="lines",
                name="Non-Habitable", line=dict(color="red")
            ), row=2, col=1)

            fig.update_xaxes(range=[0, 10], row=2, col=1)

            # Update layout
            fig.update_layout(
                height=800, width=700, showlegend=True,
                xaxis2_title="Hill Sphere (AU)", yaxis2_title="Density",
                xaxis_title="Habitability", yaxis_title="Semi-Major Axis (AU)",
                template="plotly_white"
            )

            # Store results in Markdown format
            description = f"""
            ### **Conclusion:**
            #### Hypothesis 2: Semi-Major Axis vs. Habitability
            1. **Chi-square test p-value**: {p_value_sma:.3f}.
            2. **{contingency_table.loc[True, 1] if True in contingency_table.index else 0} habitable** planets fall within 0.5 - 2 AU.
            3. **{contingency_table.loc[True, 0] if True in contingency_table.index else 0} non-habitable** planets fall within 0.5 - 2 AU.
            4. **{"A" if p_value_sma < 0.05 else "No"} significant correlation** was found between semi-major axis and habitability.

            #### Hypothesis 7: Hill Sphere vs. Habitability
            1. **T-test p-value**: {p_value_hill:.3f}.
            2. **Mean Hill Sphere radius (Habitable)**: {hill_habitable.mean():.3f} AU.
            3. **Mean Hill Sphere radius (Non-Habitable)**: {hill_non_habitable.mean():.3f} AU.
            4. **{"A" if p_value_hill < 0.05 else "No"} statistically significant difference** in Hill Sphere sizes between habitable and non-habitable planets.
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-6 is selected
        elif param == "Hypothesis-6":
            # Constants
            k = 1.38e-23  # Boltzmann constant (J/K)
            m_avg = (18.015 + 32.00 + 28.014) / 3 / 1000 / 6.022e23  # Average molecular mass (kg)
            threshold_habitable = 1.5  # Escape velocity must be 1.5x thermal velocity

            # Calculate Thermal Velocity (average gas)
            cleaned_df["V_thermal"] = np.sqrt((2 * k * cleaned_df["P_TEMP_EQUIL"]) / m_avg) / 1000  # Convert to km/s

            # Classify planets as habitable or not based on atmospheric retention
            cleaned_df["Habitable"] = cleaned_df["P_ESCAPE"] > cleaned_df["V_thermal"] * threshold_habitable

            # Count results
            num_habitable = cleaned_df["Habitable"].sum()
            num_not_habitable = len(cleaned_df) - num_habitable
            habitable_percentage = (num_habitable / len(cleaned_df)) * 100

            # Correlation between escape velocity and equilibrium temperature
            corr = cleaned_df["P_ESCAPE"].corr(cleaned_df["P_TEMP_EQUIL"])

            # Additional insight: Are most habitable planets large or small?
            median_escape_velocity = cleaned_df["P_ESCAPE"].median()
            large_planets = cleaned_df[cleaned_df["P_ESCAPE"] > median_escape_velocity]["Habitable"].mean() * 100
            small_planets = cleaned_df[cleaned_df["P_ESCAPE"] <= median_escape_velocity]["Habitable"].mean() * 100

            # Interactive Scatter Plot
            fig = px.scatter(
                cleaned_df,
                x="P_ESCAPE",
                y="V_thermal",
                color="Habitable",
                color_discrete_map={True: "green", False: "red"},
                labels={"P_ESCAPE": "Escape Velocity (km/s)", "V_thermal": "Thermal Velocity (km/s)"},
                title="Atmospheric Retention Based on Escape Velocity",
                hover_data=["P_TEMP_EQUIL"]
            )

            # Add threshold line (1.5x Thermal Velocity)
            fig.add_trace(go.Scatter(
                x=[0, max(cleaned_df["P_ESCAPE"])],
                y=[0, max(cleaned_df["P_ESCAPE"]) / threshold_habitable],
                mode="lines",
                line=dict(dash="dash", color="black"),
                name="Threshold Line (1.5x Thermal)"
            ))

            fig.update_layout(
                height=600,
                width=800,
                legend_title="Retains Atmosphere?",
                template="plotly_white",
                yaxis_range=[0, 3]
            )

            # Store markdown results
            description = f"""
            ### **Conclusions from Atmospheric Retention Analysis**
            - **Total Planets Analyzed**: {len(cleaned_df)}
            - **Habitable Planets (Can Retain Atmosphere)**: {num_habitable} (**{habitable_percentage:.2f}%**)
            - **Not Habitable Planets (Likely to Lose Atmosphere)**: {num_not_habitable}

            #### **Key Observations**
            - **Correlation between Escape Velocity and Temperature**: {corr:.3f}
            - {"Strong" if corr > 0.5 else "Moderate" if corr > 0.2 else "Weak"} correlation.
            - {"Higher temperatures are generally associated with planets having higher escape velocities." if corr > 0.5 else "Some trend exists, but other factors influence atmospheric retention."}

            #### **Size vs. Atmospheric Retention**
            - **Percentage of Large Planets (above median escape velocity) retaining atmosphere**: {large_planets:.2f}%
            - **Percentage of Small Planets (below median escape velocity) retaining atmosphere**: {small_planets:.2f}%
            - {"Larger planets are more likely to retain atmospheres, which aligns with expectations." if large_planets > small_planets else "Smaller planets unexpectedly retain atmospheres better—this could indicate strong magnetic fields or other stabilizing factors."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-7 is selected
        elif param == "Hypothesis-7":
            # Log transformation
            cleaned_df["log_SemiMajor"] = np.log10(cleaned_df["P_SEMI_MAJOR_AXIS"])
            cleaned_df["log_Temp"] = np.log10(cleaned_df["P_TEMP_EQUIL"])

            # Identify habitable planets based on temperature
            cleaned_df["In_Habitable_Zone"] = (cleaned_df["P_TEMP_EQUIL"] >= 200) & (cleaned_df["P_TEMP_EQUIL"] <= 300)

            # Scatter plot
            fig = px.scatter(
                cleaned_df, 
                x="log_SemiMajor", 
                y="log_Temp", 
                color="In_Habitable_Zone",
                color_discrete_map={True: "red", False: "blue"},
                labels={"log_SemiMajor": "log(Semi-Major Axis) [AU]", "log_Temp": "log(Equilibrium Temperature) [K]"},
                title="Orbital Distance vs. Equilibrium Temperature",
                opacity=0.7
            )

            # Customize layout
            fig.update_layout(
                legend_title="In Habitable Zone",
                template="plotly_white",
                xaxis=dict(title="log(Semi-Major Axis) [AU]"),
                yaxis=dict(title="log(Equilibrium Temperature) [K]"),
                height=600,
                width=800
            )

            # Percentage of planets in the habitable zone
            habitable_count = cleaned_df["In_Habitable_Zone"].sum()
            habitable_percentage = (habitable_count / len(cleaned_df)) * 100

            # Correlation check
            corr = cleaned_df["log_SemiMajor"].corr(cleaned_df["log_Temp"])

            # Store conclusions in Markdown format
            description = f"""
            ### **Conclusions from Orbital Distance vs. Equilibrium Temperature**
            - **Percentage of planets in the habitable zone**: {habitable_percentage:.2f}%
            {"- A significant fraction of exoplanets fall in the habitable zone, suggesting habitable planets are not rare." if habitable_percentage > 10 else
            "- A moderate fraction (5-10%) exist in the habitable zone, indicating some potential for habitability." if 5 <= habitable_percentage <= 10 else
            "- Very few planets fall within the habitable zone, suggesting that habitable exoplanets are rare in this dataset."}

            - **Correlation coefficient between log(Semi-Major Axis) and log(Equilibrium Temperature)**: {corr:.3f}
            {"- Strong negative correlation: Planets farther from their stars tend to have lower equilibrium temperatures." if corr < -0.5 else
            "- Moderate negative correlation: Temperature decreases with distance but with variations." if -0.5 <= corr < -0.2 else
            "- Weak correlation: Other factors (e.g., greenhouse effects, albedo) influence planetary temperature."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-8 is selected
        elif param == "Hypothesis-8":
            # Define tidal locking condition: Small planets (<10 Earth masses) with short orbital periods (<10 days)
            cleaned_df["Tidally_Locked"] = (cleaned_df["P_PERIOD"] < 10) & (cleaned_df["P_MASS"] < 10)

            # Percentage of tidally locked planets
            tidally_locked_percentage = (cleaned_df["Tidally_Locked"].sum() / len(cleaned_df)) * 100

            # Correlation check: Relationship between orbital period and mass
            corr = cleaned_df["P_PERIOD"].corr(cleaned_df["P_MASS"])

            # Create interactive scatter plot
            fig = px.scatter(
                cleaned_df, 
                x="P_PERIOD", 
                y="P_MASS", 
                color="Tidally_Locked",
                color_discrete_map={True: "red", False: "gray"},
                labels={"P_PERIOD": "Orbital Period (Days)", "P_MASS": "Mass (Earth Masses)"},
                title="Tidal Locking: Mass vs. Orbital Period",
                opacity=0.7
            )

            # Apply log scale for better visualization
            fig.update_layout(
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title="Orbital Period (Days)",
                yaxis_title="Mass (Earth Masses)",
                height=600,
                width=700,
                legend_title="Tidally Locked?",
                template="plotly_white"
            )

            # Store conclusions in Markdown format
            description = f"""
            ### **Conclusions from Tidal Locking Analysis**
            - **Percentage of tidally locked planets**: {tidally_locked_percentage:.2f}%
            {"- A significant fraction (>20%) of detected exoplanets are likely tidally locked, meaning one side always faces their star." if tidally_locked_percentage > 20 else
            "- A moderate fraction (10-20%) may be tidally locked, suggesting that close-in planets frequently experience synchronous rotation." if 10 <= tidally_locked_percentage <= 20 else
            "- A small percentage (<10%) of planets are tidally locked, meaning most planets in the dataset may rotate independently."}

            - **Correlation coefficient between Orbital Period and Mass**: {corr:.3f}
            {"- Strong negative correlation: Lower-mass planets tend to have shorter orbital periods, making them more prone to tidal locking." if corr < -0.5 else
            "- Moderate negative correlation: There is some tendency for lower-mass planets to have short periods, but other factors also play a role." if -0.5 <= corr < -0.2 else
            "- Weak correlation: Planet mass and orbital period do not strongly influence each other in this dataset."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)

        # If hypothesis-9 is selected
        elif param == "Hypothesis-9":
            # Ensure missing metallicity values are filled
            cleaned_df["S_METALLICITY"].fillna(cleaned_df["S_METALLICITY"].mean())

            # Create a binary column for terrestrial planets
            cleaned_df["Is_Terrestrial"] = (cleaned_df["P_TYPE"] == "Terran").astype(int)

            # Extract metallicity data for KDE
            terrestrial_metallicity = cleaned_df[cleaned_df["Is_Terrestrial"] == 1]["S_METALLICITY"].dropna()
            non_terrestrial_metallicity = cleaned_df[cleaned_df["Is_Terrestrial"] == 0]["S_METALLICITY"].dropna()

            # KDE density estimates
            x_vals = np.linspace(cleaned_df["S_METALLICITY"].min(), cleaned_df["S_METALLICITY"].max(), 200)
            kde_terrestrial = gaussian_kde(terrestrial_metallicity)
            kde_non_terrestrial = gaussian_kde(non_terrestrial_metallicity)

            # Grouped data for scatter plot
            df_grouped = cleaned_df.groupby("S_METALLICITY")["Is_Terrestrial"].mean().reset_index()

            # Extract x and y values for scatter
            x_scatter = df_grouped["S_METALLICITY"].values
            y_scatter = df_grouped["Is_Terrestrial"].values

            # Fit KDE for scatter
            kde = gaussian_kde(np.vstack([x_scatter, y_scatter]))

            # Generate grid for density
            x_grid, y_grid = np.meshgrid(
                np.linspace(x_scatter.min(), x_scatter.max(), 100),
                np.linspace(y_scatter.min(), y_scatter.max(), 100)
            )
            z_vals = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

            # Create subplot layout
            fig = make_subplots(rows=2, cols=1, subplot_titles=[
                "KDE Fit: Metallicity Distribution for Terrestrial & Non-Terrestrial Planets",
                "KDE Fit: Metallicity vs. Likelihood of Terrestrial Planets"
            ])

            # --- Subplot 1: KDE of metallicity distribution ---
            fig.add_trace(
                go.Scatter(x=x_vals, y=kde_terrestrial(x_vals), mode='lines', name="Terrestrial Planets", line=dict(color="blue")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=x_vals, y=kde_non_terrestrial(x_vals), mode='lines', name="Non-Terrestrial Planets", line=dict(color="red")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[0.0, 0.0], y=[0, max(kde_terrestrial(x_vals).max(), kde_non_terrestrial(x_vals).max())], 
                        mode="lines", name="Solar Metallicity (Sun)", line=dict(color="black", dash="dash")),
                row=1, col=1
            )

            # --- Subplot 2: Scatter + KDE Heatmap ---
            fig.add_trace(
                go.Scatter(x=x_scatter, y=y_scatter, mode='markers', marker=dict(color="blue", size=6), name="Data Points"),
                row=2, col=1
            )
            fig.add_trace(
                go.Contour(x=np.linspace(x_scatter.min(), x_scatter.max(), 100),
                        y=np.linspace(y_scatter.min(), y_scatter.max(), 100),
                        z=z_vals, colorscale="Viridis", opacity=0.6, showscale=False, name="Density"),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=[0.0, 0.0], y=[0, 1], mode="lines", name="Solar Metallicity (Sun)", line=dict(color="black", dash="dash")),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                height=800,
                width=600,
                showlegend=True,
                xaxis_title="Host Star Metallicity",
                xaxis2_title="Host Star Metallicity",
                yaxis_title="Density",
                yaxis2_title="Fraction of Terrestrial Planets",
                template="plotly_white"
            )

            # Compute conclusions
            mean_terrestrial = terrestrial_metallicity.mean()
            mean_non_terrestrial = non_terrestrial_metallicity.mean()
            correlation = np.corrcoef(x_scatter, y_scatter)[0, 1]

            # Store conclusions in Markdown format
            description = f"""
            ### **Conclusions from Metallicity Distribution**
            - **The average metallicity of host stars with terrestrial planets is** {mean_terrestrial:.3f}.
            - **The average metallicity of host stars with non-terrestrial planets is** {mean_non_terrestrial:.3f}.
            {"- Terrestrial planets tend to form around metal-rich stars." if mean_terrestrial > mean_non_terrestrial else 
            "- Terrestrial planets do not show a strong preference for metal-rich environments."}

            ### **Conclusions from Metallicity vs. Terrestrial Fraction**
            - **The correlation between host star metallicity and terrestrial planet fraction is** {correlation:.3f}.
            {"- There is a weak positive trend: metal-rich stars are slightly more likely to host terrestrial planets." if correlation > 0.1 else 
            "- There is a weak negative trend: metal-poor stars may host more terrestrial planets." if correlation < -0.1 else 
            "- No significant correlation is observed between metallicity and the likelihood of hosting terrestrial planets."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-10 is selected
        elif param == "Hypothesis-10":
            # Compute Pearson correlations
            corr_ecc_temp, p_ecc_temp = pearsonr(cleaned_df["P_ECCENTRICITY"], cleaned_df["P_TEMP_EQUIL"])

            # Scatter plot
            fig = px.scatter(
                cleaned_df,
                x="P_ECCENTRICITY",
                y="P_TEMP_EQUIL",
                color="S_TYPE_TEMP",
                title="Eccentricity vs. Surface Temperature (Colored by Star Type)",
                labels={"P_ECCENTRICITY": "Orbital Eccentricity", "P_TEMP_EQUIL": "Surface Temperature (K)"},
                opacity=0.7,
                template="plotly_white"
            )
            # Markdown Description
            description = f"""
            ### **Eccentricity vs. Surface Temperature Analysis**
            - **Pearson Correlation:** {corr_ecc_temp:.3f}
            - **p-value:** {p_ecc_temp:.3f}

            **Findings:**
            {"✅ Significant correlation: High eccentricity planets may experience larger temperature variations." if p_ecc_temp < 0.05 else "❌ Weak correlation: Eccentricity alone may not strongly influence planetary temperature."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-11 is selected
        elif param == "Hypothesis-11":
            # Compute Pearson correlation
            corr_flux_temp, p_flux_temp = pearsonr(cleaned_df["P_FLUX"], cleaned_df["P_TEMP_EQUIL"])

            # Scatter plot (Log Scale for Flux)
            fig = px.scatter(
                cleaned_df,
                x="P_FLUX",
                y="P_TEMP_EQUIL",
                color="S_TYPE_TEMP",
                title="Flux vs. Surface Temperature (Colored by Star Type)",
                labels={"P_FLUX": "Stellar Flux", "P_TEMP_EQUIL": "Surface Temperature (K)"},
                opacity=0.7,
                template="plotly_white",
                log_x=True  # Log scale for better visualization
            )
            # Markdown Description
            description = f"""
            ### **Flux vs. Surface Temperature Analysis**
            - **Pearson Correlation:** {corr_flux_temp:.3f}
            - **p-value:** {p_flux_temp:.3f}

            **Findings:**
            {"✅ Strong correlation: Stellar flux significantly impacts planetary temperature." if p_flux_temp < 0.05 else "❌ Weak correlation: Flux does not strongly predict temperature in this dataset."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
        
        # If hypothesis-12 is selected
        elif param == "Hypothesis-12":
            # Drop missing values
            data = cleaned_df["P_ECCENTRICITY"].dropna()

            # Fit KDE
            kde = gaussian_kde(data)
            x_vals = np.linspace(min(data), max(data), 100)

            # Create histogram trace
            hist_trace = go.Histogram(
            x=data,
            nbinsx=30,
            histnorm="probability density",
            marker=dict(color="steelblue", line=dict(color="black", width=1)),
            opacity=0.7,
            name="Histogram"
            )

            # Create KDE trace
            kde_trace = go.Scatter(
            x=x_vals,
            y=kde(x_vals),
            mode="lines",
            line=dict(color="red", width=2),
            name="KDE Fit"
            )

            # Combine both traces
            fig = go.Figure([hist_trace, kde_trace])

            # Update layout
            fig.update_layout(
            title="Histogram with KDE Fit",
            xaxis_title="Orbital Eccentricity",
            yaxis_title="Density",
            template="plotly_white"
            )

            # --- Statistical Tests ---

            # Separate habitable and non-habitable planets
            habitable = cleaned_df[cleaned_df["P_HABITABLE"] == 1]["P_ECCENTRICITY"].dropna()
            non_habitable = cleaned_df[cleaned_df["P_HABITABLE"] == 0]["P_ECCENTRICITY"].dropna()

            # Perform tests
            t_stat, t_pval = ttest_ind(habitable, non_habitable, equal_var=False)  # T-test
            u_stat, u_pval = mannwhitneyu(habitable, non_habitable)  # Mann-Whitney U test

            # Store results in Markdown format
            description = f"""
            ### **Statistical Analysis of Orbital Eccentricity**
            - **T-test p-value:** {t_pval:.5f}
            - **Mann-Whitney U test p-value:** {u_pval:.5f}

            **Interpretation:**
            {"✅ There is a significant difference in orbital eccentricity between habitable and non-habitable planets." if t_pval < 0.05 else "❌ No significant difference in orbital eccentricity between habitable and non-habitable planets."}
            """
            return True, fig, dcc.Markdown(description, dangerously_allow_html=True)
    return is_open, dash.no_update, dash.no_update

# ========================================== METHOD TO DYNAMIC GRAPH AND DESCRIPTION IN CONCLUSION SECTION ===============================================================
@app.callback(
    [Output("conclusion-graph", "figure"),
     Output("conclusion-description", "children")],
    Input("next-slide", "n_clicks")
)
def update_conclusion(n):
    step = n % 6  # Cycle through 6 different graphs
    return generate_graph(step)

if __name__ == "__main__":
    app.run_server(debug=True)
