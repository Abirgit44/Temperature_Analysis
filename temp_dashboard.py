# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:13:29 2023

@author: 91771
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_pandas_profiling import st_profile_report

# Define page functions


def home_page():
    # Add content for the Home page
    st.markdown(
        """
    <h1 align="center">
      <span style="font-size: 2.5em; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
         Temperature Explorer 🌡️
      </span>
    </h1>

    <p align="center">
      <span style="font-size: 1.5em; color: #FFD700; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        🚀 Dive into the World of Temperature Data 🚀
      </span>
    </p>

    <hr style="height: 2px; border-width: 0; color: #4A90E2; background-color: #4A90E2; opacity: 0.8;">

    <p align="center">
      <span style="font-size: 1.2em; color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        Explore the climate with dazzling visualizations and insights!
      </span>
    </p>

    <p align="center">
      <span style="font-size: 1.2em; color: #FFD700; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        🌐 Discover temperature patterns, trends, and more!
      </span>
    </p>

    <p align="center">
      <span style="font-size: 1.2em; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        📈 Dive into the data and embark on a journey of insights.
      </span>
    </p>

    <p align="center">
      <span style="font-size: 1.5em; color: #FFD700; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        Ready to get started? Choose a section from the <i>Navigation Pane</i> by accessing it from top left corner!
      </span>
    </p>

    <hr style="height: 2px; border-width: 0; color: #4A90E2; background-color: #4A90E2; opacity: 0.8;">

    <p align="center">
      <span style="font-size: 1.2em; color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        Powered by Python, Streamlit, and my curiosity.
      </span>
    </p>

    <div style="border: 2px solid #4A90E2; border-radius: 10px; padding: 20px; background-color: #231E0B; box-shadow: 0px 0px 20px #4A90E2; margin: 20px;">
        <p align="center">
            <span style="font-size: 1.2em; font-style: italic; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
                This dataset contains Monthly, Seasonal, and Annual mean temperatures for the year range of 1901 to 2021.
            </span>
        </p>
        <p align="center">
            The dataset was published by
            <a href="https://data.gov.in/resource/monthly-seasonal-and-annual-mean-temperature-series-period-1901-2021" style="text-decoration: none; color: #4A90E2; transition: color 0.3s;">
                <u>IMD Pune</u>
            </a>
            and accessed from
            <a href="https://data.gov.in/" style="text-decoration: none; color: #4A90E2; transition: color 0.3s;">
                <u>data.gov.in</u>
            </a>.
        </p>
    </div>

    <p align="center">
      <span style="font-size: 1.2em; color: #FFD700; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        📊 Created with love for data enthusiasts!
      </span>
    </p>
    """,
        unsafe_allow_html=True,
    )


# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("temperature_data.csv", encoding="ISO-8859-1")
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def data_page():
    st.markdown(
        """
    <div align="center">
      <div style="border: 2px solid #191970; border-radius: 10px; padding: 20px; background-color: #B2FBD6; box-shadow: 0px 0px 20px #191970;">
        <h2 align="center">
          <span style="font-size: 1.8em; color: #191970; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
            About the Data
          </span>
        </h2>
        <p align="center" style="font-size: 1.2em; color: #333; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);">
          Explore the Historical Temperature Dataset and uncover valuable insights.
        </p>
        <div style="background-color: rgba(25, 25, 112, 0); padding: 10px; border-radius: 10px; box-shadow: 0px 0px 20px #4A90E2;">
            <p style="font-size: 1.2em; color: #191989; text-align: center; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); font-style: italic;">
                This dataset below is a treasure trove of Monthly, Seasonal, and Annual mean temperatures spanning from 1901 to 2021. Originally published by IMD, Pune, I've accessed this invaluable data from <a href="https://data.gov.in/" style="text-decoration: none; color: #2C19C5;">data.gov.in</a>.
            </p>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load and display the 'temperature_data.csv' data
    data = load_data()
    # st.dataframe(data)


    # Display the styled DataFrame
    st.write(data, unsafe_allow_html=True)

    st.markdown(
        """
        <h2 style="color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
            Summary of the Dataset
        </h2>
    """,
        unsafe_allow_html=True,
    )

    def format_data_types(data):
        dtypes = data.dtypes
        formatted_types = [
            f"{col}: {dtype}" for col, dtype in zip(data.columns, dtypes)
        ]
        return formatted_types

    # Display data types in the first row and summary in the second row
    st.markdown(
        """
    <div style="display: flex; flex-direction: column;">
        <div style="flex: 1;">
            <div style="color: #FFD700; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); padding: 10px;">
                <h3 style="color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Data Types</h3>
                <div style="overflow-x: auto;">
                    {}
                </div>
            </div>
        </div>
        <hr style="height: 2px; width: 100%; border: 0; background-color: #000; opacity: 0.5;">
        <div style="flex: 1;">
            <div style="border: 2px solid #4A90E2; border-radius: 10px; padding: 20px; background-color: #191970; box-shadow: 0px 0px 20px #4A90E2;">
                <h3 style="color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Basic Statistics</h3>
                <div style="overflow-x: auto;">
                    {}
                </div>
            </div>
        </div>
    </div>
    """.format(
            "<table><tr><th>Column</th><th>Data Type</th></tr>"
            + "".join(
                [
                    f"<tr><td>{col.strip()}</td><td>{str(dtype).strip()}</td></tr>"
                    for col, dtype in zip(data.columns, data.dtypes)
                ]
            )
            + "</table>",
            data.describe().to_html(classes=["dataframe"], header="true"),
        ),
        unsafe_allow_html=True,
    )


def visualizations_page():
    # Set the title of the app
    st.title("Temperature Data Dashboard :bar_chart:")
    # Select a visualization option
    visualization_option = st.selectbox(
        "Select a Visualization",
        (
            "Monthly Trends",
            "Heatmap",
            "Line Trend",
            "Seasonal Decomposition",
            "3D Surface Plot with Contours",
            "Parallel Coordinates",
            "Polar Scatter",
            "Animated Heatmap",
            "Histogram of Annual Temperature",
            "Time Series Plot",
            "Overall temperature distributions",
            "Box Plot of Monthly Temperatures",
            "3D Scatter Plot of Annual Temperature Trends",
        ),
    )

    data = load_data()

    description_style = """
        font-size: 18px;
        color: #3366ff;
        margin-bottom: 10px;
    """

    emoji_style = """
        font-size: 28px;
        margin-right: 10px;
    """

    if visualization_option == "Monthly Trends":
        st.header("Monthly Temperature Trends")
        st.markdown(
            """
        <span style="{}">📊 Explore the monthly temperature trends over the years.</span>
        """.format(
                description_style
            ),
            unsafe_allow_html=True,
        )
        fig = go.Figure()

        # Add traces for each month
        for month in data.columns[1:13]:
            fig.add_trace(
                go.Scatter(x=data["YEAR"], y=data[month], mode="lines", name=month)
            )

        # Update layout
        fig.update_layout(title="Interactive Time Series Plot with Range Selector")
        fig.update_xaxes(title="Year", rangeslider_visible=True)

        # Display the plot
        st.plotly_chart(fig)

    elif visualization_option == "Heatmap":
        st.header("Monthly Temperature Heatmap")
        st.markdown(
            """
        <span style="{}">🌡️ The heatmap provides a visual representation of temperature data over time.</span></br>
        <span style="{}">Whiter shades represent higher temperatures, while bluish shades indicate cooler temperatures.</span>
        """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )
        heatmap_data = data[
            [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        ]
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values, x=heatmap_data.columns, y=data["YEAR"]
            )
        )
        fig_heatmap.update_layout(title="Monthly Temperature Heatmap")
        st.plotly_chart(fig_heatmap)

    elif visualization_option == "Line Trend":
        st.header("Temperature Trend Visualization")
        st.markdown(
            """
            <span style="{}">🎻 Visualize temperature distributions over time based on violin plotting concept.</span></br>
            <span style="{}">It shows the spread of temperature values for each year.</span>
            """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )

        # Create a violin plot
        fig = px.violin(data, x="YEAR", y="ANNUAL", box=True, points="all")

        # Calculate the mean
        mean_data = data.groupby("YEAR")["ANNUAL"].mean().reset_index()

        # Create a trace for the mean
        mean_trace = px.line(
            mean_data, x="YEAR", y="ANNUAL", labels={"YEAR": "Year", "ANNUAL": "Mean"}
        )
        fig.add_trace(mean_trace.data[0])

        # Customize the appearance of the plot
        fig.update_traces(
            marker=dict(size=4, opacity=0.5), line_color="red", fillcolor="lightblue"
        )

        # Customize layout options
        fig.update_layout(
            title="Plot of Annual Temperature Trends with Mean",
            xaxis_title="Year",
            yaxis_title="Annual Temperature",
            font=dict(size=12),
            template="plotly_white",
        )

        # Display the plot
        st.plotly_chart(fig)

    elif visualization_option == "Seasonal Decomposition":
        st.header("Seasonal Decomposition")
        st.markdown(
            """
        <span style="{}">📈 Seasonal decomposition breaks down temperature data into its components: trend, seasonal, and residual.</span></br>
        <span style="{}">It helps identify recurring patterns and anomalies.</span>
        """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )
        decomposition = seasonal_decompose(data["ANNUAL"], model="additive", period=12)
        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(
            go.Scatter(
                x=data["YEAR"], y=decomposition.trend, mode="lines", name="Trend"
            )
        )
        fig_seasonal.add_trace(
            go.Scatter(
                x=data["YEAR"], y=decomposition.seasonal, mode="lines", name="Seasonal"
            )
        )
        fig_seasonal.add_trace(
            go.Scatter(
                x=data["YEAR"], y=decomposition.resid, mode="lines", name="Residual"
            )
        )
        fig_seasonal.update_layout(title="Seasonal Decomposition of Annual Temperature")
        st.plotly_chart(fig_seasonal)

    elif visualization_option == "3D Surface Plot with Contours":
        st.header("3D Surface Plot with Contours")
        st.markdown(
            """
        <span style="{}">🌋 Explore temperature trends in a 3D surface plot with contour lines.</span></br>
        <span style="{}">Contours help visualize temperature changes over time.</span>
        """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )
        fig = go.Figure(
            data=[go.Surface(z=data.iloc[:, 1:13].values, colorscale="Viridis")]
        )

        # Add contour lines
        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )

        fig.update_layout(
            scene=dict(zaxis_title="Temperature"),
            title="3D Surface Plot of Monthly Temperature Trends with Contours",
        )
        st.plotly_chart(fig)

    elif visualization_option == "Parallel Coordinates":
        st.header("Parallel Coordinates Plot")
        st.markdown(
            """
        <span style="{}">🌐 Visualize the relationship between multiple temperature attributes using parallel coordinates.</span></br>
        <span style="{}">It's a great way to spot patterns and trends.</span>
        """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )
        fig = px.parallel_coordinates(
            data,
            dimensions=data.columns[1:13],
            color="ANNUAL",
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Parallel Coordinates Plot of Monthly Temperature Trends",
        )
        st.plotly_chart(fig)

    elif visualization_option == "Polar Scatter":
        st.header("Polar Scatter Plot")
        st.markdown(
            """
        <span style="{}">🌟 Discover temperature patterns with a polar scatter plot.</span></br>
        <span style="{}">The radial axis represents temperature values, and the angular axis represents months.</span>
        """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )
        fig = px.scatter_polar(
            data,
            r=data.columns[1:13],
            theta=data.columns[1:13],
            title="Polar Scatter Plot of Monthly Temperature Trends",
        )
        fig.update_layout(polar=dict(radialaxis=dict(title="Temperature")))
        st.plotly_chart(fig)

    elif visualization_option == "Animated Heatmap":
        st.header("Animated Heatmap")
        st.markdown(
            """
        <span style="{}">🎥 Watch temperature changes over time with an animated heatmap.</span></br>
        <span style="{}">It's an intuitive way to see how temperatures evolve.</span>
        """.format(
                description_style, emoji_style
            ),
            unsafe_allow_html=True,
        )
        heatmap_data = data[
            [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]
        ].values.T
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Year", y="Month", color="Temperature"),
            x=data["YEAR"],
            y=data.columns[1:13],
            color_continuous_scale="Viridis",
            title="Animated Heatmap of Monthly Temperature Trends Over the Years",
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig)

    elif visualization_option == "Histogram of Annual Temperature":
        st.subheader("Histogram of Annual Temperature")
        st.markdown(
            """<div style="font-size: 18px; color: #3366ff; margin-bottom: 10px;">
            📊 Visualize temperature distributions over time using a histogram plot.
        </div>""",
            unsafe_allow_html=True,
        )
        fig_hist = px.histogram(
            data, x="ANNUAL", title="Histogram of Annual Temperature"
        )
        st.plotly_chart(fig_hist)

    elif visualization_option == "Time Series Plot":
        st.subheader("Interactive Time Series Plot")
        st.markdown(
            """<div style="font-size: 18px; color: #3366ff; margin-bottom: 10px;">
            📈 Explore an interactive time series plot of annual temperature data.
        </div>""",
            unsafe_allow_html=True,
        )
        data["YEAR"] = pd.to_datetime(data["YEAR"], format="%Y")
        fig_timeseries = px.line(
            data,
            x="YEAR",
            y="ANNUAL",
            title="Interactive Time Series Plot of Annual Temperature",
        )
        st.plotly_chart(fig_timeseries)

    elif visualization_option == "Overall temperature distributions":
        st.subheader("Overall temperature distributions")
        st.markdown(
            """<div style="font-size: 18px; color: #3366ff; margin-bottom: 10px;">
            🎻 Visualize overall temperature distributions using a violin plot.
        </div>""",
            unsafe_allow_html=True,
        )
        # Create a violin plot
        fig = px.violin(data, x="YEAR", y="ANNUAL", box=True, points="all")
        st.plotly_chart(fig)

    elif visualization_option == "Box Plot of Monthly Temperatures":
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=data.iloc[:, 1:13], orient="v", palette="Set2", showfliers=True
        )
        plt.title("Box Plot of Monthly Temperatures with Outliers")
        st.markdown(
            """<div style="font-size: 18px; color: #3366ff; margin-bottom: 10px;">
            🌐 Visualize monthly temperature distributions with a box plot.
        </div>""",
            unsafe_allow_html=True,
        )
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot()

    elif visualization_option == "3D Scatter Plot of Annual Temperature Trends":
        st.subheader("3D Scatter Plot of Annual Temperature Trends")
        st.markdown(
            """<div style="font-size: 18px; color: #3366ff; margin-bottom: 10px;">
            🌋 Explore a 3D scatter plot of annual temperature trends.
        </div>""",
            unsafe_allow_html=True,
        )
        fig = px.scatter_3d(
            data,
            x="YEAR",
            y="ANNUAL",
            z="ANNUAL",
            color="ANNUAL",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig)


html_content = """
<style>
    .animated-block span {
        opacity: 0;
        animation: fadeIn 1s ease-in-out forwards;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    .card {
        border: 2px solid #4A90E2;
        border-radius: 10px;
        padding: 20px;
        background-color: #EFEFEF;
        box-shadow: 0px 0px 20px #4A90E2;
    }

    .glowing {
        animation: glow 1s ease-in-out infinite alternate;
    }

    @keyframes glow {
        0% {
            box-shadow: 0px 0px 20px #4A90E2;
        }
        100% {
            box-shadow: 0px 0px 40px #4A90E2, 0px 0px 60px #4A90E2;
        }
    }
</style>
<h1 align="center">
    <span style="font-size: 1.2em; color: #FFFF00; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);"> Acknowledgments </span>
</h1>
<div align="center">
    <div class="animated-block">
        <h2 align="center">
            <span style="font-size: 1.8em; color: #191970; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">About the Creator</span>
        </h2>
        <hr>
        <div class="card glowing">
            <p align="center">
                <span style="font-size: 1.3em; color: #FC0344; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Hello, I'm Abir Maiti!</span>
            </p>
            <p align="center">
                <span style="font-size: 1.2em; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">A Passionate Data Scientist and AI enthusiast</span>
            </p>
            <p align="center">
                <span style="font-size: 1.2em; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Bringing Data to Life Through Visualization</span>
            </p>
            <p align="center">
                <span style="font-size: 1.3em; color: #FC0344; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
                    A results-driven and analytical individual with a strong background in data science along with Python, SQL, MS Excel, and Tableau. I like to listen to music, study new technology news, delve into history and culture, etc., in my spare time. Connect with me using the below links.
                </span>
            </p>
            <p align="center">
                <a href="https://bit.ly/Abirgit44" style="text-decoration: none;">
                    <img src="https://img.shields.io/badge/GitHub-Profile-blue?logo=github" alt="GitHub Profile">
                </a>
            </p>
            <p align="center">
                <a href="https://bit.ly/linkAbir" style="text-decoration: none;">
                    <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin" alt="LinkedIn Profile">
                </a>
            </p>
        </div>
    </div>
</div>
<hr style="height: 2px; border-width: 0; color: #191970; background-color: #191970; opacity: 0.8;">
<p align="center">
    <span style="font-size: 1.2em; color: #FFFF00; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Thank you for exploring this app!</span>
</p>
<p align="center">
    <span style="font-size: 1.2em; color: #FC0344; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">Your support and curiosity drive its success!</span>
</p>
"""


def acknowledgment_page():
    st.markdown(html_content, unsafe_allow_html=True)


# Create a dictionary to map page names to functions
pages = {
    "🏠 Home": home_page,
    "📈 Data": data_page,
    "📊 Visualizations": visualizations_page,
    "🙏 Acknowledgment": acknowledgment_page,
}

# Create a sidebar with page selection using a selectbox
st.sidebar.markdown(
    """<h2 style="color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); font-size: 24px; animation: glowe 2s infinite;">Navigation Pane</h2>

<style>
  @keyframes glowe {
    0% {
      text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    50% {
      text-shadow: 2px 2px 8px rgba(74, 144, 226, 0.8);
    }
    100% {
      text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
  }
</style>
""",
    unsafe_allow_html=True,
)
selected_page = st.sidebar.selectbox("Go to", list(pages.keys()))

# Display the selected page content
pages[selected_page]()

# Custom content for each page in the sidebar
def home_page_sidebar():
    st.sidebar.markdown(
        """<p style="font-size: 18px; color: #FFD700; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        <span style="font-size: 28px; margin-right: 10px;">🏠</span> Welcome to the Temperature Explorer!
    </p>
    <p style="font-size: 16px; color: #EFEFEF; margin-top: 10px;">
        Embark on a fascinating journey through decades of temperature data.
        Explore, visualize, and analyze temperatures from 1901 to 2021.
        Uncover seasonal patterns, trends, and more in this user-friendly app.
    </p>
    """,
        unsafe_allow_html=True,
    )


def data_page_sidebar():
    st.sidebar.markdown(
        """
    <p style="font-size: 18px; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        <span style="font-size: 28px; margin-right: 10px;">📈</span> Dive into the heart of the dataset!
    </p>
    <p style="font-size: 16px; color: #EFEFEF; margin-top: 10px;">
        The Data Page provides access to the treasure trove of temperature data spanning 1901-2021.
        Discover rich insights, statistics, and visualizations in this extensive dataset sourced from IMD Pune via <a href="https://data.gov.in/" style="text-decoration: none; color: #4A90E2;">data.gov.in</a>.
    </p>
    """,
        unsafe_allow_html=True,
    )


def visualizations_page_sidebar():
    st.sidebar.markdown(
        """
    <p style="font-size: 24px; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">📊 Explore Temperature Data Visually</p>
    <p style="font-size: 18px; color: #EFEFEF; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); opacity: 0.8;">
        Immerse yourself in a spectrum of temperature insights. From animated heatmaps to violin plots,
        these visualizations unveil patterns and trends in temperature data. 🌡️
        <strong>Choose your preferred visualizations from the <i>selectbox</i> in the main page.</strong>
    </p>
    """,
        unsafe_allow_html=True,
    )


def acknowledgment_page_sidebar():
    st.sidebar.markdown(
        """<p style="font-size: 18px; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);">
        <span style="font-size: 28px; margin-right: 10px;">🙏</span> Acknowledgments
    </p>
    <p style="font-size: 16px; color: #EFEFEF; margin-top: 10px;">
        This project was made possible by the my passion for the field of data science.
        Special thanks to the data provider, without whom this app would not be feasible. Your support fuels my commitment to environmental data exploration.
    </p>
    """,
        unsafe_allow_html=True,
    )


# Create a dictionary to map page names to their respective sidebar content functions
sidebar_content = {
    "🏠 Home": home_page_sidebar,
    "📈 Data": data_page_sidebar,
    "📊 Visualizations": visualizations_page_sidebar,
    "🙏 Acknowledgment": acknowledgment_page_sidebar,
}

# Display the custom sidebar content based on the selected page
sidebar_content[selected_page]()
