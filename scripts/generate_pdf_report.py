from typing import Dict, Union, List, Tuple
import pathlib
import json
import os
import datetime as dt
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from jinja2 import Environment, FileSystemLoader


def generate_interview_report(
    payload: Dict[str, Dict[str, Union[float, int, str]]]
) -> None:
    """
    Generate the interviewer assessment report by parsing the payload

    The report will analyze and provide relevant questions based on the strengths and weaknesses of the candidate

    Args:
        param1(Dict[str, Dict[str, int | str]]): The candidate's profile and assessment results

    Returns:
        None

    Raises:
        TypeError: Must receieve nested dictionaries as an argument

    Notes:
        The PDF report generated is stored in the results directory
    """

    _validate_payload(payload)
    (
        dict_candidate,
        dict_job_fitment,
        df_all_scores,
        list_series_agent_scores,
    ) = _parse_payload(payload)
    _generate_job_fitment_bar(dict_job_fitment)
    _generate_gauge_charts(df_all_scores["Self"])
    _generate_spider_plot(*list_series_agent_scores)
    _generate_final_report(dict_candidate, df_all_scores["Self"])


def _validate_payload(payload: Dict[str, Dict[str, Union[float, int, str]]]) -> None:
    """
    Data validation step to make sure that the input is of the right type

    Args:
        param1(Dict[str, Dict[str, int | str]]): The candidate's profile and assessment results

    Returns:
        None

    Raises:
        TypeError: Must receieve nested dictionaries as an argument
    """
    if not isinstance(payload, dict):
        raise TypeError(
            "Input must be nested dictionaries with values as either string, int, or float"
        )

    stack = [payload]
    while stack:
        current_dict = stack.pop()

        for key, value in current_dict.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Input must be nested dictionaries with values as either string, int, or float"
                )

            if isinstance(value, dict):
                stack.append(value)
            elif not (
                isinstance(value, float)
                or isinstance(value, int)
                or isinstance(value, str)
            ):
                raise TypeError(
                    "Input must be nested dictionaries with values as either string, int, or float"
                )


def _parse_payload(
    payload: Dict[str, Dict[str, Union[float, int, str]]]
) -> Tuple[Dict[str, Union[float, int, str]]]:
    """
    Parses the payload to make the data more manageable

    Args:
        param1(Dict[str, Dict[str, int | str]]): The candidate's profile and assessment results

    Returns:
        Tuple[Dict[str, Union[float, str]]]: a tuple that breaks down the payload into discrete inputs for future functions
    """
    dict_candidate = payload.pop("Candidate")
    dict_job_fitment = payload.pop("Job Fitment")
    payload = {key.lower(): value for key, value in payload.items()}
    df_all_scores = pd.DataFrame(payload).transpose()
    list_series_agent_scores = [df_all_scores[col] for col in df_all_scores.columns]
    return (dict_candidate, dict_job_fitment, df_all_scores, list_series_agent_scores)


def _generate_job_fitment_bar(dict_scores: Dict[str, Union[float, int]]) -> None:
    """
    Creates horizontal gauge chart to show how the employee's resume compared to the job description.

    Args:
        param(Dict[str, Union[float, int]]): a dictionary that corresponds to how close the candidate's
        resume and the population's resumes compared to the job descriptino

    Returns:
        None
    """

    R1, R2, score = dict_scores["R1"], dict_scores["R2"], dict_scores["S"]

    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_axes([0.1, 0.35, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.25])
    ax3 = fig.add_axes([0.1, 0.7, 0.8, 0.1])

    left_color = "#FCBEC1"
    right_color = "#D9FBC8"
    center_color = "#F4F4F4"

    # Create a colormap for the left half (green to white)
    cmap_left = mcolors.LinearSegmentedColormap.from_list(
        "LeftCmap", [left_color, center_color]
    )

    # Create a colormap for the right half (white to red)
    cmap_right = mcolors.LinearSegmentedColormap.from_list(
        "RightCmap", [center_color, right_color]
    )

    # white cmap
    cmap_white = mcolors.LinearSegmentedColormap.from_list(
        "WhiteCmap", ["white", "white"]
    )

    # Combine the left and right colormaps
    colors = np.vstack(
        (cmap_left(np.linspace(0, 1, 256)), cmap_right(np.linspace(0, 1, 256)))
    )
    cmap_custom = mcolors.ListedColormap(colors)

    guage_range = np.linspace(0, 100, 512)

    norm = matplotlib.colors.Normalize(vmin=guage_range[0], vmax=guage_range[-1])

    cbar = matplotlib.colorbar.ColorbarBase(
        ax,
        cmap=cmap_custom,
        norm=norm,
        orientation="horizontal",
        boundaries=guage_range,
    )

    cbar2 = matplotlib.colorbar.ColorbarBase(
        ax2,
        cmap=cmap_white,
        norm=norm,
        orientation="horizontal",
        boundaries=guage_range,
    )

    cbar3 = matplotlib.colorbar.ColorbarBase(
        ax3,
        cmap=cmap_white,
        norm=norm,
        orientation="horizontal",
        boundaries=guage_range,
    )

    cbar.outline.set_visible(False)
    cbar2.outline.set_visible(False)
    cbar3.outline.set_visible(False)

    # R1 to R2 grey haze
    ax.axvspan(R1, R2, 0, 1, facecolor="#ECECEC")
    ax2.axvspan(R1, R2, 0.85, 1, facecolor="#ECECEC")
    ax3.axvspan(R1, R2, 0, 0.3, facecolor="#ECECEC")

    # score tick
    ax.axvspan(score - 0.5, score + 0.5, 0, 1, facecolor="#000000")
    ax2.axvspan(score - 0.5, score + 0.5, 0.6, 1, facecolor="#000000")
    ax3.axvspan(score - 0.5, score + 0.5, 0, 1, facecolor="#000000")

    path_skill_gauge_chart = (
        pathlib.Path(__file__).parent.parent / "tmp" / "job_fitment_graphic.jpg"
    )

    ax.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    annotation = plt.annotate(
        "0%", xy=(0.1, 0.15), xycoords="figure fraction", ha="center", fontsize=10
    )
    annotation2 = plt.annotate(
        "100%",
        xy=(0.9, 0.15),
        xycoords="figure fraction",
        ha="center",
        fontsize=10,
    )
    annotation3 = plt.annotate(
        str(score) + "%",
        xy=(score / 125 + 0.1, 0.85),
        xycoords="figure fraction",
        ha="center",
        fontsize=10,
    )

    annotation4 = plt.annotate(
        "R1",
        xy=(R1 / 125 + 0.1, 0.15),
        xycoords="figure fraction",
        ha="center",
        fontsize=10,
    )

    annotation5 = plt.annotate(
        "R2",
        xy=(R2 / 125 + 0.1, 0.15),
        xycoords="figure fraction",
        ha="center",
        fontsize=10,
    )

    plt.savefig(path_skill_gauge_chart, format="jpg")

    annotation.remove()
    annotation2.remove()
    annotation3.remove()
    annotation4.remove()
    annotation5.remove()


def _generate_gauge_charts(series_scores: pd.Series) -> None:
    """
    Creates gauge graphs for all skills from the individual's self-assessment and save the static image to the tmp folder

    Args:
        param(pd.Series): a pandas series that corresponds to the score receieved for each skill

    Returns:
        None
    """

    # make guage charts for only top and bottom skills
    list_top_skills, list_bottom_skills = _determine_top_and_bottom_skills(
        series_scores
    )
    list_all_skills = list_top_skills + list_bottom_skills
    series_scores = series_scores[list_all_skills]

    colors = ["lightgreen", "lightblue", "navajowhite", "salmon"]

    values = range(11)
    PI = 3.14592
    x_axis_values_polar_coords = [0, (2 / 11) * PI, (4 / 11) * PI, (7 / 11) * PI]
    x_axis_tickers = [(x + 0.5) / 11 * PI for x in range(10, -1, -1)]

    for category in series_scores.index:
        category_string = str(category) + ".jpeg"
        path_category = pathlib.Path(__file__).parent.parent / "tmp" / category_string

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1, polar=True)

        ax.bar(
            x=x_axis_values_polar_coords,
            width=[0.6, 0.6, 1.5, 1.138],
            height=0.5,
            bottom=2,
            color=colors,
            align="edge",
            linewidth=3,
            edgecolor="white",
        )

        for loc, val in zip(x_axis_tickers, values):
            plt.annotate(
                val, xy=(loc, 2.25), ha="center", fontsize=25, fontweight="bold"
            )

        plt.annotate(
            str(series_scores[category]),
            xytext=(0, 0),
            xy=(PI - ((series_scores[category] + 0.5) / 11) * PI, 2),
            arrowprops={"arrowstyle": "wedge", "color": "black", "shrinkA": 0},
            bbox={"boxstyle": "circle", "facecolor": "black", "linewidth": 1.0},
            fontsize=30,
            color="white",
            ha="center",
        )

        ax.set_axis_off()

        plt.savefig(path_category, format="jpg")
        _crop_guage_chart_image(path_category)
        plt.clf()


def _crop_guage_chart_image(path: pathlib.Path) -> None:
    """
    Crops the image of the guage chart to improve image quality

    Args:
        param(path): path to the static image

    Returns:
        None
    """
    graph = Image.open(path)

    # Calculate the crop dimensions
    width, height = graph.size
    crop_region = (0, 0, width, height // 1.5)

    # Crop the image
    cropped_image = graph.crop(crop_region)

    # Save the cropped image
    cropped_image.save(path)


def _generate_spider_plot(*args: pd.Series) -> None:
    """
    Creates spidersplot graph that displays the self-assessment scores and any other comparison scores if provided

    Args:
        param(*pd.Series): an indefinite list of series that correspond to each assessor's perception of the individual's abilities

    Returns:
        None
    """

    categories = _choose_skills_for_spider_plot(args[0])

    list_scores = [series[categories].to_list() for series in args]
    list_scores = [series + series[:1] for series in list_scores]

    # clean up strings for plotting purposes
    categories = ["\n".join(category.split(" ")) for category in categories]

    N = len(categories)
    PI = 3.14592

    # define color scheme for up to 10 comparisons
    colors = [
        "#FF0000",  # (Red)
        "#00FF00",  # (Lime Green)
    ]

    angles = [n / float(N) * 2 * PI for n in range(N)]
    angles += angles[:1]

    plt.rc("figure", figsize=(10, 10))

    ax = plt.subplot(1, 1, 1, polar=True)

    ax.set_theta_offset(PI / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color="black", size=10)
    ax.tick_params(axis="x", pad=30)

    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="black", size=10)
    plt.ylim(0, 10)

    for index, series in enumerate(list_scores):
        ax.plot(angles, series, color=colors[index], linewidth=1, linestyle="solid")
        # ax.fill(angles, series, color = colors[index], alpha = 0.5)

    ax.legend(
        [series.name for series in args], bbox_to_anchor=(-0.15, 1.1), loc="upper left"
    )

    path_spiderplot_graph = (
        pathlib.Path(__file__).parent.parent / "tmp" / "baseline_assessment.jpg"
    )
    plt.savefig(path_spiderplot_graph, format="jpg")


def _choose_skills_for_spider_plot(series_self_score: pd.Series) -> List[str]:
    """
    Algorithmn for choosing which of the 49 skills will be used in the spider plot

    Args:
        param(pd.Series): a pandas series representing the self-assessment scores

    Returns:
        List[str]: list of skills/categories selected
    """

    series_sorted = series_self_score.sort_values(ascending=True)
    list_bottom_skills = series_sorted[:5].index.to_list()
    list_top_skills = series_sorted[-5:].index.to_list()
    return list_bottom_skills + list_top_skills


def _generate_final_report(
    dict_candidate: Dict[str, str], series_self_score: pd.Series
) -> None:
    """
    Generate final report by first generating the html code and then the corresponding pdf report

    Args:
        param1(Dict[str, str]): a dictionary representing the candidate's profile
        param2(pd.Series): a pandas series representing the self-assessment scores

    Returns:
        None
    """
    _generate_html(dict_candidate, series_self_score)
    # _generate_pdf(dict_candidate)


def _generate_html(
    dict_candidate: Dict[str, str], series_self_score: pd.Series
) -> None:
    """
    Render the html file by using jinja2 and the pilot.html file to customize the html file based on the specific candidate's scores

    Args:
        param1(Dict[str, str]): a dictionary representing the candidate's profile
        param2(pd.Series): a pandas series representing the self-assessment scores

    Returns:
        None
    """
    list_top_skills, list_bottom_skills = _determine_top_and_bottom_skills(
        series_self_score
    )
    number_top_skills, number_bottom_skills = len(list_top_skills), len(
        list_bottom_skills
    )

    dict_report_text = _get_text_for_top_and_bottom_skills(
        list_top_skills, list_bottom_skills
    )

    path_templates = pathlib.Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(path_templates))
    template = env.get_template("pilot.html")

    payload = {
        "list_top_skills": list_top_skills,
        "number_top_skills": number_top_skills,
        "list_bottom_skills": list_bottom_skills,
        "number_bottom_skills": number_bottom_skills,
        "dict_report_text": dict_report_text,
        "dict_candidate": dict_candidate,
        "date": dt.date.today(),
    }

    rendered_template = template.render(payload)

    name, company = dict_candidate["name"].replace(" ", "_"), dict_candidate[
        "company_name"
    ].replace(" ", "_")
    date_today_string = dt.date.today().strftime("%Y-%m-%d")
    report_filename = "_".join([name, company, date_today_string])
    report_filename += ".html"

    path_rendered_template = (
        pathlib.Path(__file__).parent.parent / "results" / report_filename
    )

    with open(path_rendered_template, "w") as file:
        file.write(rendered_template)


def _determine_top_and_bottom_skills(series_self_score: pd.Series) -> Tuple[List[str]]:
    """
    determine which set of interview questions and skill descriptions are necessary
    based on the individual's assessment. Only select top 3 or bottom 3 skills that
    are above or below a score of 6.5

    Args:
        param(pd.Series): a pandas series representing the self-assessment scores

    Returns:
        Tuple[List[str]]: list of top and bottom skills
    """

    list_top_skills = (
        series_self_score[series_self_score > 6.5]
        .sort_values(ascending=False)
        .index.to_list()[:3]
    )
    list_bottom_skills = (
        series_self_score[series_self_score < 6.5]
        .sort_values(ascending=True)
        .index.to_list()[:3]
    )

    return list_top_skills, list_bottom_skills


def _get_text_for_top_and_bottom_skills(
    top_skills: List[str], bottom_skills: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Helper function to retrieve the text from report_dynamic_txt.json file based on the top 3 and bottom 3 skills

    Args:
        param1(List[str]): list of the top skills
        param2(List[str]): list of the bottom skills

    Returns:
        Dict[str, Dict[str, str]]: a dictionary that maps top and bottom skills to the respective text
    """
    path_text = pathlib.Path(__file__).parent.parent / "resources" / "report_text.json"
    with open(path_text) as f:
        dict_text = json.load(f)

    dict_top_bottom_skills = {}

    if top_skills:
        for skill in top_skills + bottom_skills:
            dict_top_bottom_skills[skill] = {
                key: value for key, value in dict_text[skill].items()
            }

    return dict_top_bottom_skills


def _generate_pdf(dict_candidate: Dict[str, str]) -> None:
    """
    Creates the final PDF file and saves to the results folder

    Args:
        param1(Dict[str, int | str]]): The candidate's profile

    Returns:
        None
    """
    name, company = dict_candidate["name"].replace(" ", "_"), dict_candidate[
        "company_name"
    ].replace(" ", "_")
    date_today_string = dt.date.today().strftime("%Y-%m-%d")
    report_filename = "_".join([name, company, date_today_string])
    report_filename_html = report_filename + ".html"
    report_filename_pdf = report_filename + ".pdf"

    path_html_file = (
        pathlib.Path(__file__).parent.parent / "results" / report_filename_html
    )
    path_pdf_report = (
        pathlib.Path(__file__).parent.parent / "results" / report_filename_pdf
    )


if __name__ == "__main__":
    payload = {
        "Candidate": {"name": "John Doe", "company_name": "COMPANY_NAME"},
        "Job Fitment": {"R1": 40, "R2": 80, "S": 80},
        "Achievement orientation": {"Self": 8, "Comparison": 3},
        "Adaptability": {"Self": 6, "Comparison": 9},
        "Attention to detail": {"Self": 2, "Comparison": 7},
        "Big Picture Thinking": {"Self": 4, "Comparison": 5},
        "Coaching": {"Self": 9, "Comparison": 2},
        "Collaboration Skills": {"Self": 7, "Comparison": 6},
        "Commercial Acumen": {"Self": 5, "Comparison": 4},
        "Contextualization of knowledge": {"Self": 3, "Comparison": 8},
        "Courage and risk-taking": {"Self": 1, "Comparison": 10},
        "Creative Problem Solving": {"Self": 10, "Comparison": 1},
        "Critical Thinking": {"Self": 4, "Comparison": 7},
        "Dealing with uncertainty": {"Self": 7, "Comparison": 3},
        "Developing others": {"Self": 9, "Comparison": 2},
        "Driving change and innovation": {"Self": 6, "Comparison": 4},
        "Empathetic": {"Self": 8, "Comparison": 3},
        "Empowering others": {"Self": 5, "Comparison": 6},
        "Energy, passion, and optimism": {"Self": 3, "Comparison": 9},
        "Exploring perspectives and alternatives": {"Self": 4, "Comparison": 8},
        "Fostering inclusiveness": {"Self": 7, "Comparison": 5},
        "Grit and persistence": {"Self": 6, "Comparison": 7},
        "Instilling Trust": {"Self": 9, "Comparison": 2},
        "Learnability": {"Self": 8, "Comparison": 4},
        "Motivating and inspiring others": {"Self": 3, "Comparison": 6},
        "Negotiation and Persuasion": {"Self": 2, "Comparison": 9},
        "Openness to feedback": {"Self": 5, "Comparison": 8},
        "Organizational awareness": {"Self": 7, "Comparison": 3},
        "Ownership and accountability": {"Self": 4, "Comparison": 6},
        "Planning": {"Self": 6, "Comparison": 5},
        "Positive Mindset": {"Self": 9, "Comparison": 3},
        "Presentation Skills": {"Self": 8, "Comparison": 4},
        "Project management": {"Self": 2, "Comparison": 7},
        "Promoting a culture of respect": {"Self": 6, "Comparison": 8},
        "Purpose-driven": {"Self": 9, "Comparison": 7},
        "Resilience": {"Self": 8, "Comparison": 6},
        "Role Modeling": {"Self": 7, "Comparison": 9},
        "Self-confidence": {"Self": 8, "Comparison": 7},
        "Self-control and regulation": {"Self": 7, "Comparison": 8},
        "Self-directedness": {"Self": 6, "Comparison": 9},
        "Self-motivation": {"Self": 9, "Comparison": 6},
        "Speaking with conviction": {"Self": 7, "Comparison": 8},
        "Storytelling": {"Self": 8, "Comparison": 7},
        "Strategic Thinking": {"Self": 9, "Comparison": 6},
        "Synthesizing messages": {"Self": 7, "Comparison": 9},
        "Time management and prioritization": {"Self": 8, "Comparison": 7},
        "Unconventional approach (breaking stereotypes and barriers)": {
            "Self": 9,
            "Comparison": 6,
        },
        "Understanding of the external environment": {"Self": 7, "Comparison": 8},
        "Understanding one's emotions": {"Self": 8, "Comparison": 7},
        "Understanding one's strengths": {"Self": 9, "Comparison": 6},
        "Vision Alignment": {"Self": 7, "Comparison": 9},
        "Voice, articulation, and diction": {"Self": 8, "Comparison": 7},
    }

    generate_interview_report(payload)
