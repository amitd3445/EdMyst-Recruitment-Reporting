from typing import Dict, Union, List, Tuple
import pathlib
import json
import math
import datetime as dt
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from jinja2 import Environment, FileSystemLoader
import weasyprint

import sys

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

    (
        dict_candidate,
        dict_job_fitment,
        df_technical_skill_scores,
        series_video_skills,
        df_recruiter_skill_scores,
        list_series_recruiter_scores,
    ) = _parse_payload(payload)

    _generate_job_fitment_bar(dict_job_fitment)
    _generate_technical_skills_bar_chart(df_technical_skill_scores)
    _generate_colorbars_video_skills(series_video_skills)

    if isinstance(df_recruiter_skill_scores, pd.Series):
        _generate_gauge_charts_recruiter_skills(df_recruiter_skill_scores)
    else:
        _generate_gauge_charts_recruiter_skills(df_recruiter_skill_scores["Self"])   
    _generate_spider_plot(*list_series_recruiter_scores)

    if isinstance(df_recruiter_skill_scores, pd.Series):
         _generate_final_report(dict_candidate, df_recruiter_skill_scores, series_video_skills.index)
    else:
        _generate_final_report(dict_candidate, df_recruiter_skill_scores["Self"], series_video_skills.index)


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
    
    dict_candidate = payload["Candidate"]
    dict_job_fitment = payload["job_fitment"]
    df_technical_skill_scores = pd.DataFrame(payload["technical_skills"]).transpose()
    
    dict_video_skills = {key: skill_dict["percentage"] for key, skill_dict in payload["video_data"]["skills"].items()}
    series_video_skills = pd.Series(dict_video_skills, name="skills")

    dict_recruiter_skills = {key.lower(): value for key, value in payload["skill_scores"].items()}
    if any(map(lambda x: isinstance(x, dict), dict_recruiter_skills.values())):
        df_recruiter_skill_scores = pd.DataFrame(dict_recruiter_skills).transpose()
        list_series_recrutier_scores = [df_recruiter_skill_scores[col] for col in df_recruiter_skill_scores.columns]
    elif any(map(lambda x: isinstance(x, int) or isinstance(x, float), dict_recruiter_skills.values())):
        df_recruiter_skill_scores = pd.Series(dict_recruiter_skills, name="skills")
        list_series_recrutier_scores = [df_recruiter_skill_scores]

    return (dict_candidate, dict_job_fitment, df_technical_skill_scores, series_video_skills, df_recruiter_skill_scores, list_series_recrutier_scores)


def _generate_job_fitment_bar(dict_scores: Dict[str, Union[float, int]]) -> None:
    """
    Creates horizontal gauge chart to show how the employee's resume compared to the job description.

    Args:
        param(Dict[str, Union[float, int]]): a dictionary that corresponds to how close the candidate's
        resume and the population's resumes compared to the job descriptino

    Returns:
        None
    """

    score = dict_scores["S"]

    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_axes([0.1, 0.35, 0.8, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.25])
    ax3 = fig.add_axes([0.1, 0.7, 0.8, 0.1])

    left_color = "#F6E3AE"
    right_color = "#AEF6BD"


    # Create a colormap for the right half (white to red)
    cmap_middle = mcolors.LinearSegmentedColormap.from_list(
        "MiddleCmap", [left_color, right_color]
    )

    # white cmap
    cmap_white = mcolors.LinearSegmentedColormap.from_list(
        "WhiteCmap", ["white", "white"]
    )


    guage_range = np.linspace(0, 100, 512)

    norm = matplotlib.colors.Normalize(vmin=guage_range[0], vmax=guage_range[-1])

    cbar = matplotlib.colorbar.ColorbarBase(
        ax,
        cmap=cmap_middle,
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

    ax.axvspan(score - 0.5, score + 0.5, 0, 1, facecolor="#2FB4F1")
    ax2.axvspan(score - 0.5, score + 0.5, 0.6, 1, facecolor="#2FB4F1")
    ax3.axvspan(score - 0.5, score + 0.5, 0, 1, facecolor="#2FB4F1")

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
        "S",
        xy=(score / 125 + 0.1, 0.85),
        xycoords="figure fraction",
        ha="center",
        fontsize=10,
    )

    plt.savefig(path_skill_gauge_chart, format="jpg")

    annotation.remove()
    annotation2.remove()
    annotation3.remove()


def _generate_technical_skills_bar_chart(df: pd.DataFrame) -> None:
    """
    Creates a bar chart that compares the individual's technical skills to the cohort
    """

    fig, ax = plt.subplots()

    bar_width = 0.35
    num_rows = df.shape[0]
    index = range(num_rows)
    index_for_ticks = [x + bar_width/2 for x in index]
    df.index = df.index.str.replace(' ', '\n')


    ax.bar(index, df['self'], bar_width, label='Self', align='center', color="#AEF6F3", edgecolor='black', linestyle='dashed')
    ax.bar([i + bar_width for i in index], df['cohort'], bar_width, label='Cohort', align='center', color="#C5F6AE", edgecolor='black', linestyle='dashed')

    ax.set_xticks(index_for_ticks)
    ax.set_xticklabels(df.index, fontsize=8)
    ax.legend()
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 8))

    ax.set_ylim(0, 10)

    path_technical_skills = pathlib.Path(__file__).parent.parent / "tmp" / "technical_skills.jpg"
    plt.savefig(path_technical_skills, format="jpg")


def _generate_colorbars_video_skills(series_scores: pd.Series) -> None:
    """
    Create colorbars for each of the video skills
    """

    for skill, score in series_scores.items():

        fig = plt.figure(figsize=(8, 2))
        ax = fig.add_axes([0.1, 0.35, 0.8, 0.4])

        score = math.floor(score/10)/10 + 0.05 if score < 100 else 0.95

        left_color = "#FB9D9D"
        right_color = "#BDF985"

        cmap_middle = mcolors.LinearSegmentedColormap.from_list(
            "MiddleCmap", [left_color, right_color]
        )

        guage_range = np.linspace(0, 100, 11)

        norm = matplotlib.colors.Normalize(vmin=guage_range[0], vmax=guage_range[-1])

        cbar = matplotlib.colorbar.ColorbarBase(
            ax,
            cmap=cmap_middle,
            norm=norm,
            orientation="horizontal",
            boundaries=guage_range,
        )

        cbar.outline.set_visible(False)

        for i in range(10, 100, 10):
            ax.axvspan(i -0.25, i + 0.25, 0, 1, color="#FFFFFF")

        ax.set_xticks([])

        annotation = plt.annotate(
            "Need \nPractice", 
            xy=(0.1, 0.15), 
            xycoords="figure fraction", 
            ha="center", 
            fontsize=12
        )
        annotation2 = plt.annotate(
            "Great \nGoing",
            xy=(0.9, 0.15),
            xycoords="figure fraction",
            ha="center",
            fontsize=12,
        )

        metric_position = 0.1 + score * 0.8

        annotation3 = plt.annotate(
        "",
        xy=(metric_position, 0.65),
        xytext=(metric_position, 0.85),
        xycoords="figure fraction",
        ha="center",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "linewidth": 2.5, "edgecolor": "black"},
        va="center",
        )

        file_name = str(skill) + "_colorbar.jpg"
        path_skill_gauge_chart = (
            pathlib.Path(__file__).parent.parent / "tmp" / file_name
        )

        plt.savefig(path_skill_gauge_chart, format="jpg")

        annotation.remove()
        annotation2.remove()
        annotation3.remove()


def _generate_gauge_charts_recruiter_skills(series_scores: pd.Series) -> None:
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
        category_string = str(category) + "_gauge.jpg"
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
    dict_candidate: Dict[str, str], series_self_score: pd.Series, list_video_skills: List[str]
) -> None:
    """
    Generate final report by first generating the html code and then the corresponding pdf report

    Args:
        param1(Dict[str, str]): a dictionary representing the candidate's profile
        param2(pd.Series): a pandas series representing the self-assessment scores
        param3(List): a list containing all the names of the video skills

    Returns:
        None
    """
    _generate_html(dict_candidate, series_self_score, list_video_skills)
    _generate_pdf(dict_candidate)


def _generate_html(
    dict_candidate: Dict[str, str], series_recruiter_skills_score: pd.Series, list_video_skills: List[str]
) -> None:
    """
    Render the html file by using jinja2 and the pilot.html file to customize the html file based on the specific candidate's scores

    Args:
        param1(Dict[str, str]): a dictionary representing the candidate's profile
        param2(pd.Series): a pandas series representing the self-assessment scores
        param3(List): a list containing all the names of the video skills

    Returns:
        None
    """
    list_top_skills, list_bottom_skills = _determine_top_and_bottom_skills(
        series_recruiter_skills_score
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
        "list_video_skills" : list_video_skills,
        "dict_candidate": dict_candidate,
        "date": dt.date.today(),
    }

    rendered_template = template.render(payload)

    name, company = dict_candidate["name"].replace(" ", "_"), dict_candidate[
        "company"
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
        "company"
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

    weasyprint.HTML(path_html_file).write_pdf(path_pdf_report)



if __name__ == "__main__":
    file_name = pathlib.Path(__file__).parent.parent / "data" / "sample_video_data.json"
    with open(file_name) as file:
        payload = json.load(file)
    
    payload = payload['Records'][0]["Sns"]["Message"]

    generate_interview_report(payload)
