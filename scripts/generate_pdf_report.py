from typing import Dict, Union, List, Tuple
from textwrap import wrap
import pathlib
import json
import os
import datetime as dt
from PIL import Image
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import weasyprint
from jinja2 import Environment, FileSystemLoader


def generate_interview_report(payload: Dict[str, Dict[str, Union[float, int, str]]]) -> None:
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
    dict_candidate, df_all_scores, list_series_agent_scores = _parse_payload(payload)
    _generate_gauge_charts(df_all_scores["Self"])
    _generate_graph(*list_series_agent_scores)
    _generate_final_report(dict_candidate, df_all_scores["Self"])
    _delete_temp_files()


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
            elif not (isinstance(value, float)  or isinstance(value, int) or isinstance(value, str)):
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
    df_all_scores = pd.DataFrame(payload).transpose()
    list_series_agent_scores = [df_all_scores[col] for col in df_all_scores.columns]
    return (dict_candidate, df_all_scores, list_series_agent_scores)


def _generate_gauge_charts(series_scores: pd.Series) -> None:
    """
    Creates gauge graphs for all skills from the individual's self-assessment and save the static image to the resources/temp folder

    Args:
        param(pd.Series): a pandas series that corresponds to the score receieved for each skill

    Returns:
        None
    """
    colors = ["lightgreen", "lightblue", "navajowhite", "salmon"]

    values = range(11)
    PI = 3.14592
    x_axis_values_polar_coords = [0, (2 / 11) * PI, (4 / 11) * PI, (7 / 11) * PI]
    x_axis_tickers = [(x + 0.5) / 11 * PI for x in range(10, -1, -1)]

    for category in series_scores.index:
        category_string = str(category) + ".jpeg"
        path_category = (
            pathlib.Path(__file__).parent.parent
            / "resources"
            / "temp"
            / category_string
        )

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


def _generate_graph(*args: pd.Series) -> None:
    """
    Creates spidersplot graph that displays the self-assessment scores and any other comparison scores if provided

    Args:
        param(*pd.Series): an indefinite list of series that correspond to each assessor's perception of the individual's abilities

    Returns:
        None
    """
    categories = ["\n".join(wrap(category, 15)) for category in args[0].index]

    list_scores = [series.to_list() for series in args]
    list_scores = [series + series[:1] for series in list_scores]

    N = len(categories)
    PI = 3.14592

    # define color scheme for up to 10 comparisons
    colors = [
        "#FF0000",  # (Red)
        "#00FF00",  # (Lime Green)
        "#0000FF",  # (Blue)
        "#FFFF00",  # (Yellow)
        "#FF00FF",  # (Magenta)
        "#00FFFF",  # (Cyan)
        "#800080",  # (Purple)
        "#FFA500",  # (Orange)
        "#008000",  # (Green)
        "#FFC0CB",  # (Pink)
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
        pathlib.Path(__file__).parent.parent
        / "resources"
        / "temp"
        / "baseline_assessment.jpg"
    )
    plt.savefig(path_spiderplot_graph, format="jpg")


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
    _generate_pdf(dict_candidate)


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
    # determine which set of interview questions and skill descriptions are necessary
    # based on the individual's assessment. Only select top 3 or bottom 3 skills that
    # are above or below a score of 6.5
    list_top_skills = series_self_score[
        (series_self_score > 6.5) & (series_self_score.rank(ascending=False) <= 3)
    ].index.to_list()
    list_bottom_skills = series_self_score[
        (series_self_score < 6.5) & (series_self_score.rank(ascending=True) <= 3)
    ].index.to_list()
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
        "report_text": dict_report_text,
        "company": dict_candidate["company"],
    }

    rendered_template = template.render(payload)

    path_rendered_template = (
        pathlib.Path(__file__).parent.parent / "templates" / "rendered_template.html"
    )

    with open(path_rendered_template, "w") as file:
        file.write(rendered_template)


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
    path_text = (
        pathlib.Path(__file__).parent.parent / "resources" / "report_dynamic_text.json"
    )
    with open(path_text) as f:
        dict_text = json.load(f)

    dict_top_bottom_skills = {
        "Top": {"Skills Text": {}, "Interview Questions": {}},
        "Bottom": {"Skills Text": {}, "Interview Questions": {}},
    }

    if top_skills:
        for skill in top_skills:
            dict_top_bottom_skills["Top"]["Skills Text"][skill] = dict_text["Skills"][
                skill
            ]["Positive"]
            dict_top_bottom_skills["Top"]["Interview Questions"][skill] = dict_text[
                "Interview Questions"
            ][skill]

    if bottom_skills:
        for skill in bottom_skills:
            dict_top_bottom_skills["Bottom"]["Skills Text"][skill] = dict_text[
                "Skills"
            ][skill]["Negative"]
            dict_top_bottom_skills["Bottom"]["Interview Questions"][skill] = dict_text[
                "Interview Questions"
            ][skill]

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
    report_filename += ".pdf"

    path_html_file = (
        pathlib.Path(__file__).parent.parent / "templates" / "rendered_template.html"
    )
    path_pdf_report = pathlib.Path(__file__).parent.parent / "results" / report_filename

    weasyprint.HTML(path_html_file).write_pdf(path_pdf_report)


def _delete_temp_files() -> None:
    """
    Deletes all files that were created except for the PDF file (images/graphs and html/css)

    Args:
        None

    Returns:
        None
    """
    directory = pathlib.Path(__file__).parent.parent / "resources" / "temp"

    # Get a list of all files in the directory
    file_list = os.listdir(directory)

    # Iterate over the file list and delete each file
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    path_html_file = (
        pathlib.Path(__file__).parent.parent / "templates" / "rendered_template.html"
    )
    os.remove(path_html_file)


if __name__ == '__main__':
    payload =         {
            "Candidate": {
                "name": "John Doe",
                "company": "COMPANY_NAME"
            },
            "Self-Confidence": {
                "Self": 9,
                "Comparison": 2
            },
            "Resilience": {
                "Self": 8,
                "Comparison": 6
            },
            "Achievement Orientation": {
                "Self": 6.3,
                "Comparison": 5.4
            },
            "Adaptability": {
                "Self": 4,
                "Comparison": 3.4
            },
            "Learnability": {
                "Self": 3,
                "Comparison": 4.2
            },
            "Ownership And Accountability": {
                "Self": 9,
                "Comparison": 2
            },
            "Energy, Passion, and Optimism": {
                "Self": 5.3,
                "Comparison": 5
            },
            "Dealing With Uncertainity": {
                "Self": 3,
                "Comparison": 9
            },
            "Grit And Persistence": {
                "Self": 9.3,
                "Comparison": 7.2
            }
        }

    generate_interview_report(payload)