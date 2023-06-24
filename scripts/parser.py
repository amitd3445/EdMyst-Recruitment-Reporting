import csv
import re
import pathlib
import json
import unicodedata
from typing import List


def parse_data(
    path_report_text_csv_input: pathlib.PosixPath,
    path_report_text_json_output: pathlib.PosixPath,
) -> None:
    """
    Receieves paths to a csv file that contains all text associated with the recruitment
    report. Parses the data and creates two json files which are used for generating the report

    Args:
        param1(pathlib.PosixPath): path to the input csv file containing raw text
        param2(pathlib.PosixPath): path for where the focus area json file is stored

    Returns:
        None
    """

    with open(path_report_text_csv_input, "r") as csv_file:
        reader = csv.reader(csv_file)
        header_row = next(reader)

        skills_dict = {}

        for row in reader:
            skill = row[0].lower()

            for i, cell in enumerate(row):
                # Replace \u2019 with a regular apostrophe
                replaced_cell = (
                    unicodedata.normalize("NFKD", cell)
                    .encode("ascii", "ignore")
                    .decode("utf-8")
                )
                row[i] = replaced_cell

            skills_dict[skill] = {}

            # operational definition
            skills_dict[skill]["Operational Definition"] = row[1]

            # potential strength
            setence_potential_strength = break_sentence(row[2])
            skills_dict[skill]["Potential Strength"] = {}
            skills_dict[skill]["Potential Strength"][
                "Summary"
            ] = setence_potential_strength.pop(0)
            skills_dict[skill]["Potential Strength"][
                "Details"
            ] = setence_potential_strength

            # development considerations
            setence_potential_strength = break_sentence(row[3])
            skills_dict[skill]["Development Considerations"] = {}
            skills_dict[skill]["Development Considerations"][
                "Summary"
            ] = setence_potential_strength.pop(0)
            skills_dict[skill]["Development Considerations"][
                "Details"
            ] = setence_potential_strength

            # interview questions
            skills_dict[skill]["Interview Questions"] = []
            for col_number in range(4, 6):
                first_set_questions = break_sentence(row[col_number])
                skills_dict[skill]["Interview Questions"].append(
                    {
                        "Initial": first_set_questions.pop(0),
                        "Details": first_set_questions,
                    }
                )

    with open(path_report_text_json_output, "w") as json_file:
        json.dump(skills_dict, json_file)


def break_sentence(long_sentence: str) -> List[str]:
    """
    Receieves a sentence and applies a regex definition in order to break the sentence
    into multiple setences

    Args:
        param(str): long sentence

    Returns:
        List[str]: the original sentence broken up into a list
    """

    sentences = re.split(r"\d+[\.,]", long_sentence)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


if __name__ == "__main__":
    input_file = pathlib.Path(__file__).parent.parent / "resources" / "report_text.csv"
    report_file = (
        pathlib.Path(__file__).parent.parent / "resources" / "report_text.json"
    )
    parse_data(input_file, report_file)
