from flask import Flask, request, jsonify
from generate_pdf_report import generate_interview_report


app = Flask(__name__)


@app.route("/recruitment_reporting/generate_interview_questions_pdf", methods=["POST"])
def generate_pdf_endpoint():
    try:
        payload = request.get_json()
        result = jsonify(generate_interview_report(payload))
        result.status_code = 200
        return result
    except Exception as e:
        error_message = str(e)
        error = jsonify({"error": error_message})
        error.status_code = 500
        return error


if __name__ == "__main__":
    app.run()
