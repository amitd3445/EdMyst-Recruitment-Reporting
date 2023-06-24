Introduction:
    A REST API designed to generate PDF reports based on the recruitment assessment scores and the candidate's profile. Designed to be deployed on AWS. The PDF report will be saved in the results folder. All static images used to design the PDF report will be deleted after the PDF is built.

Base URL:
    /candidate_reporting/

Error Handling:
    Raises an error if the input is not a nested dictionaries with values of type float, int, or string. 
    Returns a 500 error along with the message "Input must be nested dictionaries with values as either string, int, or float"

Endpoint:
    /generate_interview_questions_pdf
Description:
    generates the pdf report that customizes the interview questions for a given candidate
URL: 
    /candidate_reporting/generate_interview_questions_pdf
HTTP Method: 
    POST
Request Parameters:
    None
Request Headers:
    Content-type: Application/JSON
Request Body:
    A json blob of this format:
    ```json
        {
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
    ```
Reponse:
    Successful invocation: 200
    Unsuccessful invocation - TypeError: 500

Response Body:
    Successful invocation: None
    Unsuccessful invocation: 
        error_message : error message