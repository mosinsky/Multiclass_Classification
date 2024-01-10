# client.py
from gradio_client import Client
import ast

user_input = input("Enter your complaint text: ")
client = Client("http://127.0.0.1:7860/")
result = client.predict(
    {"Complaint": user_input},  # Send the user input as a dictionary
    api_name="/predict"
)

result_dict = ast.literal_eval(result[0])
cleaned_text = result[1].replace("Complaint ", "")  # Remove the string "Complaint"

# Display the results in console
print(f"{result_dict['Complaint']}")
print(f"Cleaned text: {cleaned_text}")
print(f"Prediction: {result[2]}")