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
print(f"Orginal text: {result_dict['Complaint']}")
print(f"Cleaned text: {cleaned_text}")
print(f"Prediction: {result[2]}")

text = "I tried to make a transaction at a supermarket retail store, using my chase debit/atm card, but the transaction was declined. I am still able to withdraw money out of an ATM machine using the same debit card. Please resolve this issue."