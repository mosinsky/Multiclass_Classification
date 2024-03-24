from gradio_client import Client

input = input("Complaint Text: ")
client = Client("http://127.0.0.1:7860/")
result = client.predict(
		input,
		api_name="/predict"
)
print(result)



