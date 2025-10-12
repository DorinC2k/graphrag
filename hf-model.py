import requests

def query(payload):
	headers = {
		"Accept" : "application/json",
		"Authorization": "Bearer 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",
		"Content-Type": "application/json"
	}
	response = requests.post(
		"https://cgglxt1hbsrcvegq.us-east-1.aws.endpoints.huggingface.cloud",
		headers=headers,
		json=payload
	)
	return response.json()

output = query({
	"inputs": "This soundtrack is beautiful! It paints the senery in your mind so well.",
	"parameters": {}
}) 

print(output)
