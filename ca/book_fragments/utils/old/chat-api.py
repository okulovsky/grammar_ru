import requests
import json
from pathlib import Path

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

history = []
json_file = open(Path('./fragments/martin_fragments.json'))
json_data = json.load(json_file)
json_file.close()

for fragment in json_data['fragments']:
    print(fragment['text'])
    user_message = fragment['text']
    history.append({"role": "user", "content": user_message})
    data = {
        "model": "openchat_3.5.Q5_K_M.gguf",
        "messages": history
    }

    response = requests.post(url, headers=headers, json=data, verify=False)
    assistant_message = response.json()['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_message})
    print(assistant_message)
    fragment['retell'] = assistant_message

with open(Path('./fragments/martin_fragments.json'), 'w') as json_file:
    json.dump(json_file)