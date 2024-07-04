import requests
import json


def update_link(value):
    url = "https://cchoi.cydevaltd.com/api/v1/general/update/domain/"

    payload = json.dumps({
        "key": "DOMAIN_AI",
        "value": f"{value}/transfer/v2/"
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
