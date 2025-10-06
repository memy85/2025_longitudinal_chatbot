import pickle
import requests
from pathlib import Path
import os, sys
import logging
import random
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_FILE_PATH = Path(__file__).absolute()
LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

class Config :

    def __init__(self, config_file) :
        self.file = config_file

    @property 
    def project_path(self) :
        return Path(self.file['project_path'])

    def model_path(self, model_name) :
        return self.file['model_path'][model_name]

    @property 
    def device(self) :
        return self.file['device']
    
    def load_model(self, model_name) :
        model_path = self.model_path(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    def load_logger(self, filename) :
        id = random.randrange(0,10000)
        filepath = self.project_path.joinpath(f"logs/{filename}_{id}.log").as_posix()
        print(f"logging at {filepath} / id : {id}")
        logging.basicConfig(filename=filepath, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
        return logging
    
    def send_messages(self, message) :
        url = 'https://api.line.me/v2/bot/message/push'

        # Request headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
        }

        # Request body
        data = {
            'to': LINE_USER_ID,
            'messages': [
                {
                    'type': 'text',
                    'text': message
                }
            ]
        }

        # Send the POST request
        response = requests.post(url, headers=headers, json=data)

        # Check the response
        if response.status_code == 200:
            print('Message sent successfully')
        else:
            print(f'Failed to send message. Status code: {response.status_code}, Response: {response.text}')


def load_config() :
    project_path = CURRENT_FILE_PATH.parents[1]
    config_path = project_path.joinpath("config/config.yaml")

    with open(config_path) as f : 
        config = yaml.load(f, yaml.SafeLoader)

    return Config(config)

config = load_config()
PROJECT_PATH = config.project_path


def send_line_message(message):
    # LINE Messaging API endpoint
    url = 'https://api.line.me/v2/bot/message/push'

    # Request headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}'
    }

    # Request body
    data = {
        'to': LINE_USER_ID,
        'messages': [
            {
                'type': 'text',
                'text': message
            }
        ]
    }
    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Check the response
    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print(f'Failed to send message. Status code: {response.status_code}, Response: {response.text}')

    


class BaseAgent : 

    def _init__(self) :
        self.model = None
        pass

    def respond(self, prompt) :
        self.model.generate(None)

        return 

