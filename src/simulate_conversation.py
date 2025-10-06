"""
Simulation Code
written by Won Seok Jang 10/06/2025
"""
import os, sys
from typing import List, Dict
import pandas as pd
import torch
import vllm 
from vllm import SamplingParams, LLM
import jinja2
import argparse
from openai import OpenAI, AsyncOpenAI
from utils import *
# from langchain_community.llms import VLLM
# from langchain.chat_models import init_chat_model

DATA_PATH = PROJECT_PATH.joinpath("data/processed")
PROMPT_PATH = PROJECT_PATH.joinpath("prompts/")

GRADE_LEVEL = [('elementary', 4), ('middle school', 7), ('high school', 10), ('college', 13), ('graduate', 16)]
MAX_TURN = 20


def invoke_model(model, params : SamplingParams, prompt : List[Dict]) :
    output = model.generate(prompt, params)
    output_text = output.outputs[0].text
    return output_text

def load_prompt(name) :
    loader = jinja2.FileSystemLoader(PROMPT_PATH)
    prompt = loader.load(name + ".jinja")
    return prompt

def format_prompt(prompt_name : str, information : Dict ) : 
    prompt = load_prompt(prompt_name)
    formatted_prompt = prompt.render(information)

    return  formatted_prompt

def load_patient_profile(df : pd.DataFrame, patient_idx : int, readability_score : int) : 

    patient_profile = {}
    patient_information = df[(df.subject_idx == patient_idx) & (df.read_level_score == readability_score)].iloc[0].to_dict()
    
    patient_profile['subject_id'] = patient_information['subject_id']
    patient_profile['age'] = patient_information['anchor_age']
    patient_profile['gender'] = patient_information['gender']
    patient_profile['readability_score'] = patient_information['read_level_score']
    patient_profile['readability_level'] = patient_information['read_level']

    return patient_profile


def call_chatbot_noteaid(agent : OpenAI, conversation_history : List[str], prompt_name : str, agent_info : Dict) :

    # format the prompt
    agent_info['conversation_history'] = conversation_history
    formatted_prompt = format_prompt(prompt_name, agent_info)

    outputs = agent.completions.create(
            prompt=[{"role" : "user", "content" : formatted_prompt}],
            max_tokens=200,
            temperature=0.1
            )

    response = outputs.choices[0].message
    return response

def argument_parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_idx", type=int, default=0)
    parser.add_argument("readability", type=int, default=4)
    args = parser.parse_args()
    return args

def main() :

    args = argument_parser()
    patient_idx = args.patient_idx
    readability_score = args.readability

    patient_idx=0
    readability_score=4

    # Load dataset
    simulation_df = pd.read_json(DATA_PATH.joinpath("mimic_notes_with_readability.json"), lines=True, orient='records')

    # load patient information
    patient_information = load_patient_profile(simulation_df, patient_idx, readability_score)

    # educator information 
    note = simulation_df[(simulation_df.subject_idx == patient_idx) & (simulation_df.read_level_score == readability_score)].iloc[0]['text']
    educator_information = {"medical_note" : note}

    # load agent urls
    # load chatbot noteaid, load virual patient + patient profile
    educator_agent = OpenAI(base_url="http://localhost:18001/v1")
    patient_agent = OpenAI(base_url="http://localhost:18000/v1")

    # start conversation
    conversation_record = []
    TURN_COUNT = 0
    while True :
        
        # educator response
        educator_utterance = call_chatbot_noteaid(educator_agent, conversation_record, "educator_prompt", educator_information)
        conversation_record.append(f"CHATBOT: {educator_utterance}")

        # patient response
        patient_utterance = call_chatbot_noteaid(patient_agent, conversation_record, "patient_prompt", patient_information)
        conversation_record.append(f"PATIENT: {patient_utterance}")
        
        TURN_COUNT += 1
        if TURN_COUNT >= MAX_TURN : 
            break 

    # save the whole simulate conversation
    with open(DATA_PATH.joinpath(f"conversation_record_{}.pkl"), "wb") as f :
        pickle.dump(conversation_record, f)
    


if __name__ == "__main__" :
    main()


