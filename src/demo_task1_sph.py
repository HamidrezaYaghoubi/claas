"""
	Instruction Tuning of LLM for Trait-conditioned Style Impact Caliberation
"""
import yaml     # type: ignore
import pandas as pd # type: ignore
import os
from PIL import Image # type: ignore
import gradio as gr

import torch    # type: ignore
from langchain_community.chat_models import ChatOllama # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage # type: ignore
from langchain_ollama import OllamaEmbeddings # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from pydantic import BaseModel # format LLM output as JSON # type: ignore
from unsloth import FastVisionModel, FastModel, FastLanguageModel # type: ignore
from transformers import TextStreamer # type: ignore
from unsloth.chat_templates import get_chat_template    # type: ignore
from unsloth.chat_templates import standardize_sharegpt # type: ignore
from transformers import TextIteratorStreamer

from utils import convert_to_base64, load_config, process_trait_info # type: ignore
from tqdm import tqdm   # type: ignore
from termcolor import colored # type: ignore
import threading
import random
import numpy as np
import random

import threading
generation_lock = threading.Lock()

def generate_base(**generate_kwargs):
	# summary + explanation: same model, fine-tuned adapter switched off
	with model.disable_adapter():
		model.generate(**generate_kwargs)

device ='cuda' if torch.cuda.is_available() else 'cpu'

TRAIT_VALUES = {
	"Gender": [
		"Male", "Female", "Non-binary/third gender", "Leave Blank",
	],
	"Age": [
		"18–24", "25–34", "35–44", "45–54", "55–64", "65 or older", "Leave Blank",
	],
	"Current Profession": [
		"Healthcare/Medical", "Government/Public Service",
		"Business/Finance",
		"Technology/Engineering", "Education", "Arts/Entertainment",
		"Retail/Hospitality/Food Service",
		"Skilled Trades/Labor (e.g., construction, electrician, landscaper, house cleaner)",
		"Student",
		"Unemployed/Looking for work", "Retired",
		"Other", 
		"Leave Blank",
	],
	"Race/Ethnicity" : [
		"Asian", "Black/African American", "Hispanic/Latino", 
		"Native American/Alaska Native", "Native Hawaiian/Other Pacific Islander",
		"White/Caucasian", "Other", "Leave Blank",
	],
	"Religious/Cultural Group": [
		"Christianity", "Islam", "Hinduism", "Judaism", "Buddhism", "None of the above", "Leave Blank",
	],
	"Political Affiliation": [
		"Conservative", "Apolitical/Not involved in politics", "Independent",
		"Libertarian", "Moderate", "Liberal", "Leave Blank",
	],
	"Highest Education": [
		"Less than high school", "High school diploma or equivalent", "Some college, no degree", 
		"Associate’s degree", "Bachelor’s degree", 
		"Master’s degree", "Doctoral or professional degree",
		"Leave Blank",
	],
	"Annual Household Income": [
		"Less than $25,000", "$25,000–$49,999", "$50,000–$74,999", 
		"$75,000–$99,999", "$100,000–$149,999", "$150,000 or more", 
		"Leave Blank",
	],
	"Family Status": [
		"Single, living alone", "Single, living with family", "Single Parent with children",
		"Married/Partnered, no children", "Married/Partnered, with children",
		"Multi-generation family (e.g., with parents, grandparents, or extended family)",
		"Leave Blank",
	],
}

HEALTH_TOPICS = {
	"Chronic Obstructive Pulmonary Disease (COPD)": "COPD1.1", 
	"Heart Disease": "HD1", 
	"HIV": "HIV1.1", 
	"Mental Health": "MH1.1", 
	"Nutrition": "N2.1", 
	"Substance Abuse": "SA4.1", 
	"Sexual Practice": "SP7.1", 
	"Vaccination": "V7.1", 
	"Cystic Fibrosis": "CF1.1", 
}	

health_topics = ""
for topic in HEALTH_TOPICS:
	health_topics += topic + '\n'



##########################################################
### To increase style variability to avoid repetitiveness
##########################################################
# * Style variants
style_variants = [
	"Write with a slightly informal and reflective tone.",
	"Write in a straightforward conversational tone.",
	"Write with mild emotional coloring, but still natural.",
	"Write in a calm, matter-of-fact tone.",
	"Write in a slightly narrative, flowing tone.",
	"Write in a concise but personable tone.",
	"Write in a informal, pragmatic tone, focusing on clarity and utility.",
]
# --- Add small lexical noise / synonym variation ---
lexical_flavors = [
	"Feel free to vary sentence structures slightly.",
	"Use a mix of simple and slightly complex sentences.",
	"Use a light mix of paraphrasing expressions.",
	"Feel free to choose different synonyms for common emotional words.",
	"Introduce subtle variation in connectors like 'however', 'still', or 'overall'.",
]
openers = [
	"This message",
	"From this message",
	"Through the message",
	"After seeing this message",
	"Looking at this poster",
	"Based on what this poster conveys",
	"Hmmm I think that this message",
	"Reflecting on the message here",
	"Considering what this poster is trying to say",
]
openers_generic = [
	"Hmmm when thinking about",
	"When I think about",
	"My impression about",
	"On top of my head",
	"My general thoughts about",
	"The way I see it,",
	"From my point of view on",
	"My initial take on",
	"In my own words,",
	"As I see things,",
	"Just speaking for myself,",
	"At a glance,",
]
openers_poster_summary = [
	"This poster",
	"This poster seems to",
	"My interpretation of the poster is",
	"From what this poster shows, it seems to",
    "Looking at the poster as a whole, it appears to",
    "Based on the imagery and tone, the poster seems to",
    "Visually, the poster comes across as trying to",
    "To me, this poster is trying to",
    "When I look at this poster, it feels like it aims to",
	"The poster gives me the impression that it intends to",
]
openers_explain = [
    "The reason why I think that is because",
    "To explain why I",
    "Well, to explain my thoughts",
    "To put it simply, I feel this way because",
    "My reasoning behind that is",
    "What leads me to that view is",
    "A big part of why I think that is",
    "To give some context for my view,",
    "Here’s why I lean that way:",
    "I see it that way mainly because",
    "Let me explain why I think so",
    "Thinking through it, I realize it's because",
    "To unpack my thinking a bit,"
]



""" 
	Generate LLM response given a single user prompt and input image
"""
def vlm_response(user_input, history, health_topic,
		gender, age, profession, race, religion,
		political, education, income, family_status,
		# extraversion, agreeableness, conscientiousness, neuroticism, openness,
	):
	with generation_lock:
		"""Clear any lingering state"""
		torch.cuda.empty_cache()  # Clear GPU memory

		""" [NOTE] we have not use `history` for this generation """
		# get uploaded image
		image = Image.open(user_input['files'][0]) if user_input['files'] else None
		image_uploaded = True
		if image is None:
			image = Image.new('RGB', (24,24))
			image_uploaded = False
		# image_b64 = convert_to_base64(image)
		print(health_topic)
		# print("Image uploaded:", image_uploaded)

		
		
		#################################################
		# 1. Construct traits from user inputs
		#################################################
		demo_dict = {
			"Gender": gender,
			"Age": age,
			"Current Profession": profession,
			"Race/Ethnicity": race,
			"Religious/Cultural Group": religion,
			"Political Affiliation": political,
			"Highest Education": education,
			"Annual Household Income": income,
			"Family Status": family_status,
		}
		# big5_dict = {
		# 	"Extraversion": extraversion,
		# 	"Agreeableness": agreeableness,
		# 	"Conscientiousness": conscientiousness,
		# 	"Neuroticism": neuroticism,
		# 	"Open-Mindedness": openness,
		# }
		
		demo_info = ""
		for trait, value in demo_dict.items():
			if value != "Leave Blank": # only add non-blank values
				demo_info += f"{trait}: {value}\n"
			else:
				demo_info += f"{trait}: [Not specified]\n"
		persona_score = ""
		persona_score += "Big-Five Trait Scores:\n"
		# for trait, value in big5_dict.items():
		# 	persona_score += f"{trait}: {value}\n"
		# no locus of control trait score
		locus = None

		######################################################################################
		# 1*. modify trait info based on trait selection setings
		#       demo_full: wheter include full demographic traits or only selected ones
		#       include_big5, include_facet, include_locus: include big5 / facet / locus of control traits or not
		#     format: <trait>: <value> if available; else <trait>: [Not specified]
		######################################################################################
		demo_info, persona_score, locus = process_trait_info(
			demo_info, persona_score, locus,
			demo_full=False, include_big5=True,
			include_facet=False, include_locus=False,
			train_mode=False,
		)
		# print(demo_info)
		# print(persona_score)

		###############################################
		### 		Add style variability           ###
		###############################################
		style_hint = random.choice(style_variants) # increase style variant
		lexical_hint = random.choice(lexical_flavors) # increase lexical variant
		opening_phrase = random.choice(openers) # increase opening variant
		opening_generic = random.choice(openers_generic) # increase opening variant
		opening_poster = random.choice(openers_poster_summary) # poster summary variation
		opening_explain = random.choice(openers_explain) # thought explanation
		print('Style:', style_hint)
		print('Lexical:', lexical_hint)
		print('Opening:', opening_phrase)
		print('Generic opening:', opening_generic)

		
		
		if image_uploaded:
			"""###############################################################
				Case 1: a health poster is uploaded
				=> VLM-enabled response prediction to that specific poster
			###############################################################"""
			################################################
			# * IMAGE UNDERSTANDING
			################################################
			PROMPT = (
				f"Describe the content and main message in given heatlh campaign poster and how it's related to {health_topic}. ",
				"Note that the message could be non-direct or subtle (e.g. irony, fear-driven evoke without explicit texts, etc). Only provide the answer (in 2-4 sentences). ",
				f"Start the response with {opening_poster}"	
			)
			messages = [
				{"role": "user", "content": [
					{"type": "image"},
					{"type": "text", "text": PROMPT}
				]}
			]
			input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
			inputs = tokenizer(
				image.convert("RGB"),
				input_text,
				add_special_tokens = False,
				return_tensors = "pt",
			).to(device)
			# Model inference
			gen_tokens = model.generate(
				**inputs, 
				max_new_tokens = 512,
				use_cache = True,
				# do_sample=cfgs["stochastic"],
				# temperature=cfgs["temperature"],
				# min_p=0.9,
				# min_p=0.3,
				top_k=15,
				temperature=0.8,
				do_sample=True, # cfgs["stochastic"]
			)
			outs = tokenizer.batch_decode(gen_tokens[:, inputs.input_ids.shape[1]:])[0]
			image_desc = outs.replace(tokenizer.eos_token, "")
			image_desc = image_desc.replace("<end_of_turn>", "")

			################################################
			# 2. Construct SYSTEM and USER PROMPT
			################################################
			SYSTEM_PROMPT = cfg_prompts["SYSTEM_SIM"]
			SIM_PROMPT = ""
			# prompt for role-playing information
			SIM_PROMPT += f"You are: Demographics:\n{demo_info}\n"
			# SIM_PROMPT += f"Your personality test shows you have (min score = 0; max score = 5):\nBig-Five Trait Scores:\n{persona_score}\n\n"
			# SIM_PROMPT += f"You also have {locus}\n"
			# situation description (role-playing)
			SIM_PROMPT += cfg_prompts["SIMULATION_SIM"]

			################################################
			# 3. Stage 1: VLM-enabled response prediction
			# 	 Predict Trait-aware Likert Scale Responses
			################################################
			assert cfgs["infer_engine"] == "unsloth", "Only unsloth inference is supported"
			assert cfgs["vision"] == True, "Must have vision input"
			# load a sample row to extract Likert scale questions	
			df = pd.read_csv(os.path.expandvars(cfgs["data_path"]))
			# extract sample with given health_topic for correct question set
			sample = df[df['Poster_id'] == HEALTH_TOPICS[health_topic]].iloc[0]
			del df # free memory
			""" Iterate through each question"""
			# answers_json = {}
			answers_numeric = ""
			# for question in [
			# 	"This message makes me more concerned about the health risks in the poster - Scale: 1 (not at all) - 9 (extremely)",
			# 	"The message motivates me to engage in healthier lifestyle and habit - Scale: 1 (not at all) - 9 (extremely)",
			# 	"In your opinion, how harmful is ignoring the health risks in the poster? - Scale: 1 (not at all) - 9 (extremely",
			# 	"How open are you to engaging in the activity in the poster? - Scale: 1 (not at all) - 9 (extremely)",
			# ]:
			for i in range(1,16,1):
				# a. parse specific Likert score question
				col = f"Q{i}"
				if pd.isna(sample[col]):
					continue
				question = sample[col].replace("\n", " ")
				# instruction prompt to answer in proper format
				if "type in" in question.lower():
					continue # skip free-text questions for demo
				elif "make you feel" in question.lower():
					continue # skip emotional questions: imprecise
				elif "how open" in question.lower():
					continue # skip intentional question: low-accuracy
				# b. intialize USER PROMPT with SIMULATION PROMPT 
				#    with full demographic+personality data
				USER_PROMPT = SIM_PROMPT 
				USER_PROMPT += f"Question: {question}\n\n" 
				# instruction prompt to answer in proper format
				USER_PROMPT += cfg_prompts['INSTRUCTION_MCQ']
				# c. Contruct LLM message: response prediction
				messages = [
					{"role": "user", "content": [
						{"type": "image"},
						{"type": "text", "text": SYSTEM_PROMPT + USER_PROMPT}
					]}
				]
				input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
				inputs = tokenizer(
					image.convert("RGB"),
					input_text,
					add_special_tokens = False,
					return_tensors = "pt",
				).to(device)
				# d. Model inference
				gen_tokens = model.generate(
					**inputs, 
					max_new_tokens = 16,
					use_cache = True,
					do_sample=cfgs["stochastic"],
					temperature=cfgs["temperature"],
					min_p=0.9,
				)
				outs = tokenizer.batch_decode(gen_tokens[:, inputs.input_ids.shape[1]:])[0]
				answer = outs.replace(tokenizer.eos_token, "")
				answer = answer.replace("<end_of_turn>", "")
				# answers_json[col] = answer
				answers_numeric += f"{question}. Your answer: {answer}\n"
			# print(answers_json)
			print(answers_numeric)
			
			################################################
			# 4. Stage 2: LLM Summarization of all answers
			#   => final response generation based on 
			#	   all Likert answers to the poster
			# 	=> one-shot prompting 
			################################################
			SYSTEM_PROMPT = "You are a helpful assistant."
			# USER_PROMPT = f"Please convert these questions and answers into a concise and coherent \
			# summary of your overall reactions, feelings, and perspectives about the poster: {answers_numeric} \
			# Please provide the final response only." 
			# USER_PROMPT = f"Summarize the main points from questions and answers below into a concise and coherent overall reaction to the poster:\
			# {answers_numeric}. Provide the final response only.\n" 
			USER_PROMPT = (
			"Summarize the following survey responses into a short, natural paragraph that captures your overall sentiment, motivation, and thinking. "
			f"Write as if paraphrasing what a person might say in conversation. Adjust your style based on your demographic/personality traits."
			"Do NOT repeat numeric scores. "
			"Preserve polarity: low scores → low concern/motivation/openness; high scores → high concern/motivation/openness. "
			"If answers are mixed (e.g., believes something is harmful but isn't personally moved), reflect that nuance explicitly. "
			"Keep to 1-5 sentences.\n\n"

			"**STRICTLY FOLLOW THESE RULES:**\n"
			"- Infer direction from each item's Scale description (e.g., 1-9: higher = more; 0-6: higher = more). "
			"- Use calibrated wording: 1-2 = very low, 3-4 = low, 5 = moderate, 6-7 = high, 8-9 = very high; for 0-6: 0-1 = not/slight, 2-3 = somewhat, 4-5 = high, 6 = very. "
			"- VERY IMPORTANT: provide ONLY the final summarized response, without anything else!"
			f"- The response MUST have a consistent health topic: {health_topic}. Ground each sentence to the impact of campaign message.\n"
			"- Never invert sentiment. Prefer hedged phrases (e.g., “not particularly,” “only somewhat,” “very open,” “not open at all”).\n\n"
			f"- Mimic the talking style of emulated demographic as realistic as possible."

			"**Example input 1:**\n"
			"The message makes me more concerned about the health risks of poor eating habits - Scale: 1-9. Your answer: 9\n"
			"The message motivates me to make healthy eating choices - Scale: 1-9. Your answer: 9\n"
			"In your opinion, how harmful is neglecting proper nutrition and weight management to your overall health? - Scale: 0–6. Your answer: 5\n"
			"How open are you to adopting healthier eating habits and lifestyle changes? - Scale: 1-9. Your answer: 9\n"
			"**Example output 1:**\n"
			"This message really heightened my awareness of how unhealthy eating can be. The content in the message strongly motivates me to make better choices, and I feel very ready to follow through.\n\n"

			"**Example input 2:**\n"
			"The message makes me more concerned about the health risks of COPD and smoking - Scale: 1-9. Your answer: 1\n"
			"The message motivates me to not smoke. - Scale: 1-9. Your answer: 1\n"
			"In your opinion, how harmful is smoking to your general health? - Scale: 0-6. Your answer: 6\n"
			"How open are you to smoking in the future? - Scale: 1-9. Your answer: 1\n"
			"**Example output 2:**\n"
			"From this message, I recognize smoking is very harmful, but the content in the message didn't increase my concern or motivate me much. It does somewhat make me understand that smoking is harmful, however. Anyway, I'm not open to smoking in the future.\n\n"

			"**Example input 3:**\n"
			"The message makes me more concerned about the effects of lack of exercise - Scale: 1-9. Your answer: 4\n"
			"The message motivates me to be more active - Scale: 1-9. Your answer: 3\n"
			"How open are you to exercising regularly? - Scale: 1-9. Your answer: 4\n"
			"**Example output 3:**\n"
			"Through the message, I get that exercise matters and the message raised my awareness a bit, but the poster content itself didn't really motivate me. The content in the message has some small impact in motivating me to change my routine.\n\n"

			# "**Example input 4:**\n"
			# "The message makes me more concerned about the health risks of substance abuse - Scale: 1 (not at all) - 9 (extremely). Your answer: 6\n"
			# "The message motivates me to not use substances. - Scale: 1 (not at all) - 9 (extremely). Your answer: 6\n"
			# "In your opinion, how harmful is substance use to your general health? - Scale: 0 (not at all)-6 (extremely harmful). Your answer: 5\n"
			# "How open are you to trying a substance in the future? - Scale: 1 (not at all)-9 (extremely). Your answer: 1\n"
			# "**Example output 4:**\n"
			# "This message somewhat makes me more concerned about the health risks of substance abuse motivates me not to use them. However, the message itself doesn't completely convince me that substance abuse is harmful. However, I'm not open to trying substance at all!!\n"
			f"Start the response with '{opening_phrase}'  (Style hint: {style_hint}; Lexical hint: {lexical_hint})\n"
			f"Input: {answers_numeric}. "
			)

			# Contruct LLM message
			messages = [
				{"role": "user", "content": [
					# {"type": "image"},
					{"type": "text", "text": SYSTEM_PROMPT + USER_PROMPT}
				]}
			]
			input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
			inputs = tokenizer(
				# image.convert("RGB"),
				input_text,
				add_special_tokens = False,
				return_tensors = "pt",
			).to(device)

			############################
			### Text LLM Streaming   ###
			############################
			# generation with streamer
			generate_kwargs = dict(
				**inputs,
				streamer=streamer_aux,
				max_new_tokens=512,
				use_cache=True,
				# min_p=0.3,
				top_k=15,
				temperature=0.8,
				do_sample=True, # cfgs["stochastic"]
			)
			# separate thread to run generation
			thread = threading.Thread(
				target=generate_base,
				kwargs=generate_kwargs
			)
			thread.start()
			# stream out generation
			outputs = [
				f"Emulated traits:\n {demo_info}\n" + '='*20 + "\n\n", 
				image_desc + "\n\n"
			]
			try:
				for new_token in streamer_aux:
					outputs.append(new_token)
					final_output = ''.join(outputs)
					yield final_output
			finally:
				thread.join() # the adapter must be back on before anything else runs

			# text representation of final response
			response = "".join(outputs[2:]) # ignore trait summary & image description
			print(colored('Traits', 'green'), demo_info)
			print(colored('Emulated response:', 'green'), response)
			print('='*100)

			
			################################################
			# 5. Stage 3: provide explanation (demo purpose)
			#   => condition on {trait} AND {reponse}
			################################################
			SYSTEM_PROMPT = cfg_prompts["SYSTEM_SIM"]
			SIM_PROMPT = ""
			# prompt for role-playing information
			SIM_PROMPT += f"You are: Demographics:\n{demo_info}\n"
			# SIM_PROMPT += f"Your personality test shows you have (min score = 0; max score = 5):\nBig-Five Trait Scores:\n{persona_score}\n\n"
			# SIM_PROMPT += f"You also have {locus}\n"
			# situation description (role-playing)
			SIM_PROMPT += cfg_prompts["SIMULATION_SIM"]
			SIM_PROMPT += (
				f"After seeing the uploaded impage, your response were {response}. "
				"Briefly explain WHY you responded that way, based on your demographic background. "
				f"Keep the explanation concise and direct. Start the response with '{opening_explain}' "
				f"(Style hint: {style_hint}, concise; Lexical hint: {lexical_hint}). "
				"Afterward, give a few *generic and succinct* suggestions to improve the poster's persuasiveness."
			)
			USER_PROMPT = SIM_PROMPT

			# Contruct LLM message
			messages = [
				{"role": "user", "content": [
					{"type": "image"},
					{"type": "text", "text": SYSTEM_PROMPT + USER_PROMPT}
				]}
			]
			input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
			inputs = tokenizer(
				image.convert("RGB"),
				input_text,
				add_special_tokens = False,
				return_tensors = "pt",
			).to(device)

			############################
			### Text LLM Streaming   ###
			############################
			# generation with streamer
			generate_kwargs = dict(
				**inputs,
				streamer=streamer_aux,
				max_new_tokens=512,
				use_cache=True,
				min_p=0.85,
				temperature=0.1,
				do_sample=True, # cfgs["stochastic"]
			)
			# separate thread to run generation
			thread = threading.Thread(
				target=generate_base,
				kwargs=generate_kwargs
			)
			thread.start()
			# stream out generation
			# outputs = [image_desc + "\n\n"]
			outputs += ["\n"]
			try:
				for new_token in streamer_aux:
					outputs.append(new_token)
					final_output = ''.join(outputs)
					yield final_output
			finally:
				thread.join()
			

			return answer
		else:
			"""###############################################################
				Case 2: no health poster is uploaded
				=> General Response to the health topic
				=> not conditioned on any particular health poster
			###############################################################"""
			################################################
			# 2. Construct SYSTEM and USER PROMPT
			################################################
			SYSTEM_PROMPT = (
				"You are a person with unique demographic and personality traits. "
				"Based on your background, you naturally have thoughts, feelings, and reactions to what you see."
			)
			SIM_PROMPT = ""
			# prompt for role-playing information
			SIM_PROMPT += f"You are: {demo_info}\n"
			# SIM_PROMPT += f"Your personality test shows you have (min score = 0; max score = 5): {persona_score}\n"
			# SIM_PROMPT += f"You also have {locus}\n"
			# situation description (role-playing)
			SIM_PROMPT += f"You are being asked a general question to share your *general* opinions and beliefs about a given health topic.\n"
			################################################
			# 3. LLM-enabled response prediction
			# 	 Predict Trait-aware Likert Scale Responses
			################################################
			assert cfgs["infer_engine"] == "unsloth", "Only unsloth inference is supported"
			USER_PROMPT = SIM_PROMPT 
			USER_PROMPT += (
				f"What are your *general* thoughts and opinions about the {health_topic} health topic? "
				f" What's your attitude and feeling when talking about {health_topic} in general and why?"
				f" How familiar are you with {health_topic}? How much do you care or know about it?" 
				f" Do you think {health_topic} is an important topic to talk about?"
				f" What is its impacts and importance {health_topic} in society and your life? Why?"
				f" Do you have any strong opinions about it?"
				f" Are you interested in learning more about it?"
			)
			# instruction prompt to answer in proper format
			USER_PROMPT += (
				"Your personality, locus of control, and demographic traits influence your response. Adjust your style based on your demographic personality traits.\n"
				"**STRICTLY FOLLOW THESE RULES:**\n"
				"- Human-like, casual, everyday conversational response. Only answer the questions\n"
				f"- The response MUST have a consistent health topic: {health_topic}.\n"
				# "- Answer briefly in **5-7 sentences**.\n"
				"- Only provide the answer. DO NOT REPEAT THE PROMPT!\n"
				"- Condition your response on your *demographic/personality traits provided earlier, IGNORING the [Not specified] ones*.\n"
				"- MUST provide *reasonable* and *informative* answers aligned with your background."
				f"- Start the response with '{opening_generic}' ; {style_hint} {lexical_hint}\n"
				# f"- Start the answer some variations of \'About my personal thoughts on *{health_topic}*, I \' \n"
				# f"- Start the answer with something like: When thinking about {health_topic}, I ..."
			)
			# c. Contruct LLM message
			# print("USER PROMPT:", USER_PROMPT)
			messages = [
				{"role": "user", "content": SYSTEM_PROMPT + USER_PROMPT}
			]
			assert "gemma" in cfgs["model"], "Currently only gemma model is supported for no-image input"
			input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
			inputs = tokenizer(
				input_text,
				add_special_tokens = False,
				return_tensors = "pt",
			).to(device)
			############################
			### Text LLM Streaming   ###
			############################
			# generation with streamer
			generate_kwargs = dict(
				**inputs,
				streamer=streamer,
				max_new_tokens=512,
				use_cache=True,
				# min_p=0.3,
				top_k=15,
				temperature=0.8,
				do_sample=True, # cfgs["stochastic"]
			)
			# separate thread to run generation
			thread = threading.Thread(
				target=model.generate,
				kwargs=generate_kwargs
			)
			thread.start()
			# stream out generation
			outputs = [f"Emulated traits:\n {demo_info}\n" + '='*20 + "\n\n"]
			try:
				for new_token in streamer:
					outputs.append(new_token)
					final_output = ''.join(outputs)
					yield final_output
			finally:
				thread.join()

"""###########################################################################
	Evaluate a given model (specified in model_cfgs)
	on posters with given test_style

	Args:
		+ cfgs			: specify model type (e.g. gemma or llama), 
						  data source, and export paths
		+ prompts		: set of prompts

	Outputs:
		=> save model in cfgs["export_path"] (CSV file)
			+ if cfgs["export_path"] not exists, initialize it with cfgs["data_path"] 
				=> original survey data with ground-truth responses
			+ add column "<model>:<version>": store AI-simulated responses
			+ support concurrent evaluation on different jobs
##########################################################################"""
if __name__ == '__main__':
	"""==========================================
		1. load model settings & prompts format
	=========================================="""
	######################################
	# Load model configs & prompts
	######################################
	model_cfg = "./configs/task1_demo_sph.yaml"
	prompt_cfg = "./configs/prompts.yaml"
	cfgs = load_config(model_cfg)
	cfg_prompts = load_config(prompt_cfg)

	"""==========================================
		2. Evaluate model defined in configs
	=========================================="""
	print(colored('MODEL USE:', 'green'), cfgs["model"])
	# print(prompts['SYSTEM'])
	# print(prompts['INSTRUCTION'])

	"""===============================
		3. Initialize model
		=> `model`, `tokenizer`
			are initialized here
	==============================="""
	assert cfgs["infer_engine"] == "unsloth", "Only unsloth inference is supported"
	assert cfgs["vision"] == True, "Must have vision input"
	if cfgs["vision"]:
		#################################################
		### MAIN MODEL
		###	=> response emulation, fine-tuned model
		###	=> summary + explanation use the same model
		###	   with its adapter off (generate_base)
		#################################################
		# WITH VISUAL STIMULI
		model, tokenizer = FastVisionModel.from_pretrained(
			model_name=cfgs["model"],
			load_in_4bit=True,
		)
		FastVisionModel.for_inference(model)
		if "gemma" in cfgs["model"]:
			# gemma-specific tokenizer chat template
			tokenizer = get_chat_template(
				tokenizer,
				chat_template = "gemma-3",
			)

	# initialize streamer tokens
	streamer = TextIteratorStreamer(
		tokenizer, skip_prompt=True, skip_special_tokens=True
	)
	streamer_aux = TextIteratorStreamer(
		tokenizer, skip_prompt=True, skip_special_tokens=True
	)
	
	"""=============================================
		4. User-input Dropdown Traits
	============================================="""
	#################################
	###   Gradio Interface       ###
	#################################
	with gr.Blocks(theme="gradio/dark") as interface:
		# --- Title Page with Logo ---
		LOGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/umd_logo.png"))
		gr.Image(value=LOGO_PATH, show_label=False, interactive=False, height=100)
		gr.Markdown(
			"""
			<div style="text-align: center;">
			<h1 style="margin-bottom: 0.5em;">
				UMD AI-Empowered Response Prediction in Public Health Messaging
			</h1>
			</div>

			<hr style="margin-top: 0.8em; margin-bottom: 0.8em;"> <!-- thinner spacing around line -->

			<div style="text-align: center;">
			<h2 style="margin-top: 0.3em; margin-bottom: 0.6em;">
				User Guide
			</h2>
			</div>

			<ul style="text-align: left; max-width: 800px; margin: auto;">
			<li>This program emulates <b>demographic- and personality-conditioned responses</b> to public health posters using our trait-aligned Vision-Language Model (VLM).</li>
			<li>To begin, (1) specify the target demographic traits, then (2) upload a public health poster to predict responses.</li>
			<li>If a health poster is uploaded, the model first summarizes its understanding of the image.</li>
			<li><b>Please note:</b>
				<ul>
				<li>Each interaction only uses the uploaded image and selected traits (no conversation history).</li>
				<li>You don’t need to type any text prompt; just upload the Health Poster and click <b>Submit</b>.</li>
				<li>If no poster or image is uploaded, the program automatically generates the emulated person’s <b>general opinion</b> on the selected Health Topic.</li>
				<li>Please do not interrupt the generation process as it can lead to unexpected results. In case it happens, simply refresh the web app.</li>
				<li><b>Limitation:</b> The model may generate less realistic emulations to some under-represented demographics in the survey dataset (e.g., Asian seniors). We are conducting more comprehensive survey to effectively address this limitation.</li>
				</ul>
			</li>
			</ul>

			<hr style="margin-top: 0.8em; margin-bottom: 1.2em;">
			""",
			elem_id="intro-section"
		)

		# Scroll to intro section on load
		gr.HTML("""
		<script>
		window.onload = function() {
		window.scrollTo({ top: 0, behavior: 'smooth' });
		}
		</script>
		""")

		##########################
		### Demographic Traits ###
		##########################
		gr.Markdown("## 1.a) Please specify the target demographic traits to be emulated here:")
		# Dropdowns (single-select, no custom values)
		with gr.Row():
			gender = gr.Dropdown(
				label="Gender",
				choices=TRAIT_VALUES["Gender"],
				allow_custom_value=False,
				value="Female",
			)
			age = gr.Dropdown(
				label="Age",
				choices=TRAIT_VALUES["Age"],
				allow_custom_value=False,
				value="25–34",
			)
			profession = gr.Dropdown(
				label="Current Profession",
				choices=TRAIT_VALUES["Current Profession"],  # keep given order
				allow_custom_value=False,
				value="Student",
			)
		with gr.Row():
			race = gr.Dropdown(
				label="Race/Ethnicity",
				choices=TRAIT_VALUES["Race/Ethnicity"],
				allow_custom_value=False,
				value="White/Caucasian",
			)
			religion = gr.Dropdown(
				label="Religious/Cultural Group",
				choices=TRAIT_VALUES["Religious/Cultural Group"],
				allow_custom_value=False,
				value="Leave Blank",
			)
			political = gr.Dropdown(
				label="Political Affiliation",
				choices=TRAIT_VALUES["Political Affiliation"],
				allow_custom_value=False,
				value="Leave Blank",
			)
		with gr.Row():
			education = gr.Dropdown(
				label="Highest Education",
				choices=TRAIT_VALUES["Highest Education"],
				allow_custom_value=False,
				value="Leave Blank",
			)
			income = gr.Dropdown(
				label="Annual Household Income",
				choices=TRAIT_VALUES["Annual Household Income"],
				allow_custom_value=False,
				value="$75,000–$99,999",
			)
			family_status = gr.Dropdown(
				label="Family Status",
				choices=TRAIT_VALUES["Family Status"],
				allow_custom_value=False,
				value="Leave Blank"
			)
		# ##########################
		# ### Big Five Traits    ###
		# ##########################
		# gr.Markdown("## 1.b) Please adjust the Big Five Personality Traits to be emulated:")
		# with gr.Accordion("Big Five Personality Traits (1 = very low, 5 = very high)", open=True):
		# 	gr.Markdown(
		# 		"Adjust the sliders to represent the target personality profile. "
		# 		"Leave them as-is if not applicable."
		# 	)
		# 	with gr.Row():
		# 		with gr.Column(scale=1):
		# 			openness = gr.Slider(
		# 				label="Open-Mindedness",
		# 				minimum=1, maximum=5, step=0.2, value=2.5,
		# 				interactive=True
		# 			)
		# 		with gr.Column(scale=1):
		# 			conscientiousness = gr.Slider(
		# 				label="Conscientiousness",
		# 				minimum=1, maximum=5, step=0.2, value=2.5,
		# 				interactive=True
		# 			)
		# 		with gr.Column(scale=1):
		# 			extraversion = gr.Slider(
		# 				label="Extraversion",
		# 				minimum=1, maximum=5, step=0.2, value=2.5,
		# 				interactive=True
		# 			)
		# 	with gr.Row():
		# 		with gr.Column(scale=1):
		# 			neuroticism = gr.Slider(
		# 				label="Neuroticism",
		# 				minimum=1, maximum=5, step=0.2, value=2.5,
		# 				interactive=True
		# 			)
		# 		with gr.Column(scale=1):
		# 			agreeableness = gr.Slider(
		# 				label="Agreeableness",
		# 				minimum=1, maximum=5, step=0.2, value=2.5,
		# 				interactive=True
		# 			)
		# 		gr.Column(scale=1)  # right spacer
				
		##########################
		###    Health Topic	   ###
		##########################
		gr.Markdown("## 2. Please specify the main Health Topic of the poster here:")
		gr.Markdown("""
		#### Notes:
		* Select the main Health Topic of the poster you will upload next.
		* Please make sure the selected Health Topic matches the uploaded poster for best results.
		* If you don’t upload a poster, the model will produce a general response emulating how a person with given traits would say about the selected Health Topic.
		""",
		)
		# ---- dropdown at ~50% page width and centered ----
		with gr.Row():
			with gr.Column(scale=1):
				health_topic = gr.Dropdown(
					label="Health Topic",
					choices=HEALTH_TOPICS,
					allow_custom_value=False,
				)
			gr.Column(scale=1)  # right spacer
		##########################
		###   Chat interface   ###
		##########################
		gr.Markdown("## 3. Upload Public Health Poster here (if no poster is uploaded, the model emulates General Response to the topic):")
		gr.Markdown("""
		#### Notes:
		##### Use case 1: Trait-conditioned Response Emulation to specific Health Poster 
		* Upload **one poster image at a time** — only the first image will be processed. The model does **not retain memory**, so please re-upload an image for each new submission. 
		* Ensure the uploaded poster **matches the selected Health Topic** for best results.  You do **not** need to type any text prompt — simply upload a poster and click **Submit**. 
		* Please allow our model 5-10 seconds to summarize its poster understanding before emulating the response.
		##### Use case 2: Trait-conditioned Generic Response Emulation to a Health Topic
		* Simply select a Health Topic and click "Send" (do not upload any poster). 
		"""
		)
		chat = gr.ChatInterface(
			fn=vlm_response,
			multimodal=True,  # text + image
			title=f"Vision-Language Model: Trait-Conditioned Response Emulation",
			type="messages",
			additional_inputs=[
				health_topic, gender, age, profession, race, religion,
				political, education, income, family_status,
				# extraversion, agreeableness, conscientiousness, neuroticism, openness,
			],
			chatbot=gr.Chatbot(height=600), # height=330
			autofocus=False,
		)

	"""=============================================
		5. Chat Interface Launch
	============================================="""
	interface.queue(
		max_size=20, 
		default_concurrency_limit=1,
	).launch(
		share=True, 
		max_threads=1,
	)