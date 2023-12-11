
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv
from typing import List
from pprint import pprint
import json
import pandas as pd



if __name__ == "__main__":

    # 환경변수 로드 , esp. OPENAI_API_KEY
    load_dotenv()
    
    # GPT-4 설정
    llm = ChatOpenAI(temperature=0.1, 
                     max_tokens=500, 
                     model="gpt-4",
                     verbose = True)
    
    # User prompt : 번역 대상 문장 목록
    user_prompt = {}
    user_prompt["src"] = ["Feeling hungry, we grabbed a pizza from the fridge and microwaved it.", 
                  "The video chronologically shows the son's finger stuck in a hole of the dog house, the rescue process by paramedics, and the conclusion of the entire incident."]
    user_prompt_str = json.dumps(user_prompt).replace('{','{{').replace('}','}}')

    # System prompt : 번역 작업 지시
    with open('prompts/1. translate.txt', 'r') as f:
        sys_prompt_str = f.read()

    # Prompt template : 번역 작업 지시 + 번역 대상 문장 목록
    prompt_template = ChatPromptTemplate.from_messages( [
        ("system", sys_prompt_str),
        ("human", user_prompt_str),
        ] )

    print(prompt_template)

    # Chaining with the above prompts
    chain = LLMChain(
        llm = llm, 
        prompt = prompt_template, 
        output_key = "output",
        verbose = True
    )

    # Make prompt with prompt template before making a request
    req = {}
    req['src_lang'] = 'english'
    req['dst_lang'] = 'korean'
    
    # Make a request to GPT
    result_str = chain(req)

    # Result Check
    result = json.loads(result_str['output'])
    pprint(result)

    # Compose result into dataframe with other info
    src_lang = req['src_lang']
    dst_lang = req['dst_lang']
    src_sentences = user_prompt['src']
    dst_sentences = result['dst']

    columns = ['src_lang', 'dst_lang', 'sentence', 'translated']
    df = pd.DataFrame(columns=columns)

    for i in range(len(src_sentences)):
        for j in range(3):
            new_row = pd.DataFrame({'src_lang': [src_lang],
                                    'dst_lang': [dst_lang],
                                    'sentence': [src_sentences[i]],
                                    'translated': [dst_sentences[i][j]]
                                    })
            df = pd.concat([df, new_row], ignore_index=True)
    
    print(df)

    df.to_excel('result.xlsx', index=False)