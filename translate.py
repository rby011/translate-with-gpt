
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()
    return prompt_template

input = '''Feeling hungry, we grabbed a pizza from the fridge and microwaved it.
The video chronologically shows the son's finger stuck in a hole of the dog house, the rescue process by paramedics, and the conclusion of the entire incident.
'''

if __name__ == "__main__":

    load_dotenv()
        
    llm = ChatOpenAI(temperature=0.1, 
                     max_tokens=500, 
                     model="gpt-4",
                     verbose = True)
        
    # prompt_template = ChatPromptTemplate.from_template(
    #     template=read_prompt_template('prompts/1. translate.txt')
    # )
    

    prompt_template = ChatPromptTemplate.from_messages( [
        ("system", read_prompt_template('prompts/1. translate.txt')),
        ("human", input),
        ] )

    print(prompt_template)

    chain = LLMChain(
        llm = llm, 
        prompt = prompt_template, 
        output_key = "output",
        verbose = True
    )
    
    req = {}
    req['src_lang'] = 'english'
    req['dst_lang'] = 'korean'
    result = chain(req)

    print(result['output'])