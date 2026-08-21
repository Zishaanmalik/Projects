import os
os.environ['HF_HOME']=r'F:\Langchain\models'

from langchain_huggingface import  ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2-1.5B-Instruct",
    task="text-generation",
    #max_new_tokens=100
)
model =ChatHuggingFace(llm=llm)
history=[]
while True:

    user_input=input('you: ')
    if user_input.lower() == 'exit':
        break
    history.append(user_input)
    result=model.invoke(history)
    history.append(result.content)
    print('AI: ',result.content)



