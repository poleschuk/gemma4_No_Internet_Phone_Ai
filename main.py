import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from config.config import system_prompt
from config.paths import MODEL_PATH

from memory.agent_memory import AgentMemory

from AI.ai import AI

if __name__ == "__main__":
    memory = AgentMemory(max_items=10, system_prompt=system_prompt)
    agent = AI(MODEL_PATH)
    while(True):
        print(f"User: ")
        user_input = input()
        
        memory.add_message(f"User: {user_input}")
        
        context = memory.get_context()
        
        messages = [{"role": "system", "content": context}]
        messages.append({"role": "user", "content": user_input})
        
        response = agent.generate(messages)
        memory.add_message(f"User: {user_input}")
        memory.add_message(f"Agent: {response}")
        
        print(f"AI: ")
        print(response)
