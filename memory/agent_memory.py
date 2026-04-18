from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any

from AI.ai import AI
from config.paths import MODEL_PATH

@dataclass
class MemoryItem:
    text: str
    importance: float = 1.0

@dataclass
class AgentMemory:
    max_items: int = 20
    
    buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    
    state: Dict[str, Any] = field(default_factory=lambda: {
        "goal": None,
        "facts": [],
        "decisions": [],
        "open_tasks": []
    })
    
    summaries: List[str] = field(default_factory=list)
    
    system_prompt: str
    
    def add_message(self, text: str, importance: float = 1.0):
        self.buffer.append(MemoryItem(text, importance))
        self._maybe_compress()
    
    def _maybe_compress(self):
        if len(self.buffer) >= self.max_items:
            self._compress_old_memory()
        
    def _compress_old_memory(self):
        cutoff = len(self.buffer) // 2
        old_items = [self.buffer.popleft() for _ in range(cutoff)]
        
        summary = self._summerize(old_items)
        self.summaries.append(summary)
    
    def _summerize(self, items: List[MemoryItem]) -> str:
        system_prompt_summerize = """
        You should carefuly summerize messages that you given. Compress it to the one sentence.
        You should also take a look to what role says what and giving to that attention.
        """
        request = [{"role": "system", "content": system_prompt_summerize}]
        request = request.append(items)
    
        summarize_model = AI(MODEL_PATH)
        
        return summarize_model.generate(request)
    
    def add_fact(self, fact: str):
        self.state["facts"].append(fact)
    
    def add_decision(self, decision: str):
        self.state["decisions"].append(decision)
    
    def get_context(self) -> str:
        recent = "\n".join(i.text for i in self.buffer)
        summary = "\n".join(self.summaries[-3:])
        
        return f"""
        SYSTEM(MOST IMPORTANT!!!): 
        {self.system_prompt}
    
        GOAL: 
        {self.state['goal']}

        FACTS:
        {self.state['facts']}

        DECISIONS:
        {self.state['decisions']}

        RECENT:
        {recent}

        SUMMARY:
        {summary}
        """
        