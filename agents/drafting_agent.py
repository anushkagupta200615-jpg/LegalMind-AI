from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DraftingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.parser = StrOutputParser()
        
    def draft_notice(self, recipient_name: str, sender_name: str, issue_description: str, relevant_sections: str) -> str:
        template = """You are an expert Indian lawyer. Draft a formal legal notice based on the following details:
        
Recipient Name: {recipient_name}
Sender Name: {sender_name}
Issue Description: {issue_description}
Relevant Sections: {relevant_sections}

Respond in formal Indian legal English."""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | self.parser
        
        return chain.invoke({
            "recipient_name": recipient_name,
            "sender_name": sender_name,
            "issue_description": issue_description,
            "relevant_sections": relevant_sections
        })
        
    def summarize_contract(self, contract_text: str) -> str:
        template = """You are an expert Indian corporate lawyer. Extract a structured summary from the following contract text.
Include the following sections:
- Parties involved
- Key obligations
- Penalty clauses
- Termination conditions

Contract Text:
{contract_text}

Respond in formal Indian legal English."""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | self.parser
        
        return chain.invoke({"contract_text": contract_text})
        
    def draft_reply(self, original_notice_text: str) -> str:
        template = """You are an expert Indian lawyer. Draft a professional reply to the following legal notice.
        
Original Notice Text:
{original_notice_text}

Respond in formal Indian legal English."""
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | self.parser
        
        return chain.invoke({"original_notice_text": original_notice_text})
