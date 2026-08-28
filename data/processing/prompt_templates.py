"""Instruction Prompt Templates for Text-to-SQL.

Defines multiple prompt formatting strategies:
- Format 1: Markdown DDL Instruction format (classic instruction tuning)
- Format 2: ChatML / Qwen2.5-Coder Conversational format (standard for modern chat models)
- Format 3: SQLite Code-Comment format (commented schema + query generation)
"""

from typing import Any, Dict, Optional


class PromptTemplate:
    """Base prompt template interface."""

    def format_input(self, schema_text: str, question: str) -> str:
        """Format the input prompt (without target SQL) for inference."""
        raise NotImplementedError

    def format_full(self, schema_text: str, question: str, sql: str) -> str:
        """Format the complete prompt including target SQL for training."""
        raise NotImplementedError


class MarkdownInstructionTemplate(PromptTemplate):
    """Format 1: Markdown Instruction Template.

    Uses explicit markdown headers to demarcate Database Schema, Question, and SQL.
    """

    def __init__(self, system_instruction: Optional[str] = None):
        self.system_instruction = system_instruction or (
            "You are an expert SQL engineer. Given the database schema, write the exact "
            "SQLite query that answers the user question."
        )

    def format_input(self, schema_text: str, question: str) -> str:
        return (
            f"{self.system_instruction}\n\n"
            f"### Database Schema:\n{schema_text}\n\n"
            f"### Question:\n{question}\n\n"
            f"### SQL:\n"
        )

    def format_full(self, schema_text: str, question: str, sql: str) -> str:
        return self.format_input(schema_text, question) + sql.strip()


class ChatMLTemplate(PromptTemplate):
    """Format 2: ChatML / Qwen2.5 Conversational Template.

    Employs standard ChatML special tokens (<|im_start|>, <|im_end|>)
    matching the Qwen2.5-Coder chat format.
    """

    def __init__(self, system_instruction: Optional[str] = None):
        self.system_instruction = system_instruction or (
            "You are an expert SQL engineer. Given the following SQLite database schema, "
            "write the exact SQL query that answers the question. Return ONLY the raw SQL query."
        )

    def format_input(self, schema_text: str, question: str) -> str:
        return (
            f"<|im_start|>system\n{self.system_instruction}\n\n"
            f"Database Schema:\n{schema_text}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def format_full(self, schema_text: str, question: str, sql: str) -> str:
        return self.format_input(schema_text, question) + f"{sql.strip()}<|im_end|>"

    def to_messages(self, schema_text: str, question: str, sql: Optional[str] = None) -> list:
        """Return conversational message list suitable for HuggingFace apply_chat_template."""
        messages = [
            {
                "role": "system",
                "content": f"{self.system_instruction}\n\nDatabase Schema:\n{schema_text}",
            },
            {"role": "user", "content": question},
        ]
        if sql is not None:
            messages.append({"role": "assistant", "content": sql.strip()})
        return messages


class CodeCommentTemplate(PromptTemplate):
    """Format 3: SQLite Code-Comment Template.

    Formats the schema as SQL comments and the question as a comment,
    prompting the model to complete the SQL query naturally.
    """

    def format_input(self, schema_text: str, question: str) -> str:
        return (
            f"/* SQLite Database Schema */\n"
            f"{schema_text}\n\n"
            f"-- Question: {question}\n"
            f"-- Target SQL:\n"
        )

    def format_full(self, schema_text: str, question: str, sql: str) -> str:
        return self.format_input(schema_text, question) + sql.strip()


TEMPLATES: Dict[str, PromptTemplate] = {
    "markdown": MarkdownInstructionTemplate(),
    "chatml": ChatMLTemplate(),
    "code_comment": CodeCommentTemplate(),
}
