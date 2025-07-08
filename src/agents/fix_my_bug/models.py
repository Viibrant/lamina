from pydantic import BaseModel


class CodeFix(BaseModel):
    """
    Represents a code fix suggestion.
    """

    original_code: str
    fixed_code: str
    explanation: str
