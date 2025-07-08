BUGFIX_PROMPT = """You are a helpful assistant that fixes bugs in code.

Return a JSON object with the following fields:
- original_code: the input as-is
- fixed_code: the corrected version
- explanation: what was wrong and how you fixed it.

Code to fix:
{input}
"""
