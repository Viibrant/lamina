from loguru import logger
from pydantic import BaseModel


def tool_schema_from_model(
    model: type[BaseModel],
    *,
    name: str | None = None,
    description: str | None = None,
    is_array: bool = False,
) -> dict:
    """
    Generate a JSON schema for a tool based on a Pydantic model.

    Args:
        model (type[BaseModel]): The Pydantic model to generate the schema from.
        name (str | None): Optional name for the tool.
        description (str | None): Optional description.
        is_array (bool): If True, returns a schema for an array of this model.

    Returns:
        dict: A JSON schema dictionary compatible with OpenAI function-calling.
    """
    if not issubclass(model, BaseModel):
        raise ValueError("The model must be a subclass of BaseModel")
    if not model.__doc__:
        raise ValueError("The model must have a docstring for the description")
    if not name:
        logger.warning(
            "No name provided for tool schema, using model's class name instead."
        )
        name = model.__name__

    json_schema = model.model_json_schema()

    if is_array:
        parameters = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            },
        }
    else:
        parameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
            "required": json_schema.get("required", []),
        }

    return {
        "name": name,
        "description": description or model.__doc__,
        "parameters": parameters,
    }
