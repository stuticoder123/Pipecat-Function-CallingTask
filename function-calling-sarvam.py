from loguru import logger
from pipecat.services.llm_service import FunctionCallParams


async def get_current_weather(
    params: FunctionCallParams,
    location: str,
    format: str,
):
    logger.info(f"Weather requested for {location} in {format}")
    await params.result_callback(
        {
            "location": location,
            "conditions": "nice",
            "temperature": "75",
            "unit": format,
        }
    )


async def get_restaurant_recommendation(
    params: FunctionCallParams,
    location: str,
):
    logger.info(f"Restaurant recommendation requested for {location}")
    await params.result_callback(
        {
            "location": location,
            "name": "The Golden Dragon",
        }
    )
