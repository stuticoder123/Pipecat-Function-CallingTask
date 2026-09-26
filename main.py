import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

# FUNCTION CALLING TOOLS
async def get_current_weather(
    params: FunctionCallParams,
    location: str,
    format: str,
):
    """Get the current weather.

    Args:
        location: The city and state.
        format: Temperature unit: celsius or fahrenheit.
    """
    logger.info(f"Weather requested for {location} in {format}")

    # Demo response
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
    """Get a restaurant recommendation.

    Args:
        location: The city and state.
    """
    logger.info(f"Restaurant recommendation requested for {location}")

    # Demo response
    await params.result_callback(
        {
            "location": location,
            "name": "The Golden Dragon",
        }
    )

# TRANSPORT CONFIGURATION
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

# BOT
async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
):
    logger.info("Starting Stuti's voice agent")

    # SARVAM STT
    stt = SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
        ),
    )

    # SARVAM TTS
    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="shubh",
        ),
    )

    # GROQ LLM
    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            system_instruction=(
                "You are a helpful voice assistant. "
                "Your responses will be spoken aloud. "
                "Keep responses short, natural and conversational. "
                "Do not use emojis, markdown, bullet points, "
                "or formatting that sounds unnatural when spoken."
            ),
        ),
    )

    # FUNCTION CALL EVENT
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(
        service,
        function_calls,
    ):
        logger.info(f"Function calls started: {function_calls}")

        await tts.queue_frame(
            TTSSpeakFrame("Let me check on that.")
        )

    # LLM Context
    context = LLMContext(
        tools=[
            get_current_weather,
            get_restaurant_recommendation,
        ]
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer()
        ),
    )

    # PIPELINE
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # WORKER
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    # RUNNER
    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint
    )

    await runner.add_workers(worker)

    # CLIENT CONNECTED
    @transport.event_handler("on_client_connected")
    async def on_client_connected(
        transport,
        client,
    ):
        logger.info("Client connected")

        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Introduce yourself to the user. "
                    "Say that you are Stuti's voice assistant "
                    "and ask how you can help."
                ),
            }
        )

        await worker.queue_frames([LLMRunFrame()])

    # CLIENT DISCONNECTED
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(
        transport,
        client,
    ):
        logger.info("Client disconnected")
        await runner.cancel()

    # START RUNNER
    await runner.run()

# PIPECAT ENTRY POINT
async def bot(
    runner_args: RunnerArguments,
):
    """Main bot entry point."""
    transport = await create_transport(
        runner_args,
        transport_params,
    )

    await run_bot(
        transport,
        runner_args,
    )

# MAIN
if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
