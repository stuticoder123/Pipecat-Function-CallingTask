import os
from dotenv import load_dotenv
from loguru import logger

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
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.base_transport import (
    BaseTransport,
    TransportParams,
)
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

from function_calling_sarvam import (
    get_current_weather,
    get_restaurant_recommendation,
)

load_dotenv(override=True)

transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
):
    logger.info("Starting Stuti's voice agent")

    stt = SarvamSTTService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
        ),
    )

    tts = SarvamTTSService(
        api_key=os.environ["SARVAM_API_KEY"],
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="shubh",
        ),
    )

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model="llama-3.1-70b-versatile",
        ),
    )

    llm.register_function(
        "get_current_weather",
        get_current_weather,
    )

    llm.register_function(
        "get_restaurant_recommendation",
        get_restaurant_recommendation,
    )

    logger.info("Function-calling tools registered")

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        logger.info(f"Function calls started: {function_calls}")
        await service.push_frame(
            TTSSpeakFrame("Let me check on that.")
        )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Stuti's helpful voice assistant. "
                    "You are having a natural voice conversation "
                    "with the user. "
                    "Your responses will be spoken aloud. "
                    "Keep responses short, natural, helpful, "
                    "and conversational. "
                    "Do not use emojis, markdown, bullet points, "
                    "or formatting that sounds unnatural when spoken. "
                    "Use the available tools whenever the user asks "
                    "for current weather or a restaurant recommendation."
                ),
            }
        ],
        tools=[
            get_current_weather,
            get_restaurant_recommendation,
        ],
    )

    user_aggregator, assistant_aggregator = (
        LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer()
            ),
        )
    )

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

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(
        handle_sigint=runner_args.handle_sigint
    )

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()

async def bot(runner_args: RunnerArguments):
    transport = await create_transport(
        runner_args,
        transport_params,
    )
    await run_bot(
        transport,
        runner_args,
    )

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
