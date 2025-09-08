"""Custom STT component for Sarvam.ai integration into pipecat-ai Pipeline."""
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from dataclasses import dataclass
from sarvamai.core.client_wrapper import AsyncClientWrapper
from typing import Optional, Dict, AsyncGenerator, Union
from sarvamai.environment import SarvamAIEnvironment
from httpx import AsyncClient
from typing import Literal
from pipecat.frames.frames import StartFrame, EndFrame, CancelFrame, Frame, TranscriptionFrame, ErrorFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.time import time_now_iso8601
from sarvamai.speech_to_text_streaming.raw_client import AsyncRawSpeechToTextStreamingClient
from sarvamai.speech_to_text_streaming.types.speech_to_text_streaming_model import SpeechToTextStreamingModel
from sarvamai.speech_to_text_translate_streaming.raw_client import AsyncRawSpeechToTextTranslateStreamingClient
from sarvamai.speech_to_text_translate_streaming.types.speech_to_text_translate_streaming_model import SpeechToTextTranslateStreamingModel
from sarvamai.core.request_options import RequestOptions
from pydantic import Field
from sarvamai.core.events import EventType
from asyncio import create_task
from base64 import b64encode
from sys import stderr
from loguru import logger

# logger.add(sink=stderr,level="DEBUG")

from audioop import ulaw2lin, ratecv
def convert_mulaw_to_pcm_16khz(mulaw_audio_chunk: bytes) -> bytes:
    pcm_8khz = ulaw2lin(mulaw_audio_chunk, 2)
    pcm_16khz, _ = ratecv(pcm_8khz, 2, 1, 8000, 16000, None)
    return pcm_16khz

@dataclass
class SarvamLiveOptions:
    model: Union[SpeechToTextStreamingModel, SpeechToTextTranslateStreamingModel] = "saaras:v2.5"
    sample_rate: int = 16000
    language: Language = Language.HI_IN
    vad_signal: Literal["true", "false"] = "false"
    high_vad_sensitivity: Literal["true", "false"] = "false"
    flush_signal: Literal["true", "false"] = "true"
    encoding: Literal["mpeg", "mp3", "mpeg3", "x-mpeg-3", "x-mp3", "wav",
                      "x-wav", "wave", "aac", "x-aac", "aiff", "x-aiff",
                      "ogg", "opus", "flac", "x-flac", "mp4", "x-m4a",
                      "amr", "x-ms-wma", "webm", "pcm_s16le", "pcm_l16", "pcm_raw"] = "pcm_l16"
    # request_options: Optional[Dict] = None
    timeout_in_second: int = 30 # The number of seconds to await an API call before timing out.
    max_retries: int = 2 # The max number of retries to attempt if the API call fails.
    additional_header: Field = None # A dictionary containing additional parameters to spread into the request's header dict
    additional_query_parameters: Field = None # A dictionary containing additional parameters to spread into the request's query parameters dict
    additional_body_parameters: Field = None # A dictionary containing additional parameters to spread into the request's body parameters dict
    chunk_size: int = 32 # The size, in bytes, to process each chunk of data being streamed back within the response.
    # This equates to leveraging chunk_size within requests or httpx, and is only leveraged for file downloads.

class SarvamSTTService(STTService):
    def __init__(
            self, api_key: str,
            headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = 30.0,
            environment: SarvamAIEnvironment = SarvamAIEnvironment.PRODUCTION,
            httpx_client: AsyncClient = AsyncClient(base_url="https://api.sarvam.ai"),
            options: SarvamLiveOptions = SarvamLiveOptions(), **kwargs
                ):
        super().__init__(**kwargs)
        self._client = AsyncClientWrapper(
            api_subscription_key=api_key,
            timeout= timeout,
            headers=headers,
            environment=environment,
            httpx_client=httpx_client)
        logger.info("Sarvam STT Service initialized")
        self._api_subscription_key = api_key
        self._settings = options.__dict__
    
    def vad_enabled(self) -> bool:
        return self._settings.get("vad_signal", False)
    
    def can_generate_metrics(self) -> bool:
        return False # Sarvam does not provide processing metrics
    
    async def set_model(self, model):
        super().set_model(model)
        logger.info(f"Switching to model: {model}")
        await self._disconnect()
        self._settings["model"] = model
        await self._connect()
    
    async def set_language(self, language):
        await super().set_language(language)
        logger.info(f"Switching to language: {language}")
        await self._disconnect()
        self._settings["language"] = language
        await self._connect()
    
    async def set_prompt(self, prompt: str):
        logger.info(f"Setting prompt: {prompt}")
        if self._connection:
            await self._connection.set_prompt(prompt)
    
    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()
    
    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()
    
    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not hasattr(self, "_connection"):
            await self._connect()
        params = {"audio": b64encode(convert_mulaw_to_pcm_16khz(audio)),
                  "encoding": f"audio/{self._settings.get('encoding')}",
                  "sample_rate": self._settings.get("sample_rate")}
        logger.debug(f"Sending audio chunk of size {len(audio)} bytes to Sarvam STT")
        if "saarika" in self._settings.get("model").lower(): await self._connection.transcribe(**params)
        else: await self._connection.translate(**params)
        if self._settings.get("flush_signal") == "true": await self._connection.flush()
        yield None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self._connect()
            self.push_frame(frame=frame, direction=direction)
        if isinstance(frame, EndFrame) or isinstance(frame, CancelFrame):
            await self._disconnect()
            self.push_frame(frame=frame, direction=direction)
    
    async def _connect(self):
        request_options = RequestOptions(**self._requestOptionsDict())
        if "saarika" in self._settings.get("model").lower():
            self._contextManager = AsyncRawSpeechToTextStreamingClient(client_wrapper=self._client).connect(
                language_code=self._settings.get("language"),
                model=self._settings.get("model"),
                high_vad_sensitivity=self._settings.get("high_vad_sensitivity"),
                vad_signals=self._settings.get("vad_signal"),
                flush_signal=self._settings.get("flush_signal"),
                api_subscription_key=self._api_subscription_key,
                request_options=request_options
            )
        else:
            self._contextManager = AsyncRawSpeechToTextTranslateStreamingClient(client_wrapper=self._client).connect(
                model=self._settings.get("model"),
                high_vad_sensitivity=self._settings.get("high_vad_sensitivity"),
                vad_signals=self._settings.get("vad_signal"),
                flush_signal=self._settings.get("flush_signal"),
                api_subscription_key=self._api_subscription_key,
                request_options=request_options
            )
        self._connection = await self._contextManager.__aenter__()
        self._listen_task = create_task(self._connection.start_listening())
        self._connection.on(EventType.MESSAGE, self._on_message)
        self._connection.on(EventType.ERROR, self._on_error)
    
    async def _disconnect(self):
        if hasattr(self, "_connection"):
            logger.info("Disconnecting Sarvam STT WebSocket connection")
            if hasattr(self._connection, "cose") and callable(getattr(self._connection, "close")):
                await self._connection.close()
            else:
                if hasattr(self, "_listen_task"):
                    self._listen_task.cancel()
                    try: await self._listen_task
                    except: pass
            if hasattr(self, "_contextManager"): await self._contextManager.__aexit__(None, None, None)
    
    async def _on_message(self, *args, **kwargs):
        data = await self._connection.recv()
        data = data.get("data", None)
        if not data:
            await self._on_error(error="No data received in message event")
        if self._settings.get("vad_signal") == "true" and hasattr(data, "signal_type"):
            if data.signal_type == "START_SPEECH": await self._call_event_handler("on_speech_started", *args, **kwargs)
            if data.signal_type == "END_SPEECH": await self._call_event_handler("on_utterance_end", *args, **kwargs)
        transcript = data.transcript if hasattr(data, "transcript") and data.transcript else ""
        language = data.language_code if hasattr(data, "language_code") and data.language_code else self._settings.get("language")
        language = Language(language) if isinstance(language, str) else language
        if len(transcript) > 0:
            await self.push_frame(TranscriptionFrame(
                text=transcript,
                user_id=self._user_id,
                timestamp=data.timestamp if hasattr(data, "timestamp") else time_now_iso8601(),
                language=language,
                result=data.dict()
                ))
    
    async def _on_error(self, *args, **kwargs):
        error = await self._connection.recv()
        error = error.get("data", None)
        await self.push_error(error=ErrorFrame(
            error=error.error if hasattr(error, "error") else "Unknown error",
            fatal=True,
            processor=self.name
            ))
        await self._connect()
    
    def _requestOptionsDict(self) -> Dict:
        request_options = dict(
            timeout_in_second=self._settings.get("timeout_in_second"),
            max_retries=self._settings.get("max_retries"),
            chunk_size=self._settings.get("chunk_size")
            )
        
        if self._settings.get("additional_header") != None:
            request_options["additional_header"] = self._settings.get("additional_header")
        if self._settings.get("additional_query_parameters") != None:
            request_options["additional_query_parameters"] = self._settings.get("additional_query_parameters")
        if self._settings.get("additional_body_parameters") != None:
            request_options["additional_body_parameters"] = self._settings.get("additional_body_parameters")
        
        return request_options