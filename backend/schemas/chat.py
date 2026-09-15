# -*- coding: utf-8 -*-
from typing import Literal, Optional

from pydantic import BaseModel, Field

ChatProvider = Literal["agnes", "deepseek"]


class ChatRequest(BaseModel):
    provider: ChatProvider
    message: str = Field(min_length=1, max_length=4000)


class ChatFrame(BaseModel):
    image_url: str
    frame: int
    time_sec: float
    caption: str
    player: Optional[str] = None
    confidence: Optional[float] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: float
    frames: list[ChatFrame] = Field(default_factory=list)


class ChatResponse(BaseModel):
    job_id: str
    provider: ChatProvider
    model: str
    message: ChatMessage


class ChatHistory(BaseModel):
    job_id: str
    provider: ChatProvider
    messages: list[ChatMessage]


class ChatProviderInfo(BaseModel):
    id: ChatProvider
    label: str
    model: str
    configured: bool


class ChatConfig(BaseModel):
    providers: list[ChatProviderInfo]


class ChatMatch(BaseModel):
    job_id: str
    name: str
    created_at: Optional[float] = None
    video_url: Optional[str] = None
    summary: Optional[dict] = None
