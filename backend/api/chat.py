# -*- coding: utf-8 -*-
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.schemas.chat import (
    ChatConfig, ChatHistory, ChatMatch, ChatProvider, ChatProviderInfo,
    ChatRequest, ChatResponse,
)
from backend.services import job_service, llm_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{job_id}/stream")
def stream_message(job_id: str, payload: ChatRequest):
    try:
        events = llm_service.stream_chat(job_id, payload.provider, payload.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def encode_events():
        try:
            for event in events:
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            events.close()

    return StreamingResponse(encode_events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


@router.get("/config", response_model=ChatConfig)
def get_config():
    return ChatConfig(providers=[ChatProviderInfo(**item) for item in llm_service.provider_config()])


@router.get("/matches", response_model=list[ChatMatch])
def get_matches():
    matches = []
    for item in job_service.history():
        if item.get("status") != "completed":
            continue
        result = job_service.job_result(item["job_id"])
        if not result or not result.get("metadata"):
            continue
        matches.append(ChatMatch(
            job_id=item["job_id"],
            name=item["name"],
            created_at=item.get("created_at"),
            video_url=result.get("video_url"),
            summary=(result.get("report") or {}).get("summary"),
        ))
    return matches


@router.get("/{job_id}/history", response_model=ChatHistory)
def get_history(job_id: str, provider: ChatProvider = Query(...)):
    try:
        messages = llm_service.get_history(job_id, provider)
        return ChatHistory(job_id=job_id, provider=provider, messages=messages)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/{job_id}/history", status_code=204)
def delete_history(job_id: str, provider: ChatProvider = Query(...)):
    try:
        llm_service.clear_history(job_id, provider)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{job_id}/messages", response_model=ChatResponse)
def send_message(job_id: str, payload: ChatRequest):
    try:
        return ChatResponse(**llm_service.chat(job_id, payload.provider, payload.message))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
