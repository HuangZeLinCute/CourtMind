@echo off
rem Good-Badminton FastAPI backend launcher
rem Uses the badminton conda env by absolute path - no manual activation needed.
cd /d "%~dp0"
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost
"C:\Users\14181\.conda\envs\badminton\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
