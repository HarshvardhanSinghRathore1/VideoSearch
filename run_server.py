"""
VideoRAG Server Launcher
Starts the FastAPI backend and serves the React frontend.
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    parser = argparse.ArgumentParser(description="Start VideoRAG Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Starting VideoRAG server at http://{args.host}:{args.port}")
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
