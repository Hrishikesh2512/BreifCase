from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="briefcase", description="Run the Briefcase server.")
    parser.add_argument("--host", default=os.getenv("BRIEFCASE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("BRIEFCASE_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Auto-reload for development.")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("briefcase.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
