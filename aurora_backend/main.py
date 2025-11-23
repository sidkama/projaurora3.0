from __future__ import annotations

import os

from dotenv import load_dotenv

def main() -> None:
    # Load environment variables for spoon-core providers, etc.
    load_dotenv(override=True)

    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=os.getenv("AURORA_HOST", "0.0.0.0"),
        port=int(os.getenv("AURORA_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
