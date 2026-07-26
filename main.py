
from __future__ import annotations
import logging, sys
import uvicorn
from src.utils.config import get_config, load_config

def _setup_logging():
    cfg = load_config()
    log_cfg = cfg.logging
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))
    fmt = logging.Formatter(log_cfg.format)
    if log_cfg.console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    if log_cfg.file:
        from pathlib import Path
        log_path = Path(log_cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

def main():
    _setup_logging()
    cfg = get_config()
    logging.getLogger(__name__).info(
        "Starting RAG API server on "+str(cfg.server.host)+":"+str(cfg.server.port)
    )
    uvicorn.run(
        "app.api.routes:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=cfg.server.reload,
        workers=cfg.server.workers if not cfg.server.reload else 1,
        log_level=cfg.logging.level.lower(),
    )

if __name__ == "__main__":
    main()
