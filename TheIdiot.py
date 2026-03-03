#!/usr/bin/env python3
"""
Project Gutenberg / ebook -> audio (chunked) TTS CLI.

Supports:
- .txt input
- .epub input (via ebooklib + BeautifulSoup)
TTS engines:
- edge-tts (online, high quality)
- pyttsx3 (offline fallback, lower quality)

Output:
- Writes chunk audio files to an output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# -----------------------------
# Ingest
# -----------------------------

def read_text_file(path: Path) -> str:
    # Try utf-8 first; fallback to latin-1 for older Gutenberg texts.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def extract_text_from_epub(path: Path) -> str:
    try:
        from ebooklib import epub  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        raise RuntimeError("EPUB support requires: ebooklib, beautifulsoup4") from e

    book = epub.read_epub(str(path))
    parts: List[str] = []
    for item in book.get_items():
        # Only HTML documents; skip images/styles.
        if item.get_type() == 9:  # ebooklib.ITEM_DOCUMENT, but avoid import constant
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text("\n")
            parts.append(text)
    return "\n\n".join(parts)

def load_ebook_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_text_file(path)
    if suffix == ".epub":
        return extract_text_from_epub(path)
    raise ValueError(f"Unsupported input type: {suffix} (use .txt or .epub)")

# -----------------------------
# Clean / Normalize
# -----------------------------

GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
GUTENBERG_END_RE   = re.compile(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Removes Gutenberg header/footer if present. If markers not found, returns original.
    """
    start_match = GUTENBERG_START_RE.search(text)
    end_match = GUTENBERG_END_RE.search(text)

    if start_match and end_match and end_match.start() > start_match.end():
        return text[start_match.end():end_match.start()].strip()

    # Fallback: some files have slightly different markers.
    # Try a simpler heuristic with "START OF" and "END OF" lines.
    start2 = re.search(r"^\*\*\*\s*START OF.*?\*\*\*\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    end2 = re.search(r"^\*\*\*\s*END OF.*?\*\*\*\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if start2 and end2 and end2.start() > start2.end():
        return text[start2.end():end2.start()].strip()

    return text

def normalize_text(text: str) -> str:
    # Normalize whitespace, fix common Gutenberg artifacts without being too aggressive.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -----------------------------
# Segment / Chunk
# -----------------------------

CHAPTER_HEADING_RE = re.compile(
    r"^(?:\s*)(CHAPTER\s+\w+|BOOK\s+\w+|PART\s+\w+|PROLOGUE|EPILOGUE)\b.*$",
    re.IGNORECASE | re.MULTILINE
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])")

@dataclass(frozen=True)
class Segment:
    title: str
    text: str

def split_into_segments(text: str) -> List[Segment]:
    """
    Attempts chapter-like segmentation using headings. If none found, returns one segment.
    """
    matches = list(CHAPTER_HEADING_RE.finditer(text))
    if not matches:
        return [Segment(title="full_text", text=text)]

    segments: List[Segment] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title_line = m.group(0).strip()
        seg_text = text[start:end].strip()
        segments.append(Segment(title=slugify(title_line) or f"chapter_{i+1:03d}", text=seg_text))
    return segments

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def chunk_text(text: str, max_chars: int = 2500) -> List[str]:
    """
    Splits into chunks <= max_chars, trying to respect sentence boundaries.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # First split into paragraphs; then split paragraphs into sentences if needed.
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf, buf_len = [], 0

    for p in paras:
        if len(p) <= max_chars:
            if buf_len + len(p) + 2 <= max_chars:
                buf.append(p)
                buf_len += len(p) + 2
            else:
                flush()
                buf.append(p)
                buf_len = len(p)
            continue

        # Paragraph too long: sentence-split it.
        sentences = SENTENCE_SPLIT_RE.split(p)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                # Hard wrap extremely long "sentences" (rare; often OCR or weird formatting).
                for j in range(0, len(sent), max_chars):
                    piece = sent[j:j+max_chars].strip()
                    if piece:
                        if buf_len + len(piece) + 1 <= max_chars:
                            buf.append(piece)
                            buf_len += len(piece) + 1
                        else:
                            flush()
                            buf.append(piece)
                            buf_len = len(piece)
                continue

            if buf_len + len(sent) + 1 <= max_chars:
                buf.append(sent)
                buf_len += len(sent) + 1
            else:
                flush()
                buf.append(sent)
                buf_len = len(sent)

    flush()
    return chunks

# -----------------------------
# TTS Engines
# -----------------------------

class TtsEngine:
    async def synthesize_to_file(self, text: str, out_path: Path) -> None:
        raise NotImplementedError

class EdgeTtsEngine(TtsEngine):
    def __init__(self, voice: str, rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    async def synthesize_to_file(self, text: str, out_path: Path) -> None:
        import edge_tts  # type: ignore
        communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
        await communicate.save(str(out_path))

class Pyttsx3Engine(TtsEngine):
    """
    Offline fallback; writes WAV if supported. (Windows SAPI5 usually works; others vary.)
    """
    def __init__(self, voice_contains: Optional[str] = None, rate_wpm: Optional[int] = None):
        self.voice_contains = voice_contains
        self.rate_wpm = rate_wpm

    async def synthesize_to_file(self, text: str, out_path: Path) -> None:
        # pyttsx3 is synchronous; run in a thread to avoid blocking.
        import pyttsx3  # type: ignore
        loop = asyncio.get_running_loop()

        def _run():
            engine = pyttsx3.init()
            if self.rate_wpm is not None:
                engine.setProperty("rate", int(self.rate_wpm))

            if self.voice_contains:
                target = self.voice_contains.lower()
                voices = engine.getProperty("voices")
                for v in voices:
                    name = getattr(v, "name", "") or ""
                    vid = getattr(v, "id", "") or ""
                    if target in name.lower() or target in vid.lower():
                        engine.setProperty("voice", v.id)
                        break

            engine.save_to_file(text, str(out_path))
            engine.runAndWait()

        await loop.run_in_executor(None, _run)

# -----------------------------
# Driver
# -----------------------------

@dataclass(frozen=True)
class Job:
    segment_idx: int
    chunk_idx: int
    title: str
    text: str

def build_jobs(segments: List[Segment], max_chars: int) -> List[Job]:
    jobs: List[Job] = []
    for si, seg in enumerate(segments, start=1):
        chunks = chunk_text(seg.text, max_chars=max_chars)
        for ci, c in enumerate(chunks, start=1):
            jobs.append(Job(segment_idx=si, chunk_idx=ci, title=seg.title, text=c))
    return jobs

async def run_tts(
    engine: TtsEngine,
    jobs: List[Job],
    out_dir: Path,
    ext: str,
    concurrency: int,
    resume: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def _one(job: Job) -> None:
        fname = f"{job.segment_idx:03d}_{job.title}_chunk_{job.chunk_idx:03d}.{ext}"
        out_path = out_dir / fname

        if resume and out_path.exists() and out_path.stat().st_size > 0:
            return

        async with sem:
            await engine.synthesize_to_file(job.text, out_path)

    # Keep ordering stable but allow parallel synthesis.
    tasks = [asyncio.create_task(_one(j)) for j in jobs]
    # Fail-fast: if one chunk errors, cancel the rest.
    try:
        for t in tasks:
            await t
    except Exception:
        for t in tasks:
            t.cancel()
        raise

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gutenberg/ebook -> chunked audiobook via TTS")
    p.add_argument("input", type=Path, help="Path to .txt or .epub")
    p.add_argument("--out", type=Path, default=Path("out_audio"), help="Output directory")
    p.add_argument("--engine", choices=["edge", "pyttsx3"], default="edge")
    p.add_argument("--strip-gutenberg", action="store_true", help="Strip Gutenberg header/footer markers if present")
    p.add_argument("--max-chars", type=int, default=2500, help="Max characters per TTS request/chunk")
    p.add_argument("--concurrency", type=int, default=2, help="Parallel chunk synthesis (edge-tts only really benefits)")
    p.add_argument("--resume", action="store_true", help="Skip chunks already rendered")

    # edge-tts options
    p.add_argument("--voice", type=str, default="en-US-AriaNeural", help="edge-tts voice name")
    p.add_argument("--rate", type=str, default="+0%", help="edge-tts rate, e.g. +10% or -10%")

    # pyttsx3 options
    p.add_argument("--voice-contains", type=str, default=None, help="pyttsx3 voice name/id substring")
    p.add_argument("--rate-wpm", type=int, default=None, help="pyttsx3 speaking rate")

    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    text = load_ebook_text(args.input)
    if args.strip_gutenberg:
        text = strip_gutenberg_boilerplate(text)
    text = normalize_text(text)

    segments = split_into_segments(text)
    jobs = build_jobs(segments, max_chars=args.max_chars)

    if args.engine == "edge":
        engine: TtsEngine = EdgeTtsEngine(voice=args.voice, rate=args.rate)
        ext = "mp3"
    else:
        engine = Pyttsx3Engine(voice_contains=args.voice_contains, rate_wpm=args.rate_wpm)
        ext = "wav"

    print(f"Input: {args.input}")
    print(f"Segments: {len(segments)} | Chunks: {len(jobs)}")
    print(f"Engine: {args.engine} | Out: {args.out} | Ext: .{ext}")

    async def _go():
        await run_tts(
            engine=engine,
            jobs=jobs,
            out_dir=args.out,
            ext=ext,
            concurrency=max(1, int(args.concurrency)),
            resume=bool(args.resume),
        )

    try:
        asyncio.run(_go())
        return 0
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
