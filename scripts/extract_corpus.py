#!/usr/bin/env python3
"""
Palimpsest - 认知资产与代码语料分层提纯引擎 (Corpus Extraction & Layering Engine)

功能:
  从冻结快照 data/cache.db 中提取会话，将 76% 的代码库附件与系统指令
  与核心用户意图进行物理剥离，物化生成供各检索算法打擂的纯净语料库 data/purified_corpus.db。

用法:
  python scripts/extract_corpus.py
  python scripts/extract_corpus.py --limit 500  # 抽样演练
  python scripts/extract_corpus.py --vacuum     # 提取后执行 VACUUM 压缩
"""

import argparse
import base64
import binascii
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 路径提取正则与符号模式
PATH_PATTERN = re.compile(
    r"[a-zA-Z0-9_\u4e00-\u9fa5\.\-]+/[a-zA-Z0-9_\u4e00-\u9fa5\.\-\/]+\.[a-zA-Z0-9]+"
)
CODE_FENCE_PATTERN = re.compile(r"```(?:\w+)?\n([\s\S]+?)\n```")
IDENTIFIER_PATTERN = re.compile(r"\b(?:class|def|interface|function|const|export)\s+([A-Za-z_][A-Za-z0-9_]+)")


def is_text_mime(mime: str) -> bool:
    """判断 MIME 是否为可阅读的代码或文本"""
    m = mime.lower()
    return (
        m.startswith("text/")
        or "json" in m
        or "xml" in m
        or "javascript" in m
        or "typescript" in m
        or "yaml" in m
        or "csv" in m
        or "sql" in m
        or "markdown" in m
        or "x-sh" in m
        or "x-python" in m
    )


def extract_symbols_from_text(text: str) -> set[tuple[str, str]]:
    """从文本中提取文件路径与声明标识符: 返回 set of (symbol_type, symbol_text)"""
    symbols = set()
    # 1. 提取文件路径
    for p in PATH_PATTERN.findall(text):
        p_clean = p.strip()
        if not p_clean.startswith("http") and "/" in p_clean:
            symbols.add(("filepath", p_clean))

    # 2. 提取核心定义标识符 (class / def / interface)
    for ident in IDENTIFIER_PATTERN.findall(text):
        if len(ident) >= 3:
            symbols.add(("identifier", ident))

    return symbols


def init_target_database(db_path: str):
    """初始化提纯语料库表结构"""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    # 1. 会话主表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            file_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            created_time TEXT,
            modified_time TEXT,
            user_turn_count INTEGER NOT NULL DEFAULT 0,
            model_turn_count INTEGER NOT NULL DEFAULT 0,
            raw_total_tokens INTEGER NOT NULL DEFAULT 0
        );
    """)

    # 2. 意图与核心交互表 (去噪后的核心评测资产)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS intent_turns (
            turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,                -- 'user' | 'model'
            payload_type TEXT NOT NULL,        -- 'text' | 'inlineFile_summary' | 'thinking'
            text_content TEXT NOT NULL,
            token_count INTEGER NOT NULL DEFAULT 0,
            is_thought INTEGER NOT NULL DEFAULT 0,
            has_code INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT,
            FOREIGN KEY(file_id) REFERENCES sessions(file_id)
        );
    """)

    # 3. 背景代码库与符号索引表 (由附件与代码块抽取的符号倒排池)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS context_symbols (
            symbol_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            symbol_type TEXT NOT NULL,         -- 'filepath' | 'identifier' | 'heading'
            symbol_text TEXT NOT NULL,
            FOREIGN KEY(file_id) REFERENCES sessions(file_id)
        );
    """)

    # 索引优化
    cur.execute("CREATE INDEX IF NOT EXISTS idx_intent_fid ON intent_turns(file_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_fid ON context_symbols(file_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbols_text ON context_symbols(symbol_text);")

    conn.commit()
    conn.close()


def process_corpus_extraction(
    source_db: str,
    target_db: str,
    limit: int | None = None,
    batch_size: int = 300,
    vacuum: bool = False,
):
    if not os.path.exists(source_db):
        raise FileNotFoundError(f"未找到源数据库文件: {source_db}")

    print("=" * 80)
    print("🔬 Palimpsest - 会话语料资产解构与物理分层提纯")
    print("=" * 80)
    print(f"📦 源数据快照: {source_db}")
    print(f"🎯 目标语料库: {target_db}")
    if limit:
        print(f"⚠️  采样限制:   仅提取前 {limit} 场会话")
    print("-" * 80)

    # 1. 探测源库会话数量
    src_conn = sqlite3.connect(f"file:{os.path.abspath(source_db)}?mode=ro", uri=True)
    src_conn.row_factory = sqlite3.Row
    src_cur = src_conn.cursor()

    src_cur.execute("SELECT COUNT(*) FROM file_cache;")
    total_in_cache = src_cur.fetchone()[0]
    total_target = min(total_in_cache, limit) if limit else total_in_cache
    print(f"📊 扫描到待提纯会话总量: {total_in_cache:,} 场 (本次处理: {total_target:,} 场)")

    # 2. 初始化目标库
    init_target_database(target_db)
    dst_conn = sqlite3.connect(target_db)
    dst_cur = dst_conn.cursor()

    # 清空已有数据以保证可重入性
    dst_cur.execute("DELETE FROM context_symbols;")
    dst_cur.execute("DELETE FROM intent_turns;")
    dst_cur.execute("DELETE FROM sessions;")
    dst_conn.commit()

    # 尝试加载 tqdm 进度条，缺失时优雅降级
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_target, desc="⚡ 正在分层提纯", unit="session", colour="cyan")
    except ImportError:
        pbar = None

    t0 = time.perf_counter()
    processed_count = 0
    total_intent_tokens = 0
    total_symbols_extracted = 0

    session_batch = []
    turns_batch = []
    symbols_batch = []

    src_cur.execute("SELECT file_id, modified_time, data FROM file_cache ORDER BY modified_time DESC;")

    while True:
        rows = src_cur.fetchmany(100)
        if not rows:
            break

        for row in rows:
            file_id = row["file_id"]
            mtime_str = row["modified_time"]
            raw_json_str = row["data"]

            try:
                data = json.loads(raw_json_str)
            except (json.JSONDecodeError, TypeError):
                continue

            # 提取会话元信息
            name = data.get("name") or data.get("title") or "Untitled"
            model = data.get("runSettings", {}).get("model", "unknown").replace("models/", "")
            ctime_str = data.get("createTime") or data.get("createdTime") or mtime_str

            chunks = data.get("chunkedPrompt", {}).get("chunks", [])
            # 兼容非 chunk 结构
            if not chunks and "contents" in data:
                chunks = []
                for c in data.get("contents", []):
                    parts = c.get("parts", [])
                    txt = "\n".join(p.get("text", "") for p in parts if "text" in p)
                    chunks.append({"role": c.get("role", "user"), "text": txt})

            user_turns = 0
            model_turns = 0
            raw_tokens = 0

            # 逐轮拆解与解耦
            for turn_idx, chunk in enumerate(chunks, start=1):
                role = chunk.get("role", "user")
                tok = int(chunk.get("tokenCount", 0) or 0)
                raw_tokens += tok
                is_thought = 1 if chunk.get("isThought", False) else 0

                c_time = chunk.get("createTime")

                # 分支 A: 纯文本交互 (User Prompt / Model 回复 / Thinking)
                if "text" in chunk:
                    text = chunk.get("text", "") or ""
                    if is_thought:
                        # 思考链只保留前 1200 字符用于推理回溯，防止膨胀
                        snippet = text[:1200] if len(text) > 1200 else text
                        turns_batch.append((
                            file_id, turn_idx, "model", "thinking", snippet, tok, 1, 0, c_time
                        ))
                    else:
                        if role == "user":
                            user_turns += 1
                        else:
                            model_turns += 1

                        has_code = 1 if ("```" in text or "def " in text or "function" in text) else 0
                        turns_batch.append((
                            file_id, turn_idx, role, "text", text, tok, 0, has_code, c_time
                        ))

                        # 提纯符号
                        for s_type, s_text in extract_symbols_from_text(text):
                            symbols_batch.append((file_id, turn_idx, s_type, s_text))

                    total_intent_tokens += tok

                # 分支 B: 外部挂载代码库 / 附件 (关键去噪点！)
                elif "inlineFile" in chunk:
                    finfo = chunk.get("inlineFile", {})
                    dname = finfo.get("displayName") or finfo.get("name") or "未命名文件"
                    mime = finfo.get("mimeType", "application/octet-stream")
                    raw_b64 = finfo.get("data", "")
                    byte_size = len(raw_b64)

                    # 只保存结构化摘要，严禁将几万行代码塞入意图表
                    summary_text = f"[挂载代码库文件: {dname} ({mime}, 约 {byte_size // 1024} KB)]"
                    turns_batch.append((
                        file_id, turn_idx, "user", "inlineFile_summary", summary_text, 50, 0, 1, c_time
                    ))

                    # 将文件路径和文件名作为核心符号归档
                    symbols_batch.append((file_id, turn_idx, "filepath", dname))

                    # 若为可读源码，抽取其中的定义符号，但不存全文
                    if is_text_mime(mime) and raw_b64:
                        try:
                            decoded = base64.b64decode(raw_b64).decode("utf-8", errors="replace")
                            for s_type, s_text in extract_symbols_from_text(decoded[:20_000]):
                                symbols_batch.append((file_id, turn_idx, s_type, s_text))
                        except (ValueError, binascii.Error):
                            pass

                # 分支 C: 云端大文档挂载
                elif "driveDocument" in chunk:
                    doc_id = chunk.get("driveDocument", {}).get("id", "unknown")
                    turns_batch.append((
                        file_id, turn_idx, "user", "driveDocument_summary", f"[挂载云端文档 ID: {doc_id}]", 30, 0, 0, c_time
                    ))
                    symbols_batch.append((file_id, turn_idx, "drive_doc_id", doc_id))

                # 分支 D: 图片附件
                elif "inlineImage" in chunk:
                    mime = chunk.get("inlineImage", {}).get("mimeType", "image/png")
                    turns_batch.append((
                        file_id, turn_idx, "user", "image_summary", f"[图片附件: {mime}]", 258, 0, 0, c_time
                    ))

            session_batch.append((
                file_id, name, model, ctime_str, mtime_str, user_turns, model_turns, raw_tokens
            ))

            processed_count += 1
            if pbar:
                pbar.update(1)
            elif processed_count % 100 == 0:
                print(f"  -> 已提纯: {processed_count}/{total_target} ({(processed_count / total_target * 100):.1f}%)")

            # 周期性批量提交
            if len(session_batch) >= batch_size:
                dst_cur.executemany(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?);", session_batch
                )
                dst_cur.executemany(
                    """INSERT INTO intent_turns 
                       (file_id, turn_index, role, payload_type, text_content, token_count, is_thought, has_code, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""", turns_batch
                )
                if symbols_batch:
                    dst_cur.executemany(
                        "INSERT INTO context_symbols (file_id, turn_index, symbol_type, symbol_text) VALUES (?, ?, ?, ?);",
                        symbols_batch
                    )
                dst_conn.commit()
                total_symbols_extracted += len(symbols_batch)

                session_batch.clear()
                turns_batch.clear()
                symbols_batch.clear()

            if limit and processed_count >= limit:
                break

        if limit and processed_count >= limit:
            break

    # 最终残余批次刷盘
    if session_batch:
        dst_cur.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?);", session_batch)
        dst_cur.executemany(
            """INSERT INTO intent_turns 
               (file_id, turn_index, role, payload_type, text_content, token_count, is_thought, has_code, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""", turns_batch
        )
        if symbols_batch:
            dst_cur.executemany(
                "INSERT INTO context_symbols (file_id, turn_index, symbol_type, symbol_text) VALUES (?, ?, ?, ?);",
                symbols_batch
            )
        dst_conn.commit()
        total_symbols_extracted += len(symbols_batch)

    if pbar:
        pbar.close()

    elapsed = time.perf_counter() - t0

    # 最终 Checkpoint
    dst_cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    if vacuum:
        print("🧹 正在执行目标数据库 VACUUM 整理...")
        dst_cur.execute("VACUUM;")
        dst_conn.commit()

    src_size = os.path.getsize(source_db) / (1024 * 1024)
    dst_size = os.path.getsize(target_db) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("🎉 语料分层提纯圆满完成！")
    print("=" * 80)
    print(f"  • 处理会话总量:       {processed_count:,} 场")
    print(f"  • 提纯意图对话轮次:   {processed_count} 场会话包含的有效交互")
    print(f"  • 抽离代码/路径符号:  {total_symbols_extracted:,} 条独立索引符号")
    print(f"  • 核心意图 Token 规模: 约 {total_intent_tokens:,} Tokens")
    print(f"  • 物理存储体积优化:   {src_size:.1f} MB (原始) ➔ {dst_size:.1f} MB (提纯库, 降幅 {((src_size - dst_size) / src_size * 100):.1f}%)")
    print(f"  • 总耗时:             {elapsed:.2f} 秒 (处理速率: {processed_count / elapsed:.1f} 会话/秒)")
    print("=" * 80 + "\n")

    src_conn.close()
    dst_conn.close()


def main():
    parser = argparse.ArgumentParser(description="Palimpsest 语料分层提纯脚本")
    parser.add_argument("--source-db", default="data/cache.db", help="源 SQLite 数据库路径 (默认: data/cache.db)")
    parser.add_argument("--target-db", default="data/purified_corpus.db", help="提纯输出数据库路径 (默认: data/purified_corpus.db)")
    parser.add_argument("--limit", type=int, default=None, help="最大处理会话数限制 (用于快速调试)")
    parser.add_argument("--batch-size", type=int, default=300, help="写入批次大小 (默认: 300)")
    parser.add_argument("--vacuum", action="store_true", help="提取完成后执行数据库整理压缩")
    args = parser.parse_args()

    process_corpus_extraction(
        source_db=args.source_db,
        target_db=args.target_db,
        limit=args.limit,
        batch_size=args.batch_size,
        vacuum=args.vacuum,
    )


if __name__ == "__main__":
    main()