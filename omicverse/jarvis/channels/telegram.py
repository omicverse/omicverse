"""
Telegram bot handlers for OmicVerse Jarvis.

Architecture (OpenClaw-inspired):
  • handle_analysis() immediately acknowledges and spawns a background asyncio.Task
  • The background task streams LLM tokens by editing a single "thinking" message
  • Progress (code execution) is sent as separate messages
  • On completion: result header → figures with captions → summary → inline keyboard
  • /cancel stops the running task gracefully
"""
from __future__ import annotations

import atexit
import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import socket
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .channel_core import (
    gather_status,
    gather_usage,
    gather_workspace,
    perform_save,
    strip_local_paths,
)
from .. import _fmt
from ..config import default_state_dir
from ..model_help import render_model_help
from ..media_ingest import (
    PreparedImage,
    build_workspace_note,
    compose_multimodal_user_text,
)
from ..channel_media import (
    inbound_upload_dir,
    prepare_inbound_image_from_file,
    MAX_INBOUND_IMAGES,
)
from ..runtime import (
    AgentBridgeExecutionAdapter,
    ConversationRoute,
    DeliveryEvent,
    MessageEnvelope,
    MessageRuntime,
    MessageRouter,
)

logger = logging.getLogger("omicverse.jarvis")

_POLLING_RESTART_DELAY_SECONDS = 1.0
_POLLING_MAX_ATTEMPTS = 2

_SESSION_INIT_ERROR_TEMPLATE = (
    "Agent 初始化失败：{exc}\n"
    "请检查 --model 参数，或运行 "
    "<code>import omicverse as ov; print(ov.list_supported_models())</code> "
    "查看可用模型。"
)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

class AccessControl:
    """Allow all users when *allowed* is empty; otherwise whitelist."""

    def __init__(self, allowed: Optional[List[str]] = None) -> None:
        self._ids: Set[int] = set()
        self._usernames: Set[str] = set()
        for entry in (allowed or []):
            entry = entry.strip()
            if entry.lstrip("-").isdigit():
                self._ids.add(int(entry))
            else:
                self._usernames.add(entry.lstrip("@").lower())

    @property
    def _open(self) -> bool:
        return not self._ids and not self._usernames

    def allows(self, user_id: int, username: Optional[str]) -> bool:
        if self._open:
            return True
        if user_id in self._ids:
            return True
        if username and username.lower() in self._usernames:
            return True
        return False


def telegram_route_from_update(update: Any) -> ConversationRoute:
    chat = getattr(update, "effective_chat", None)
    msg = getattr(update, "effective_message", None)
    user = getattr(update, "effective_user", None)
    scope_type = "dm" if (chat and chat.type == "private") else "group"
    scope_id = str(chat.id if chat else getattr(user, "id", ""))
    thread_id = getattr(msg, "message_thread_id", None) if msg is not None else None
    return ConversationRoute(
        channel="telegram",
        scope_type=scope_type,
        scope_id=scope_id,
        thread_id=(str(thread_id) if thread_id else None),
        sender_id=(str(user.id) if user else None),
    )


def _normalize_bot_username(username: Optional[str]) -> str:
    return (username or "").lstrip("@").strip().lower()


def _message_mentions_bot(message: Any, bot_username: Optional[str]) -> bool:
    name = _normalize_bot_username(bot_username)
    raw_text = getattr(message, "text", None)
    text = (raw_text or getattr(message, "caption", None) or "").strip()
    if not name or not text:
        return False

    mention_token = f"@{name}"
    if mention_token in text.lower():
        return True

    entity_attr = "entities" if raw_text else "caption_entities"
    entities = list(getattr(message, entity_attr, None) or [])
    for entity in entities:
        entity_type = str(getattr(entity, "type", "")).lower()
        if "mention" not in entity_type:
            continue
        offset = int(getattr(entity, "offset", 0) or 0)
        length = int(getattr(entity, "length", 0) or 0)
        token = text[offset : offset + length].strip().lower()
        if token == mention_token:
            return True
    return False


def _message_replies_to_bot(message: Any, bot_username: Optional[str]) -> bool:
    reply = getattr(message, "reply_to_message", None)
    if reply is None:
        return False
    from_user = getattr(reply, "from_user", None)
    if from_user is None:
        return False
    if getattr(from_user, "is_bot", False):
        name = _normalize_bot_username(bot_username)
        reply_name = _normalize_bot_username(getattr(from_user, "username", None))
        return not name or name == reply_name
    return False


def telegram_trigger_for_update(update: Any, bot_username: Optional[str]) -> Optional[str]:
    route = telegram_route_from_update(update)
    if route.is_direct:
        message = getattr(update, "effective_message", None)
        if message is not None and (
            (getattr(message, "text", None) or "").strip()
            or (getattr(message, "caption", None) or "").strip()
            or getattr(message, "photo", None)
            or getattr(message, "document", None)
        ):
            return "direct"
        return "direct"

    message = getattr(update, "effective_message", None)
    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if not text:
        return None
    if _message_mentions_bot(message, bot_username):
        return "mention"
    if _message_replies_to_bot(message, bot_username):
        return "reply"
    return None


def _strip_leading_bot_mention(text: str, bot_username: Optional[str]) -> str:
    name = _normalize_bot_username(bot_username)
    if not name:
        return text.strip()
    pattern = rf"^\s*@{re.escape(name)}[\s,:-]*"
    return re.sub(pattern, "", text.strip(), count=1, flags=re.IGNORECASE)


def telegram_runtime_envelope(
    update: Any,
    bot_username: Optional[str],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    image_note: str = "",
) -> Optional[MessageEnvelope]:
    message = getattr(update, "effective_message", None)
    user = getattr(update, "effective_user", None)
    text = (getattr(message, "text", None) or getattr(message, "caption", None) or "").strip()
    if message is None or user is None:
        return None

    route = telegram_route_from_update(update)
    trigger = telegram_trigger_for_update(update, bot_username) or "implicit"
    normalized = _strip_leading_bot_mention(text, bot_username) if trigger == "mention" else text
    normalized = normalized.strip()
    request_content = list((metadata or {}).get("request_content") or [])
    if request_content or image_note:
        normalized = compose_multimodal_user_text(normalized, image_note)
    if not normalized:
        return None

    return MessageEnvelope(
        route=route,
        text=normalized,
        sender_id=str(user.id),
        sender_username=getattr(user, "username", None),
        message_id=str(getattr(message, "message_id", "")) or None,
        trigger=trigger,
        explicit_trigger=(trigger != "implicit"),
        metadata=dict(metadata or {}),
    )


class TelegramRuntimePresenter:
    _BORING = {"分析完成", "分析完成。", "task completed", "done", "完成"}
    _DRAFT_MAX = 2800

    def ack(self, envelope: MessageEnvelope, session: Any) -> List[DeliveryEvent]:
        if session.adata is not None:
            a = session.adata
            text = _fmt.ack_message(
                envelope.text,
                adata_info=f"{a.n_obs:,} cells × {a.n_vars:,} genes",
            )
        else:
            h5ad_files = session.list_h5ad_files()
            if h5ad_files:
                names = "\n".join(
                    f"  • <code>{_fmt.esc(f.name)}</code>" for f in h5ad_files[:5]
                )
                hint = (
                    f"💡  workspace 中检测到 {len(h5ad_files)} 个文件：\n"
                    f"{names}\n使用 <code>/load &lt;文件名&gt;</code> 加载"
                )
            else:
                hint = "💡  未检测到已加载数据，Agent 将自行加载数据"
            text = _fmt.ack_message(envelope.text, workspace_hint=hint)
        return [
            DeliveryEvent(
                route=envelope.route,
                kind="text",
                text=text,
                text_format="html",
            )
        ]

    def queue_started(self, route: ConversationRoute, queued_count: int) -> List[DeliveryEvent]:
        return [
            DeliveryEvent(
                route=route,
                kind="text",
                text=f"⏭  开始执行队列中的 {queued_count} 条请求…",
                text_format="html",
            )
        ]

    def draft_open(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="open",
            target="analysis-draft",
            text="💭  <b>思考中…</b>",
            text_format="html",
        )

    def draft_update(self, route: ConversationRoute, llm_text: str, progress: str) -> DeliveryEvent:
        body = _fmt.md_to_html(self._trim_for_draft(llm_text)) if llm_text.strip() else "<i>思考中…</i>"
        if progress:
            text = f"🔄  <code>{_fmt.esc(progress[:180])}</code>\n\n💭  {body}"
        else:
            text = f"💭  {body}"
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=text,
            text_format="html",
        )

    def draft_cancelled(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text="🚫  分析已取消。",
            text_format="html",
        )

    def analysis_error(self, route: ConversationRoute, error_text: str) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=_fmt.error_message(error_text),
            text_format="html",
        )

    def typing(self, route: ConversationRoute) -> Optional[DeliveryEvent]:
        return DeliveryEvent(route=route, kind="typing")

    def quick_chat_reply(self, route: ConversationRoute, text: str) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            text=_fmt.md_to_html(text),
            text_format="html",
        )

    def quick_chat_fallback(self, route: ConversationRoute) -> DeliveryEvent:
        return DeliveryEvent(
            route=route,
            kind="text",
            text="⏳  后台分析进行中，请等待完成。使用 <code>/cancel</code> 取消。",
            text_format="html",
        )

    def analysis_status(
        self,
        route: ConversationRoute,
        *,
        has_media: bool,
        has_reports: bool,
        has_artifacts: bool,
    ) -> Optional[DeliveryEvent]:
        if has_media:
            status = "✅  正在发送图片…"
        elif has_reports:
            status = "✅  正在发送报告…"
        elif has_artifacts:
            status = "✅  结果如下"
        else:
            return None
        return DeliveryEvent(
            route=route,
            kind="text",
            mode="edit",
            target="analysis-draft",
            text=status,
            text_format="html",
        )

    def final_events(
        self,
        route: ConversationRoute,
        *,
        session: Any,
        user_text: str,
        llm_text: str,
        result: Any,
    ) -> List[DeliveryEvent]:
        del user_text
        events: List[DeliveryEvent] = []

        a_cur = result.adata or session.adata
        a_info = f"{a_cur.n_obs:,} cells × {a_cur.n_vars:,} genes" if a_cur else ""

        summary = strip_local_paths((result.summary or "").strip())
        has_summary = bool(summary and summary.lower() not in self._BORING)
        long_summary = has_summary and len(summary) > 1200
        final_text = _fmt.md_to_html(summary) if has_summary and not long_summary else ""
        if (not final_text) and a_info:
            final_text = f"📊  <code>{_fmt.esc(a_info)}</code>"
        if not final_text:
            final_text = _fmt._DIV

        has_media = bool(result.figures)
        has_reports = bool(result.reports)
        artifacts = list(getattr(result, "artifacts", []) or [])
        has_artifacts = bool(artifacts)
        is_complex = has_media or has_reports or has_artifacts or long_summary

        if is_complex:
            explain = summary if has_summary else strip_local_paths(llm_text.strip())
            for i, rep in enumerate(result.reports or [], start=1):
                header = "📝  <b>分析报告</b>" if i == 1 else f"📝  <b>分析报告（续 {i}）</b>"
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="prose",
                        text=rep,
                        metadata={"header": header, "always_expand": False},
                    )
                )
            n_figs = len(result.figures or [])
            for i, png_bytes in enumerate(result.figures or [], start=1):
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="photo",
                        binary=png_bytes,
                        caption=_fmt.figure_caption(i, n_figs),
                    )
                )
            for art in artifacts:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="document",
                        binary=art.data,
                        filename=art.filename,
                        caption=f"📎  {art.filename}",
                    )
                )

            if explain:
                if len(explain) > 1200:
                    events.append(
                        DeliveryEvent(
                            route=route,
                            kind="prose",
                            text=explain,
                            metadata={"always_expand": False},
                        )
                    )
                    events.append(
                        DeliveryEvent(
                            route=route,
                            kind="text",
                            text=_fmt._DIV,
                            text_format="html",
                            controls=("save", "status", "memory"),
                        )
                    )
                else:
                    events.append(
                        DeliveryEvent(
                            route=route,
                            kind="text",
                            text=_fmt.md_to_html(explain),
                            text_format="html",
                            controls=("save", "status", "memory"),
                        )
                    )
            else:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="text",
                        text=final_text,
                        text_format="html",
                        controls=("save", "status", "memory"),
                    )
                )
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="text",
                    mode="edit",
                    target="analysis-draft",
                    text="✅  分析完成",
                    text_format="html",
                )
            )
            return events

        if llm_text.strip():
            final_html = _fmt.md_to_html(llm_text.strip())
            if len(final_html) > 3200 or "<pre>" in final_html:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="text",
                        mode="edit",
                        target="analysis-draft",
                        text="✅  分析完成，正文如下。",
                        text_format="html",
                    )
                )
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="prose",
                        text=llm_text.strip(),
                        metadata={"always_expand": False},
                    )
                )
            else:
                events.append(
                    DeliveryEvent(
                        route=route,
                        kind="text",
                        mode="edit",
                        target="analysis-draft",
                        text=final_html,
                        text_format="html",
                        controls=("save", "status", "memory"),
                    )
                )
                return events
        else:
            events.append(
                DeliveryEvent(
                    route=route,
                    kind="text",
                    mode="edit",
                    target="analysis-draft",
                    text="✅  分析完成",
                    text_format="html",
                )
            )

        events.append(
            DeliveryEvent(
                route=route,
                kind="text",
                text=final_text,
                text_format="html",
                controls=("save", "status", "memory"),
            )
        )
        return events

    @classmethod
    def _trim_for_draft(cls, text: str, max_len: int = _DRAFT_MAX) -> str:
        if len(text) <= max_len:
            return text
        head = int(max_len * 0.55)
        tail = max_len - head - 40
        if tail < 200:
            tail = 200
        return (
            text[:head].rstrip()
            + "\n\n[...内容较长，已省略中间部分...]\n\n"
            + text[-tail:].lstrip()
        )


class TelegramDelivery:
    def __init__(
        self,
        *,
        bot: Any,
        chat_lock_factory: Callable[[int], asyncio.Lock],
        keyboard_factory: Callable[[Tuple[str, ...]], Any],
    ) -> None:
        self._bot = bot
        self._chat_lock_factory = chat_lock_factory
        self._keyboard_factory = keyboard_factory
        self._targets: Dict[str, int] = {}

    async def deliver(self, event: DeliveryEvent) -> None:
        chat_id = int(event.route.scope_id)
        thread_id = int(event.route.thread_id) if event.route.thread_id else None
        if event.kind == "typing":
            try:
                kwargs = {"chat_id": chat_id, "action": "typing"}
                if thread_id is not None:
                    kwargs["message_thread_id"] = thread_id
                await self._bot.send_chat_action(**kwargs)
            except Exception:
                pass
            return

        if event.kind == "prose":
            async with self._chat_lock_factory(chat_id):
                await _fmt.send_prose(
                    self._bot,
                    chat_id,
                    event.text,
                    header=str(event.metadata.get("header") or ""),
                    always_expand=bool(event.metadata.get("always_expand")),
                    message_thread_id=thread_id,
                )
            return

        if event.kind == "photo":
            async with self._chat_lock_factory(chat_id):
                await self._send_photo_or_file(
                    chat_id,
                    event.binary or b"",
                    event.caption,
                    thread_id=thread_id,
                )
            return

        if event.kind == "document":
            async with self._chat_lock_factory(chat_id):
                await self._send_document(
                    chat_id,
                    filename=event.filename or "artifact.bin",
                    data=event.binary or b"",
                    caption=event.caption,
                    thread_id=thread_id,
                )
            return

        if event.kind != "text":
            return

        reply_markup = self._keyboard_factory(event.controls) if event.controls else None
        parse_mode = "HTML" if event.text_format == "html" else None
        async with self._chat_lock_factory(chat_id):
            if event.mode == "open":
                msg_id = await self._send_text(
                    chat_id,
                    event.text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    thread_id=thread_id,
                )
                if msg_id is not None and event.target:
                    self._targets[self._target_key(event)] = msg_id
                return

            if event.mode == "edit":
                target_id = self._targets.get(self._target_key(event))
                if target_id is not None:
                    ok = await self._edit_text(
                        chat_id,
                        target_id,
                        event.text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                    if ok:
                        return
                msg_id = await self._send_text(
                    chat_id,
                    event.text,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
                    thread_id=thread_id,
                )
                if msg_id is not None and event.target:
                    self._targets[self._target_key(event)] = msg_id
                return

            await self._send_text(
                chat_id,
                event.text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
                thread_id=thread_id,
            )

    @staticmethod
    def _target_key(event: DeliveryEvent) -> str:
        return f"{event.route.route_key()}::{event.target or ''}"

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Any = None,
        thread_id: Optional[int] = None,
    ) -> Optional[int]:
        try:
            kwargs = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            message = await self._bot.send_message(**kwargs)
            return getattr(message, "message_id", None)
        except Exception:
            pass
        try:
            kwargs = {
                "chat_id": chat_id,
                "text": self._strip_html(text),
                "reply_markup": reply_markup,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            message = await self._bot.send_message(**kwargs)
            return getattr(message, "message_id", None)
        except Exception as exc:
            logger.warning("Failed to send message: %s", exc)
            return None

    async def _edit_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Any = None,
    ) -> bool:
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return True
        except Exception as exc:
            if self._is_not_modified_error(exc):
                return True
            pass
        try:
            await self._bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=self._strip_html(text),
                reply_markup=reply_markup,
            )
            return True
        except Exception as exc:
            if self._is_not_modified_error(exc):
                return True
            return False

    async def _send_photo_or_file(
        self,
        chat_id: int,
        png_bytes: bytes,
        caption: str,
        *,
        thread_id: Optional[int] = None,
    ) -> None:
        try:
            kwargs = {
                "chat_id": chat_id,
                "photo": BytesIO(png_bytes),
                "caption": caption,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            await self._bot.send_photo(**kwargs)
            return
        except Exception:
            pass
        await self._send_document(
            chat_id,
            filename="figure.png",
            data=png_bytes,
            caption=caption,
            thread_id=thread_id,
        )

    async def _send_document(
        self,
        chat_id: int,
        *,
        filename: str,
        data: bytes,
        caption: str,
        thread_id: Optional[int] = None,
    ) -> None:
        try:
            kwargs = {
                "chat_id": chat_id,
                "document": BytesIO(data),
                "filename": filename,
                "caption": caption,
            }
            if thread_id is not None:
                kwargs["message_thread_id"] = thread_id
            await self._bot.send_document(**kwargs)
        except Exception as exc:
            logger.warning("Failed to send artifact %s: %s", filename, exc)

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text or "")

    @staticmethod
    def _is_not_modified_error(exc: Exception) -> bool:
        return "message is not modified" in str(exc).strip().lower()


# ---------------------------------------------------------------------------
# Bot builder
# ---------------------------------------------------------------------------


@dataclass
class _PollingState:
    conflict_detected: bool = False
    conflict_message: str = ""


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _telegram_runtime_dir() -> Path:
    runtime_dir = default_state_dir() / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _telegram_lock_path(token: str) -> Path:
    return _telegram_runtime_dir() / f"telegram-{_token_fingerprint(token)}.json"


def _load_runtime_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_runtime_record(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "token_hash": _token_fingerprint(token),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cleanup_runtime_record(path: Path) -> None:
    record = _load_runtime_record(path)
    if int(record.get("pid") or 0) != os.getpid():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.debug("Failed to remove Telegram runtime record %s", path, exc_info=True)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process(pid: int, *, label: str, grace_seconds: float = 5.0) -> bool:
    if pid <= 0 or pid == os.getpid():
        return True
    if not _process_exists(pid):
        return True

    logger.warning("Stopping %s (pid=%s) before Telegram polling takeover.", label, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warning("Cannot stop %s (pid=%s): permission denied.", label, pid)
        return False

    deadline = time.time() + max(grace_seconds, 0.0)
    while time.time() < deadline:
        if not _process_exists(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        logger.warning("Cannot force stop %s (pid=%s): permission denied.", label, pid)
        return False

    for _ in range(10):
        if not _process_exists(pid):
            return True
        time.sleep(0.1)
    return not _process_exists(pid)


def _claim_telegram_instance(token: str) -> Path:
    path = _telegram_lock_path(token)
    record = _load_runtime_record(path)
    previous_pid = int(record.get("pid") or 0)
    if (
        previous_pid
        and previous_pid != os.getpid()
        and str(record.get("token_hash") or "") == _token_fingerprint(token)
    ):
        stopped = _terminate_process(
            previous_pid,
            label="existing OmicVerse Jarvis Telegram bot",
        )
        if not stopped:
            logger.warning(
                "Tracked Telegram bot instance pid=%s is still running; polling will retry once if Telegram returns 409 Conflict.",
                previous_pid,
            )

    _write_runtime_record(path, token)
    atexit.register(_cleanup_runtime_record, path)
    return path


def _register_polling_error_handler(app: Any, *, polling_state: _PollingState, conflict_type: type[BaseException]) -> None:
    async def _handle_error(update: object, context: Any) -> None:
        error = getattr(context, "error", None)
        if isinstance(error, conflict_type):
            polling_state.conflict_detected = True
            polling_state.conflict_message = str(error)
            logger.warning(
                "Telegram returned 409 Conflict because another getUpdates consumer is active. "
                "Stopping this polling loop and retrying once."
            )
            context.application.stop_running()
            return

        if error is not None:
            logger.error(
                "Unhandled Telegram bot error: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    app.add_error_handler(_handle_error)


def _telegram_conflict_error(conflict_message: str) -> str:
    detail = f" ({conflict_message})" if conflict_message else ""
    return (
        "Telegram polling still conflicts with another bot instance after Jarvis retried once"
        f"{detail}. Stop the other process using this bot token and start `omicverse jarvis` again."
    )


def _run_polling_with_restart(
    *,
    application_factory: Any,
    conflict_type: type[BaseException],
    restart_delay_seconds: float = _POLLING_RESTART_DELAY_SECONDS,
    max_attempts: int = _POLLING_MAX_ATTEMPTS,
    stop_event: Optional[threading.Event] = None,
) -> None:
    last_conflict_message = ""
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        app = application_factory()
        if stop_event is not None:
            def _stop_when_requested() -> None:
                stop_event.wait()
                try:
                    app.stop_running()
                except Exception:
                    pass

            threading.Thread(
                target=_stop_when_requested,
                daemon=True,
                name="telegram-stop-watcher",
            ).start()
        polling_state = _PollingState()
        _register_polling_error_handler(
            app,
            polling_state=polling_state,
            conflict_type=conflict_type,
        )
        logger.info("OmicVerse Jarvis bot starting (polling)...")
        app.run_polling(
            drop_pending_updates=True,
            stop_signals=None,
        )
        if stop_event is not None and stop_event.is_set():
            return
        if not polling_state.conflict_detected:
            return
        last_conflict_message = polling_state.conflict_message
        if attempt >= attempts:
            break
        time.sleep(max(restart_delay_seconds, 0.0))

    raise RuntimeError(_telegram_conflict_error(last_conflict_message))


def run_bot(
    token: str,
    session_manager: Any,
    access_control: AccessControl,
    verbose: bool = False,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Build and start the Telegram application (blocking)."""
    try:
        from telegram.error import Conflict
        from telegram.ext import Application
    except ImportError as exc:
        raise ImportError(
            "python-telegram-bot is required.  "
            "Install with: pip install omicverse[jarvis]"
        ) from exc

    _claim_telegram_instance(token)

    def _build_application() -> Any:
        app = Application.builder().token(token).concurrent_updates(True).build()
        _register_handlers(app, session_manager, access_control, verbose)
        return app

    _run_polling_with_restart(
        application_factory=_build_application,
        conflict_type=Conflict,
        stop_event=stop_event,
    )


def _register_handlers(app: Any, sm: Any, ac: AccessControl, verbose: bool) -> None:
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.ext import (
        CallbackQueryHandler,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )

    # Per-chat outbound lock (OpenClaw-style channel sequencing)
    _chat_locks: Dict[int, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Guard / session helpers
    # ------------------------------------------------------------------

    async def _guard(update: Update) -> bool:
        user = update.effective_user
        if user is None:
            return False
        if not ac.allows(user.id, user.username):
            await update.message.reply_text("⛔ 您没有访问权限。")
            return False
        return True

    async def _get_session(update: Update, route: Optional[ConversationRoute] = None):
        try:
            return runtime.get_session(route or telegram_route_from_update(update))
        except Exception as exc:
            logger.exception("Failed to create session")
            await update.message.reply_text(
                _fmt.error_message(_SESSION_INIT_ERROR_TEMPLATE.format(exc=exc)),
                parse_mode="HTML",
            )
            return None

    def _analysis_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("💾 保存",  callback_data="jarvis:save"),
            InlineKeyboardButton("📊 状态",  callback_data="jarvis:status"),
            InlineKeyboardButton("🧠 历史",  callback_data="jarvis:memory"),
        ]])

    def _chat_lock(chat_id: int) -> asyncio.Lock:
        lock = _chat_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            _chat_locks[chat_id] = lock
        return lock

    async def _prepare_telegram_request_images(
        message: Any,
        bot: Any,
        session: Any,
    ) -> List[PreparedImage]:
        prepared: List[PreparedImage] = []
        upload_dir = inbound_upload_dir(session.workspace_dir, "telegram")
        if getattr(message, "photo", None):
            photo = message.photo[-1]
            tg_file = await bot.get_file(photo.file_id)
            raw_path = upload_dir / f"telegram_photo_{getattr(photo, 'file_unique_id', photo.file_id)}.jpg"
            await tg_file.download_to_drive(raw_path)
            image = prepare_inbound_image_from_file(raw_path, workspace_root=session.workspace_dir, channel_name="telegram")
            if image.path != raw_path and raw_path.exists():
                raw_path.unlink(missing_ok=True)
            prepared.append(image)

        document = getattr(message, "document", None)
        doc_mime = str(getattr(document, "mime_type", None) or "").strip().lower() if document else ""
        doc_name = str(getattr(document, "file_name", None) or "").strip()
        if document is not None and doc_mime.startswith("image/") and not doc_name.lower().endswith(".h5ad"):
            tg_file = await bot.get_file(document.file_id)
            raw_path = upload_dir / (doc_name or f"telegram_image_{document.file_unique_id}")
            await tg_file.download_to_drive(raw_path)
            image = prepare_inbound_image_from_file(raw_path, workspace_root=session.workspace_dir, channel_name="telegram", mime_type=doc_mime)
            if image.path != raw_path and raw_path.exists():
                raw_path.unlink(missing_ok=True)
            prepared.append(image)

        return prepared[:MAX_INBOUND_IMAGES]

    async def _handle_incoming_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        route = telegram_route_from_update(update)
        session = await _get_session(update, route)
        if session is None:
            return

        bot_username = getattr(context.bot, "username", None)
        request_images = await _prepare_telegram_request_images(update.effective_message, context.bot, session)
        image_note = build_workspace_note(
            session.workspace_dir,
            request_images,
            header="[Attached Telegram images saved in workspace]",
        )
        envelope = telegram_runtime_envelope(
            update,
            bot_username,
            metadata={
                "request_content": [item.request_block for item in request_images],
            } if request_images else {},
            image_note=image_note,
        )
        if envelope is None:
            return
        try:
            await runtime.handle_message(envelope)
        except Exception as exc:
            logger.exception("Failed to handle Telegram analysis message")
            await update.message.reply_text(
                _fmt.error_message(_SESSION_INIT_ERROR_TEMPLATE.format(exc=exc)),
                parse_mode="HTML",
            )

    def _keyboard_for_controls(controls: Tuple[str, ...]) -> Any:
        if set(controls) == {"save", "status", "memory"}:
            return _analysis_keyboard()
        return None

    delivery = TelegramDelivery(
        bot=app.bot,
        chat_lock_factory=_chat_lock,
        keyboard_factory=_keyboard_for_controls,
    )
    runtime = MessageRuntime(
        router=MessageRouter(sm),
        presenter=TelegramRuntimePresenter(),
        execution_adapter=AgentBridgeExecutionAdapter(),
        deliver=delivery.deliver,
        web_bridge=getattr(sm, "gateway_web_bridge", None),
    )

    # ------------------------------------------------------------------
    # /start
    # ------------------------------------------------------------------

    async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        text = (
            "👋  <b>OmicVerse Jarvis</b>\n"
            f"{_fmt._DIV}\n"
            "移动端单细胞分析助手，支持中英文自然语言指令。\n\n"
            "<b>快速开始</b>\n"
            "1. 通过 <code>scp</code>/<code>sftp</code> 上传 .h5ad 到 workspace\n"
            "2. <code>/workspace</code> 查看文件 → <code>/load</code> 加载\n"
            "3. 直接发送分析需求，例如：\n"
            "   <i>质控 nUMI&gt;500 mito&lt;0.2，然后标准化、UMAP</i>\n\n"
            "<b>数据命令</b>\n"
            "<code>/workspace</code>  查看 workspace\n"
            "<code>/ls</code>         列出文件\n"
            "<code>/find</code>       搜索文件\n"
            "<code>/load</code>       加载数据\n"
            "<code>/shell</code>      执行 shell 命令\n\n"
            "<b>会话命令</b>\n"
            "<code>/kernel</code>     当前 kernel 状态\n"
            "<code>/kernel ls</code>  列出所有 kernel\n"
            "<code>/kernel new x</code> 新建并切换 kernel\n"
            "<code>/kernel use x</code> 切换 kernel\n"
            "<code>/memory</code>     分析历史\n"
            "<code>/usage</code>      token 用量\n"
            "<code>/model</code>      切换模型\n"
            "<code>/status</code>     数据状态\n"
            "<code>/save</code>       下载 h5ad\n"
            "<code>/cancel</code>     取消当前分析\n"
            "<code>/reset</code>      重置会话\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # /help
    # ------------------------------------------------------------------

    async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        text = (
            "📚  <b>使用指南</b>\n"
            f"{_fmt._DIV}\n"
            "<b>分析示例</b>\n"
            "• <code>质控 nUMI&gt;500 mito&lt;0.2</code>\n"
            "• <code>标准化、高变基因、PCA、UMAP</code>\n"
            "• <code>Leiden 聚类 resolution=0.5</code>\n"
            "• <code>差异表达 cluster 1 vs 2</code>\n"
            "• <code>GO 富集分析 cluster 0</code>\n\n"
            "<b>数据管理</b>\n"
            "• <code>/workspace</code> — workspace 概览\n"
            "• <code>/ls [路径]</code> — 列出文件\n"
            "• <code>/find &lt;模式&gt;</code> — 搜索文件\n"
            "• <code>/load &lt;文件名&gt;</code> — 加载数据\n"
            "• <code>/shell &lt;命令&gt;</code> — 执行 shell"
            "（白名单：ls find cat head wc file du pwd tree）\n\n"
            "<b>会话管理</b>\n"
            "• <code>/kernel</code> — 当前 kernel 健康 + prompt 余量\n"
            "• <code>/kernel ls</code> — 列出可用 kernels\n"
            "• <code>/kernel new 名称</code> — 新建并切换 kernel\n"
            "• <code>/kernel use 名称</code> — 切换到指定 kernel\n"
            "• <code>/memory</code> — 近两天分析日志\n"
            "• <code>/usage</code> — 最近一次 token 用量\n"
            "• <code>/model [名称]</code> — 查看/切换 LLM 模型\n"
            "• <code>/status</code> — 当前数据信息\n"
            "• <code>/save</code> — 下载 .h5ad\n"
            "• <code>/cancel</code> — 取消正在运行的分析\n"
            "• <code>/reset</code> — 清空会话并重启 kernel\n\n"
            "<b>自定义指令</b>\n"
            "在 workspace 创建 <code>AGENTS.md</code>，写入偏好（语言、分析风格等），"
            "每次请求自动注入。"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ------------------------------------------------------------------
    # /status
    # ------------------------------------------------------------------

    async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user    = update.effective_user
        route = telegram_route_from_update(update)
        session = await _get_session(update, route)
        if session is None:
            return

        info = gather_status(
            session,
            session_manager=sm,
            is_running=runtime.task_state(route).running,
        )

        lines = [f"👤  <b>用户</b>  <code>{user.id}</code>"]
        if info.kernel_name:
            lines.append(f"🧩  Kernel：<code>{_fmt.esc(info.kernel_name)}</code>")
        if info.adata_shape:
            n_obs, n_vars = info.adata_shape
            lines.append(f"🔬  {n_obs:,} cells × {n_vars:,} genes")
            if info.obs_columns:
                cols = ", ".join(info.obs_columns)
                lines.append(f"📋  obs: <code>{_fmt.esc(cols)}</code>")
        else:
            lines.append("📭  暂无数据  ·  使用 <code>/load</code> 加载")

        if info.is_running:
            lines.append("⚙️  分析中…  ·  <code>/cancel</code> 取消")

        if info.prompt_count is not None:
            mp = info.max_prompts
            if getattr(session, "max_prompts_setting", 0) <= 0:
                mp = "∞"
            lines.append(f"💬  会话  {info.prompt_count}/{mp}")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /reset
    # ------------------------------------------------------------------

    async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        route = telegram_route_from_update(update)
        await runtime.cancel(route)
        session = await _get_session(update, route)
        if session is None:
            return
        session.reset()
        await update.message.reply_text(
            "✅  会话已重置，kernel 已重启。\n"
            "<i>变量已清空，adata 仍可通过 /load 重新加载。</i>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /cancel
    # ------------------------------------------------------------------

    async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        cancelled = await runtime.cancel(telegram_route_from_update(update))
        if cancelled:
            await update.message.reply_text("🚫  分析已取消。")
        else:
            await update.message.reply_text("ℹ️  当前没有正在运行的分析。")

    # ------------------------------------------------------------------
    # /save
    # ------------------------------------------------------------------

    async def handle_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return

        result = perform_save(session)
        if result.no_data:
            await update.message.reply_text("❌  没有数据，请先 <code>/load</code> 加载。", parse_mode="HTML")
            return

        await update.message.reply_text("⏳  正在保存…")
        if result.success and result.path:
            n_obs, n_vars = result.adata_shape
            with open(result.path, "rb") as fh:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=fh,
                    filename="current.h5ad",
                    caption=f"💾  {n_obs:,} cells × {n_vars:,} genes",
                )
        else:
            await update.message.reply_text(f"❌  {result.error or '保存失败，请重试。'}")

    # ------------------------------------------------------------------
    # /usage
    # ------------------------------------------------------------------

    async def handle_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return

        info = gather_usage(session)
        if not info.has_data:
            await update.message.reply_text("ℹ️  暂无用量数据，请先进行一次分析。")
            return

        lines = [
            "📊  <b>Token 用量</b>  （最近一次）",
            _fmt._DIV,
            f"输入：<code>{info.input_tokens}</code>",
            f"输出：<code>{info.output_tokens}</code>",
            f"合计：<code>{info.total_tokens}</code>",
        ]
        if info.cache_read and info.cache_read != "?":
            lines.append(f"缓存读取：<code>{info.cache_read}</code>")
        if info.cache_creation and info.cache_creation != "?":
            lines.append(f"缓存写入：<code>{info.cache_creation}</code>")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /model [name]
    # ------------------------------------------------------------------

    async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        args = context.args or []

        if not args:
            await update.message.reply_text(
                render_model_help(sm._model, html=True),
                parse_mode="HTML",
            )
            return

        new_model = args[0]
        sm._model = new_model
        await update.message.reply_text(
            f"✅  模型已切换为 <code>{_fmt.esc(new_model)}</code>\n"
            f"<i>请 /reset 重启 kernel 使新模型生效（当前 kernel 不受影响）。</i>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /kernel
    # ------------------------------------------------------------------

    async def handle_kernel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        user_id = update.effective_user.id
        args = context.args or []

        # /kernel  -> status of active kernel
        if not args:
            route = telegram_route_from_update(update)
            session = await _get_session(update, route)
            if session is None:
                return
            st    = session.kernel_status()
            alive = st["alive"]
            p     = st["prompt_count"]
            mp    = st["max_prompts"]
            sid   = st["session_id"] or "—"
            kname = sm.get_active_kernel(user_id)

            icon      = "🟢" if alive else "🔴"
            remaining = (mp - p) if isinstance(mp, int) else "∞"

            lines = [
                "⚙️  <b>Kernel 状态</b>",
                _fmt._DIV,
                f"🧩  当前：<code>{_fmt.esc(kname)}</code>",
                f"{icon}  {'运行中' if alive else '未启动 / 已关闭'}",
                f"💬  Prompts：<code>{p}</code> / <code>{mp}</code>（剩余 {remaining}）",
                f"🆔  Session：<code>{_fmt.esc(str(sid))}</code>",
                "",
                "子命令：<code>/kernel ls</code> · <code>/kernel new 名称</code> · <code>/kernel use 名称</code>",
            ]
            if isinstance(mp, int) and p >= mp * 0.8:
                lines += [
                    "",
                    "⚠️  即将到达上限，下次分析后 kernel 将重启（变量清空）。",
                    "   可用 <code>/reset</code> 手动重启，或启动时增大 <code>--max-prompts</code>。",
                ]
            if runtime.task_state(route).running:
                lines += ["", "⚙️  当前有分析正在运行  ·  <code>/cancel</code> 取消"]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        sub = args[0].lower()
        if sub in {"ls", "list"}:
            active = sm.get_active_kernel(user_id)
            names = sm.list_kernels(user_id)
            lines = [
                "🧩  <b>Kernel 列表</b>",
                _fmt._DIV,
            ]
            for name in names:
                mark = "✅" if name == active else "•"
                lines.append(f"{mark}  <code>{_fmt.esc(name)}</code>")
            lines += [
                "",
                "切换：<code>/kernel use 名称</code>",
                "新建：<code>/kernel new 名称</code>",
            ]
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            return

        if sub in {"new", "create", "use", "switch"}:
            if runtime.task_state(telegram_route_from_update(update)).running:
                await update.message.reply_text(
                    "⏳  当前有分析正在运行，请先等待完成或使用 <code>/cancel</code>。",
                    parse_mode="HTML",
                )
                return
            if len(args) < 2:
                await update.message.reply_text(
                    "用法：<code>/kernel new 名称</code> 或 <code>/kernel use 名称</code>\n"
                    "名称规则：字母/数字/._-，长度 1-32。",
                    parse_mode="HTML",
                )
                return
            name = args[1]
            try:
                if sub in {"new", "create"}:
                    sm.create_kernel(user_id, name, switch=True)
                    action = "新建并切换"
                else:
                    sm.switch_kernel(user_id, name, create=False)
                    action = "切换"
            except Exception as exc:
                await update.message.reply_text(
                    _fmt.error_message(str(exc)), parse_mode="HTML"
                )
                return
            await update.message.reply_text(
                f"✅  已{action}到 kernel：<code>{_fmt.esc(sm.get_active_kernel(user_id))}</code>",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text(
            "用法：<code>/kernel</code> | <code>/kernel ls</code> | "
            "<code>/kernel new 名称</code> | <code>/kernel use 名称</code>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /workspace
    # ------------------------------------------------------------------

    async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return

        ws_info = gather_workspace(session)

        lines = [
            "📁  <b>Workspace</b>",
            _fmt._DIV,
            f"<code>{ws_info.path}</code>",
            "",
        ]
        if ws_info.h5ad_files:
            lines.append(f"📊  <b>数据文件</b>  ({ws_info.h5ad_total})")
            for name, mb in ws_info.h5ad_files:
                if mb is not None:
                    lines.append(f"  • <code>{_fmt.esc(name)}</code>  <i>{mb:.1f} MB</i>")
                else:
                    lines.append(f"  • <code>{_fmt.esc(name)}</code>")
            if ws_info.h5ad_total > len(ws_info.h5ad_files):
                lines.append(f"  <i>… 还有 {ws_info.h5ad_total - len(ws_info.h5ad_files)} 个</i>")
        else:
            lines.append("📊  <b>数据文件</b>  (空)")
            lines.append(f"  <i>scp *.h5ad user@host:{ws_info.path}</i>")

        lines += [
            "",
            f"📋  AGENTS.md  {'✅' if ws_info.has_agents_md else '—'}",
            f"🧠  今日记忆  {'✅' if ws_info.has_today_memory else '—'}",
            "",
            "<code>/load &lt;文件名&gt;</code>  ·  <code>/ls</code>  ·  <code>/memory</code>",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    # ------------------------------------------------------------------
    # /ls [subpath]
    # ------------------------------------------------------------------

    async def handle_ls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        args    = context.args or []
        subpath = args[0] if args else ""
        cmd     = f"ls -lh {subpath}".strip() if subpath else "ls -lh"
        out     = session.shell.exec(cmd, cwd=session.workspace)
        await _fmt.send_code(context.bot, update.effective_chat.id, out, header=f"$ {cmd}")

    # ------------------------------------------------------------------
    # /find <pattern>
    # ------------------------------------------------------------------

    async def handle_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "用法：<code>/find &lt;模式&gt;</code>，例如 <code>/find *.h5ad</code>",
                parse_mode="HTML",
            )
            return
        pattern = args[0]
        cmd     = f"find . -name {pattern}"
        out     = session.shell.exec(cmd, cwd=session.workspace)
        await _fmt.send_code(context.bot, update.effective_chat.id, out, header=f"$ {cmd}")

    # ------------------------------------------------------------------
    # /load <filename>
    # ------------------------------------------------------------------

    async def handle_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "用法：<code>/load &lt;文件名&gt;</code>，例如 <code>/load pbmc3k.h5ad</code>\n"
                "<code>/workspace</code> 查看可用文件。",
                parse_mode="HTML",
            )
            return

        filename = args[0]
        await update.message.reply_text(
            f"⏳  加载 <code>{_fmt.esc(filename)}</code>…", parse_mode="HTML"
        )
        try:
            adata = session.load_from_workspace(filename)
        except Exception as exc:
            logger.exception("Failed to load from workspace")
            await update.message.reply_text(_fmt.error_message(str(exc)), parse_mode="HTML")
            return

        if adata is None:
            h5ad_files = session.list_h5ad_files()
            hint = ""
            if h5ad_files:
                names = "  ".join(f.name for f in h5ad_files[:5])
                hint  = f"\n可用文件：<code>{_fmt.esc(names)}</code>"
            await update.message.reply_text(
                f"❌  未找到 <code>{_fmt.esc(filename)}</code>{hint}", parse_mode="HTML"
            )
            return

        await update.message.reply_text(
            f"✅  加载成功\n{_fmt._DIV}\n"
            f"🔬  <b>{adata.n_obs:,} cells × {adata.n_vars:,} genes</b>\n"
            f"📁  <code>{_fmt.esc(filename)}</code>",
            parse_mode="HTML",
        )

    # ------------------------------------------------------------------
    # /shell <cmd>
    # ------------------------------------------------------------------

    async def handle_shell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        raw   = update.message.text or ""
        parts = raw.split(None, 1)
        if len(parts) < 2:
            await update.message.reply_text(
                "用法：<code>/shell &lt;命令&gt;</code>\n"
                "允许：<code>ls  find  cat  head  wc  file  du  pwd  tree</code>",
                parse_mode="HTML",
            )
            return
        cmd = parts[1].strip()
        out = session.shell.exec(cmd, cwd=session.workspace)
        await _fmt.send_code(context.bot, update.effective_chat.id, out, header=f"$ {cmd}")

    # ------------------------------------------------------------------
    # /memory
    # ------------------------------------------------------------------

    async def handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        session = await _get_session(update)
        if session is None:
            return
        text = session.get_recent_memory_text()
        async with _chat_lock(update.effective_chat.id):
            await _fmt.send_prose(
                context.bot,
                update.effective_chat.id,
                text,
                header="🧠  <b>分析历史</b>（近两天）",
                always_expand=True,
                message_thread_id=getattr(update.effective_message, "message_thread_id", None),
            )

    # ------------------------------------------------------------------
    # Document handler (.h5ad upload)
    # ------------------------------------------------------------------

    async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        doc = update.message.document
        if doc is None:
            return
        filename = doc.file_name or ""
        if str(getattr(doc, "mime_type", "") or "").startswith("image/") and not filename.endswith(".h5ad"):
            await _handle_incoming_analysis(update, context)
            return
        if not filename.endswith(".h5ad"):
            await update.message.reply_text("⚠️  请发送 <code>.h5ad</code> 格式的文件。", parse_mode="HTML")
            return

        await update.message.reply_text("⏳  正在下载并加载…")
        session = await _get_session(update)
        if session is None:
            return
        try:
            tg_file = await context.bot.get_file(doc.file_id)
            dest    = str(session.workspace_dir / "current.h5ad")
            await tg_file.download_to_drive(dest)
            import anndata
            session.adata = anndata.read_h5ad(dest)
            a = session.adata
            await update.message.reply_text(
                f"✅  加载成功\n{_fmt._DIV}\n"
                f"🔬  <b>{a.n_obs:,} cells × {a.n_vars:,} genes</b>\n"
                f"📁  <code>{_fmt.esc(filename)}</code>",
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.exception("Failed to load h5ad")
            await update.message.reply_text(_fmt.error_message(str(exc)), parse_mode="HTML")

    # ------------------------------------------------------------------
    # Inline keyboard callback handler
    # ------------------------------------------------------------------

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()

        user    = query.from_user
        chat_id = query.message.chat_id
        if not ac.allows(user.id, user.username):
            return

        data = query.data or ""
        try:
            route = telegram_route_from_update(update)
            session = runtime.get_session(route)
        except Exception:
            return

        if data == "jarvis:save":
            result = perform_save(session)
            if result.no_data:
                await context.bot.send_message(chat_id=chat_id, text="❌  没有数据。")
                return
            if result.success and result.path:
                n_obs, n_vars = result.adata_shape
                with open(result.path, "rb") as fh:
                    await context.bot.send_document(
                        chat_id=chat_id,
                        document=fh,
                        filename="current.h5ad",
                        caption=f"💾  {n_obs:,} cells × {n_vars:,} genes",
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌  {result.error or '保存失败。'}",
                )

        elif data == "jarvis:status":
            info = gather_status(session, is_running=runtime.task_state(route).running)
            if info.adata_shape:
                n_obs, n_vars = info.adata_shape
                lines = [f"🔬  {n_obs:,} cells × {n_vars:,} genes"]
                if info.obs_columns:
                    cols = ", ".join(info.obs_columns)
                    lines.append(f"📋  obs: <code>{_fmt.esc(cols)}</code>")
                await context.bot.send_message(
                    chat_id=chat_id, text="\n".join(lines), parse_mode="HTML"
                )
            else:
                await context.bot.send_message(chat_id=chat_id, text="📭  暂无数据。")

        elif data == "jarvis:memory":
            text = session.get_recent_memory_text()
            async with _chat_lock(chat_id):
                await _fmt.send_prose(
                    context.bot,
                    chat_id,
                    text,
                    header="🧠  <b>分析历史</b>",
                    always_expand=True,
                    message_thread_id=getattr(query.message, "message_thread_id", None),
                )

    # ------------------------------------------------------------------
    # Text analysis handler  — launches background task immediately
    # ------------------------------------------------------------------

    async def handle_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _guard(update):
            return
        await _handle_incoming_analysis(update, context)

    # ------------------------------------------------------------------
    # Register all handlers
    # ------------------------------------------------------------------

    app.add_handler(CommandHandler("start",     handle_start))
    app.add_handler(CommandHandler("help",      handle_help))
    app.add_handler(CommandHandler("status",    handle_status))
    app.add_handler(CommandHandler("reset",     handle_reset))
    app.add_handler(CommandHandler("cancel",    handle_cancel))
    app.add_handler(CommandHandler("save",      handle_save))
    app.add_handler(CommandHandler("usage",     handle_usage))
    app.add_handler(CommandHandler("model",     handle_model))
    app.add_handler(CommandHandler("kernel",    handle_kernel))
    app.add_handler(CommandHandler("workspace", handle_workspace))
    app.add_handler(CommandHandler("ls",        handle_ls))
    app.add_handler(CommandHandler("find",      handle_find))
    app.add_handler(CommandHandler("load",      handle_load))
    app.add_handler(CommandHandler("shell",     handle_shell))
    app.add_handler(CommandHandler("memory",    handle_memory))
    app.add_handler(CallbackQueryHandler(handle_callback, pattern=r"^jarvis:"))
    app.add_handler(MessageHandler(filters.Document.ALL,             handle_document))
    app.add_handler(MessageHandler(filters.PHOTO,                    handle_analysis))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  handle_analysis))
