"""Select immutable defense sessions and build their libraries off the UI thread."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

from classbot.asr.funasr_streaming import FunAsrStreamingRecognizer
from classbot.audio.loopback import LoopbackCapture, list_loopback_devices
from classbot.audio.segmenter import SpeechSegmenter
from classbot.config import Settings
from classbot.defense_detector import DefenseQuestionDetector
from classbot.knowledge import KnowledgeIndex, LocalEmbedder
from classbot.knowledge.cache_manager import (
    CacheBusyError,
    CacheLimitError,
    CacheManager,
    CacheSafetyError,
)
from classbot.knowledge.index import IndexBuildCancelled, model_identity
from classbot.llm.deepseek_client import DeepSeekClient
from classbot.pipeline import LivePipeline
from classbot.scenarios import ScenarioProfile, ScenarioStore
from classbot.ui.scenario_dialog import ScenarioDialog

_LOGGER = logging.getLogger(__name__)
_MAX_CACHED_LIBRARIES = 3


def _make_defense_segmenter() -> SpeechSegmenter:
    # A brief thinking pause or comma must not split a compound defense
    # question. The classroom factory retains its existing 320ms timing.
    return SpeechSegmenter(end_silence_ms=600)


class WorkspaceController:
    def __init__(self, settings: Settings, window: Any, runtime: Any) -> None:
        self.settings = settings
        self.window = window
        self.runtime = runtime
        self.profiles: tuple[ScenarioProfile, ...] = ()
        self.indices: OrderedDict[str, KnowledgeIndex] = OrderedDict()
        self.reports: OrderedDict[str, KnowledgeIndex] = OrderedDict()
        self._library_lru: OrderedDict[str, None] = OrderedDict()
        self._embedders: OrderedDict[str, LocalEmbedder] = OrderedDict()
        self._build_cancel: threading.Event | None = None
        self._building_profile: str | None = None
        self.store = ScenarioStore(settings.cache_dir / "scenarios.json")
        self._tasks: set[asyncio.Task[Any]] = set()
        self._busy = False
        self._closed = False
        self._generation = 0
        self._cache_dialog = None
        self.runtime.prepare_session = self.prepare_session

    def initialize(self) -> None:
        load_error = None
        try:
            self.profiles = self.store.load()
        except ValueError as error:
            load_error = str(error)
        selected = (
            "__classroom__"
            if self.settings.startup_mode == "classroom"
            else (self.profiles[0].scenario_id if self.profiles else "__defense__")
        )
        self.window.set_scenarios(self.profiles, selected)
        self._show_index_status()
        if load_error:
            self.window.set_library_status(
                f"场景配置读取失败：{load_error}；原文件已保留。"
            )
        self.window.scenario_changed.connect(self._select_scenario)
        self.window.scenario_add_requested.connect(lambda: self._edit_scenario(False))
        self.window.scenario_edit_requested.connect(lambda: self._edit_scenario(True))
        self.window.index_requested.connect(lambda: self._schedule(self.build_index()))
        report_requested = getattr(self.window, "index_report_requested", None)
        if report_requested is not None:
            report_requested.connect(self.show_index_report)
        cache_requested = getattr(self.window, "cache_management_requested", None)
        if cache_requested is not None:
            cache_requested.connect(self.show_cache_manager)
        self.window.audio_source_changed.connect(
            lambda _source: self._schedule(self.refresh_devices())
        )
        self.window.close_requested.connect(self.close)

    @property
    def selected_profile(self) -> ScenarioProfile | None:
        return next(
            (
                profile
                for profile in self.profiles
                if profile.scenario_id == self.window.selected_scenario_id
            ),
            None,
        )

    def _schedule(self, awaitable) -> None:
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.error("workspace operation failed", exc_info=error)
            if not self._closed:
                self.window.set_library_status(f"操作失败：{error}")

    def _select_scenario(self, _scenario_id: str) -> None:
        self._generation += 1
        self._show_index_status()

    def _show_index_status(self) -> None:
        profile = self.selected_profile
        if profile is not None:
            self._touch_library(profile.scenario_id)
        if hasattr(self.window, "setWindowTitle"):
            self.window.setWindowTitle(
                "Classbot · "
                + (
                    self.settings.subject
                    if self.window.selected_scenario_id == "__classroom__"
                    else profile.name
                    if profile
                    else "线上答辩助手"
                )
            )
        if profile is None:
            self._set_report_available(False)
            self.window.set_library_status(
                "创建场景并选择资料文件夹，然后建立本地语义索引。"
            )
            return
        index = self.indices.get(profile.scenario_id)
        report_index = self.reports.get(profile.scenario_id) or index
        self._set_report_available(report_index is not None)
        report_path = getattr(report_index, "report_path", None)
        report_note = (
            f"\n报告：{report_path}"
            if report_path is not None and Path(report_path).is_file()
            else ""
        )
        if index is None:
            self.window.set_library_status(
                (
                    "未解析到可检索文本，请查看“索引详情”。"
                    if profile.scenario_id in self.reports
                    else "尚未加载索引。点击“建立/刷新索引”；首次会下载本地向量模型。"
                )
                + report_note
            )
            return
        report = index.report
        ready = report.chunks > 0
        limited = getattr(report, "limited_files", 0)
        duplicate = getattr(report, "duplicate_files", 0)
        text = (
            ("部分资料已索引：" if limited else "本地语义索引：")
            + f"{report.indexed_files} 文件 / {report.chunks} 片段；"
            f"受限 {limited} 项；跳过 {report.skipped_files} 项；重复 {duplicate} 项。"
            f"快照：{index.built_at}。资料变化后请刷新。"
        )
        if report.issues:
            text += "\n" + "\n".join(
                issue if len(issue) <= 120 else issue[:119] + "…"
                for issue in report.issues[:3]
            )
            if len(report.issues) > 3:
                text += f"\n另有 {len(report.issues) - 3} 条诊断，请查看“索引详情”。"
        text += report_note
        self.window.set_library_status(text, ready=ready)

    def _touch_library(self, scenario_id: str) -> None:
        for key in (*self.indices, *self.reports):
            self._library_lru.setdefault(key, None)
        if scenario_id in self._library_lru:
            self._library_lru.move_to_end(scenario_id)
        protected = {self.window.selected_scenario_id, self._building_profile}
        while len(self._library_lru) > _MAX_CACHED_LIBRARIES:
            victim = next((key for key in self._library_lru if key not in protected), None)
            if victim is None:
                break
            self._library_lru.pop(victim)
            self.indices.pop(victim, None)
            self.reports.pop(victim, None)
        if not self._busy:
            retained = [getattr(index, "_embedder", None) for index in (
                *self.indices.values(), *self.reports.values(),
            )]
            if any(item is not None for item in retained):
                for key, embedder in tuple(self._embedders.items()):
                    if not any(embedder is item for item in retained):
                        self._embedders.pop(key)
            else:
                while len(self._embedders) > 1:
                    self._embedders.popitem(last=False)

    def _shared_embedder(self, model_name: str, cache_dir: Path) -> LocalEmbedder:
        identity = model_identity(model_name)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
        embedder = self._embedders.get(key)
        if embedder is None:
            embedder = LocalEmbedder(
                model_name, cache_dir=cache_dir / "models",
            )
            embedder.index_model_identity = identity
            self._embedders[key] = embedder
        self._embedders.move_to_end(key)
        return embedder

    def _set_report_available(self, available: bool) -> None:
        setter = getattr(self.window, "set_report_available", None)
        if setter is not None:
            setter(available)

    def show_index_report(self) -> None:
        profile = self.selected_profile
        if profile is None:
            return
        index = self.reports.get(profile.scenario_id) or self.indices.get(
            profile.scenario_id
        )
        if index is None:
            return
        dialog = QDialog(self.window)
        dialog.setWindowTitle(f"索引详情 · {profile.name}")
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(dialog)
        text.setObjectName("indexReportText")
        text.setReadOnly(True)
        text.setPlainText(index.report_text())
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def show_cache_manager(self) -> None:
        """Release idle in-memory snapshots before offering disk cleanup."""
        from classbot.main import RuntimeDependencies
        from classbot.ui.cache_dialog import CacheDialog

        if self._busy or self._closed or self.runtime.current_runtime is not None:
            return
        if self._cache_dialog is not None:
            self._cache_dialog.raise_()
            self._cache_dialog.activateWindow()
            return
        selected = self.window.selected_scenario_id
        # A stopped runtime's factories can retain the preceding index through
        # their closures. prepare_session will recreate them on the next start.
        self.runtime.dependencies = RuntimeDependencies()
        for key in tuple(set(self.indices) | set(self.reports)):
            if key != selected:
                self.indices.pop(key, None)
                self.reports.pop(key, None)
                self._library_lru.pop(key, None)
        protected = {
            index.cache_key
            for index in (*self.indices.values(), *self.reports.values())
            if getattr(index, "cache_key", None)
        }
        manager = CacheManager(
            self.settings.cache_dir,
            source_roots=tuple(Path(profile.library_path) for profile in self.profiles),
        )
        dialog = CacheDialog(
            manager, profiles=self.profiles, protected_keys=protected,
            parent=self.window,
        )
        dialog.setModal(True)
        self._cache_dialog = dialog
        dialog.cache_changed.connect(self._cache_changed)
        dialog.finished.connect(lambda _result: self._cache_manager_closed(dialog))
        dialog.show()

    def _cache_changed(self, result: object) -> None:
        if not isinstance(result, dict) or result.get("operation") != "clear_library":
            return
        key = result.get("key")
        for scenario_id, index in tuple(self.indices.items()):
            if getattr(index, "cache_key", None) == key:
                self.indices.pop(scenario_id, None)
                self.reports.pop(scenario_id, None)
                self._library_lru.pop(scenario_id, None)
        self._show_index_status()

    def _cache_manager_closed(self, dialog) -> None:
        if self._cache_dialog is dialog:
            self._cache_dialog = None
        dialog.deleteLater()

    def _edit_scenario(self, editing: bool) -> None:
        if self._busy or self.runtime.current_runtime is not None:
            return
        previous = self.selected_profile if editing else None
        if editing and previous is None:
            return
        dialog = ScenarioDialog(self.window, profile=previous)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            profile = ScenarioProfile(
                previous.scenario_id if previous else uuid4().hex,
                **dialog.values(),
            )
            profiles = tuple(
                item for item in self.profiles if item.scenario_id != profile.scenario_id
            ) + (profile,)
            self.store.save(profiles)
        except (OSError, TypeError, ValueError, CacheBusyError, CacheLimitError, CacheSafetyError) as error:
            QMessageBox.warning(self.window, "场景保存失败", str(error))
            return
        self.profiles = profiles
        self.indices.pop(profile.scenario_id, None)
        self.reports.pop(profile.scenario_id, None)
        self._library_lru.pop(profile.scenario_id, None)
        self._generation += 1
        self.window.set_scenarios(profiles, profile.scenario_id)
        self._show_index_status()
        if self.store.last_backup_path is not None:
            self.window.set_library_status(
                f"场景已保存；原配置已备份至：{self.store.last_backup_path}。请建立/刷新索引。",
            )
        self._schedule(self.refresh_devices())

    async def refresh_devices(self) -> None:
        if self._closed or self.window.audio_source != "loopback":
            return
        generation = self._generation

        def still_selected() -> bool:
            return (
                not self._closed
                and generation == self._generation
                and self.window.audio_source == "loopback"
                and self.runtime.current_runtime is None
            )

        try:
            devices = await asyncio.to_thread(list_loopback_devices)
            if still_selected():
                self.window.set_audio_devices(devices, self.window.audio_device)
        except Exception as error:  # noqa: BLE001 - native audio backends surface diverse errors.
            if still_selected():
                self.window.set_library_status(
                    f"播放设备枚举失败：{error}；检查系统播放设备。"
                )

    async def build_index(self) -> None:
        if (self._busy or self._closed or self._cache_dialog is not None
                or self.runtime.current_runtime is not None):
            return
        profile = self.selected_profile
        if profile is None:
            return
        self._busy = True
        cancellation = threading.Event()
        self._build_cancel = cancellation
        self._building_profile = profile.scenario_id
        self.window.set_configuration_busy(True)
        generation = self._generation
        loop = asyncio.get_running_loop()
        settings = self.settings
        progress_lock = threading.Lock()
        pending_message: str | None = None
        progress_scheduled = False

        def show_progress() -> None:
            nonlocal pending_message, progress_scheduled
            with progress_lock:
                message = pending_message
                pending_message = None
                progress_scheduled = False
            if (message is not None and not self._closed and not cancellation.is_set()
                    and generation == self._generation and self._build_cancel is cancellation):
                self.window.set_library_status(message)

        def progress(message: str) -> None:
            nonlocal pending_message, progress_scheduled
            if self._closed or cancellation.is_set():
                return
            with progress_lock:
                pending_message = message
                if progress_scheduled:
                    return
                progress_scheduled = True
            try:
                loop.call_soon_threadsafe(show_progress)
            except RuntimeError:
                cancellation.set()
                with progress_lock:
                    pending_message = None
                    progress_scheduled = False

        def build() -> KnowledgeIndex:
            return KnowledgeIndex.build(
                Path(profile.library_path), settings.cache_dir,
                model_name=settings.embedding_model,
                embedder=self._shared_embedder(settings.embedding_model, settings.cache_dir),
                progress=progress, mode=profile.index_mode, cancelled=cancellation.is_set,
            )

        try:
            operation = asyncio.to_thread(build)
            create_task = getattr(loop, "create_task", None)
            if create_task is None:
                # Lightweight inline adapters may complete without an event loop.
                index = await operation
            else:
                worker = create_task(operation)
                try:
                    index = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancellation.set()
                    await self._settle_canceled_build(worker)
                    raise
            if self._closed or cancellation.is_set() or generation != self._generation:
                return
            self.reports[profile.scenario_id] = index
            self._set_report_available(True)
            if index.report.chunks == 0:
                raise ValueError(
                    "未解析到可检索文本。检查文件格式；扫描PDF需先OCR，旧.doc需另存为.docx。"
                )
            self.indices[profile.scenario_id] = index
            self._show_index_status()
        except Exception as error:  # noqa: BLE001 - report third-party parser/model failures in the UI.
            if not self._closed and generation == self._generation:
                previous = self.indices.get(profile.scenario_id)
                self.window.set_library_status(
                    f"索引失败：{error}"
                    + ("；已保留上次索引快照。" if previous else ""),
                    ready=previous is not None,
                )
        finally:
            self._busy = False
            self._build_cancel = None
            self._building_profile = None
            self._touch_library(self.window.selected_scenario_id)
            if not self._closed:
                self.window.set_configuration_busy(False)

    @staticmethod
    async def _settle_canceled_build(worker: asyncio.Task[Any]) -> None:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - observe the result below while preserving cancellation.
                break
        if not worker.cancelled():
            error = worker.exception()
            if error is not None and not isinstance(error, IndexBuildCancelled):
                _LOGGER.error("canceled index worker failed", exc_info=error)

    async def prepare_session(self) -> None:
        from classbot.main import RuntimeDependencies

        if self._busy or self._closed or self._cache_dialog is not None:
            raise ValueError("请等待资料索引结束")
        if self.window.selected_scenario_id == "__classroom__":
            self.runtime.settings = replace(self.settings, audio_source="microphone")
            self.runtime.dependencies = RuntimeDependencies()
            return
        profile = self.selected_profile
        index = self.indices.get(profile.scenario_id) if profile else None
        if profile is None or index is None:
            raise ValueError("请选择答辩场景并建立资料索引")
        self._touch_library(profile.scenario_id)
        if await asyncio.to_thread(index.is_stale):
            self.window.set_library_status("资料文件已更新，请刷新索引后再开始。")
            raise ValueError("资料文件已更新，需要刷新索引")
        settings = replace(self.settings, subject=profile.name, audio_source="loopback")
        device = self.window.audio_device
        self.runtime.settings = settings
        self.runtime.dependencies = RuntimeDependencies(
            capture_factory=lambda: LoopbackCapture(device),
            segmenter_factory=_make_defense_segmenter,
            recognizer_factory=lambda device: FunAsrStreamingRecognizer(device=device),
            detector_factory=lambda _subject: DefenseQuestionDetector(profile.name),
            client_factory=lambda _settings: DeepSeekClient(
                settings, scenario=profile, knowledge_index=index
            ),
            pipeline_factory=lambda *args: LivePipeline(
                *args, scenario_name=profile.name
            ),
        )

    def close(self) -> None:
        self._closed = True
        self._generation += 1
        if self._build_cancel is not None:
            self._build_cancel.set()
        for task in tuple(self._tasks):
            task.cancel()
        if self._cache_dialog is not None:
            self._cache_dialog.close()

    async def aclose(self) -> None:
        """Drain cooperative builders before the host closes the event loop."""
        self.close()
        current = asyncio.current_task()
        pending = tuple(task for task in self._tasks if task is not current)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        dialog = self._cache_dialog
        if dialog is not None:
            while dialog.busy:
                await asyncio.sleep(0.01)
            dialog.close()
