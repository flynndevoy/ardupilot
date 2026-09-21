# guidance.py
from __future__ import annotations
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, Awaitable
import numpy as np


@dataclass
class Submission:
    # Only the precomputed payload is needed now
    daidalus_input: Dict[str, Any]


class Guidance_Thread:
    """
    Dedicated thread + asyncio loop that:
      - consumes prebuilt `daidalus_input` payloads from the outer loop
      - awaits send_fn(daidalus_input) asynchronously
      - publishes results into the provided `datastore`:
          * 'daidalus_free'
          * 'daidalus_out'
          * 'guidance_consumed'
    """

    def __init__(
        self,
        *,
        send_fn: Callable[[dict], Awaitable[dict]],
        tracker: Any = None,           # kept for backward compatibility; unused
        datastore: Dict[str, Any],
    ):
        self.send_fn = send_fn
        self.datastore = datastore

        # thread / loop
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event() 

        # async tasks
        self._consumer_task: Optional[asyncio.Task] = None

        # comms
        self._queue: asyncio.Queue[Tuple[Submission, float]] = asyncio.Queue()

        # lock for shared datastore writes
        self._lock = threading.Lock()

    # ---------------------------
    # Public API
    # ---------------------------
    def start(self, timeout=10.0):
        print("Starting up Guidance Thread...")
        if self._thread and self._thread.is_alive():
            if not self._ready_evt.is_set():
                self._ready_evt.wait(timeout)
            print("Started!")
            return
        self._stop_evt.clear()
        self._ready_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, name="GuidanceThread", daemon=True)
        self._thread.start()

        # (optional) mark free at startup
        with self._lock:
            self.datastore["daidalus_free"] = True
            self.datastore["guidance_consumed"] = True

        if not self._ready_evt.wait(timeout):
            raise RuntimeError("Guidance thread failed to start within timeout.")
        print("Started!")

    def stop(self, timeout: Optional[float] = 2.0):
        self._stop_evt.set()
        if self._loop:
            fut = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=timeout)

    def submit(self, daidalus_input: Dict[str, Any]):
        """Non-blocking. Safe to call from your main/outer loop."""
        if not self._loop:
            raise RuntimeError("Guidance not started. Call start() first.")
        sub = Submission(daidalus_input=daidalus_input)
        asyncio.run_coroutine_threadsafe(self._queue.put((sub, time.time())), self._loop)

    # ---------------------------
    # Internal: thread + loop
    # ---------------------------
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._consumer_task = self._loop.create_task(self._consumer())
        self._ready_evt.set()
        try:
            self._loop.run_until_complete(self._consumer_task)
        finally:
            pending = asyncio.all_tasks(loop=self._loop)
            for t in pending:
                t.cancel()
            try:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()
            self._loop = None

    async def _shutdown_async(self):
        if self._consumer_task:
            self._consumer_task.cancel()
        # drain queue so the thread can exit
        try:
            while True:
                self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def _consumer(self):
        while not self._stop_evt.is_set():
            try:
                sub, _ = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            await self._handle_submission(sub)

    # ---------------------------
    # Core handling
    # ---------------------------
    async def _handle_submission(self, sub: Submission):
        # mark busy
        with self._lock:
            self.datastore["daidalus_free"] = False

        try:
            loop = asyncio.get_running_loop()
            # run blocking send in a worker thread; returns dict when ready
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.send_fn(sub.daidalus_input)),
                timeout=10.0
            )
        except Exception as e:
            err_payload = {
                'horizontal_guidance': [], 'altitude_guidance': [],
                'horizontal_speed_guidance': [], 'vertical_speed_guidance': [],
                'intruders': [], 'suggested_hdir_deg': np.nan,
                'suggested_hspd_knot': np.nan, 'suggested_vspd_fpm': np.nan,
                'suggested_alt_ft': np.nan, 'timestamp': time.time(),
                'error': str(e),
            }
            with self._lock:
                self.datastore['daidalus_out'] = err_payload
                self.datastore["daidalus_free"] = True
                self.datastore["guidance_consumed"] = False
            return

        # success: publish then mark free
        # get the current time of the ownship when the response if recieved
        ts = self.datastore['track']['processed_ownship_state']['time']
        shdir = response.get('suggested_hdir_deg', np.nan)
        shspd_knot = response.get('suggested_hspd_knot', np.nan)
        svspd_fpm = response.get('suggested_vspd_fpm', np.nan)
        salt_ft = response.get('suggested_alt_ft', np.nan)
        payload = {
            'horizontal_guidance': response.get('horizontal_guidance', []),
            'altitude_guidance': response.get('altitude_guidance', []),
            'horizontal_speed_guidance': response.get('horizontal_speed_guidance', []),
            'vertical_speed_guidance': response.get('vertical_speed_guidance', []),
            'intruders': response.get('intruders', []),
            'suggested_hdir_deg': float(shdir) if np.isfinite(shdir) else np.nan,
            'suggested_hspd_knot': float(shspd_knot) if np.isfinite(shspd_knot) else np.nan,
            'suggested_vspd_fpm': float(svspd_fpm) if np.isfinite(svspd_fpm) else np.nan,
            'suggested_alt_ft': float(salt_ft) if np.isfinite(salt_ft) else np.nan,
            'timestamp': ts,
        }
        with self._lock:
            self.datastore['daidalus_out'] = payload
            self.datastore["daidalus_free"] = True
            self.datastore["guidance_consumed"] = False
