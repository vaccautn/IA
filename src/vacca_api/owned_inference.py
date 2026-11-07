"""Bounded native work owns its permit, not its cancellable async caller."""

import asyncio
import threading


async def run_owned(gate, operation):
    # Caller has acquired the gate. No cancellation point exists before start.
    loop = asyncio.get_running_loop()
    completion = loop.create_future()

    def publish(value, error):
        gate._work_threads.discard(thread)
        gate.release()
        if not completion.done():
            if error is None:
                completion.set_result(value)
            else:
                completion.set_exception(error)

    def execute():
        value, error = None, None
        try:
            value = operation()
        except BaseException as exc:
            error = exc
        finally:
            try:
                loop.call_soon_threadsafe(publish, value, error)
            except RuntimeError:
                # A closed lifespan cannot serve more requests. Keep its permit
                # held rather than release from a foreign thread/new event loop.
                gate._work_threads.discard(thread)

    thread = threading.Thread(target=execute, name="vacca-bcs", daemon=True)
    gate._work_threads.add(thread)
    try:
        thread.start()
    except BaseException:
        gate._work_threads.discard(thread)
        gate.release()
        raise
    # Cancelling this waiter must not cancel completion or native ML work.
    completion.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
    return await asyncio.shield(completion)
