// A request owner suppresses late results even when its transport ignores abort.
export function createRequestOwner() {
  let current;
  function cancel() {
    current?.abort();
    current = undefined;
  }
  return {
    cancel,
    async run(request, accept, reject) {
      cancel();
      const controller = new AbortController();
      current = controller;
      try {
        const data = await request(controller.signal);
        if (current === controller) accept(data);
      } catch (error) {
        if (current === controller) reject(error);
      }
    }
  };
}

// Schedule after settlement, not at fixed intervals: polls never overlap.
export function startPolling(refresh, interval, clock = globalThis) {
  let active = true, timer;
  async function tick() {
    await refresh();
    if (active) timer = clock.setTimeout(tick, interval);
  }
  tick();
  return () => {
    active = false;
    clock.clearTimeout(timer);
  };
}
