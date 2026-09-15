import type { WorkerResult } from './types';

type Waiting = {
  resolve: (value: WorkerResult) => void;
  reject: (error: Error) => void;
};

export class PlaygroundWorkerClient {
  private worker: Worker | null = null;
  private serial = 0;
  private waiting = new Map<number, Waiting>();

  constructor(
    private readonly createWorker: () => Worker,
    private readonly onProgress: (message: string) => void,
    private readonly onFailure: (error: Error) => void,
  ) {}

  start() {
    this.stop(new Error('Worker restarted.'));
    const worker = this.createWorker();
    this.worker = worker;
    worker.onmessage = ({ data }) => {
      if (data.progress) {
        this.onProgress(String(data.progress));
        return;
      }
      const waiting = this.waiting.get(data.id);
      if (!waiting) return;
      this.waiting.delete(data.id);
      if (data.error) waiting.reject(new Error(String(data.error)));
      else waiting.resolve(data.result as WorkerResult);
    };
    worker.onerror = (event) => {
      const error = new Error(event.message || 'Browser worker failed.');
      this.stop(error);
      this.onFailure(error);
    };
  }

  request(action: string, data: Record<string, unknown> = {}) {
    if (!this.worker) return Promise.reject(new Error('Python worker is not running.'));
    const id = ++this.serial;
    return new Promise<WorkerResult>((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker!.postMessage({ id, action, ...data });
    });
  }

  stop(reason = new Error('Worker stopped.')) {
    this.worker?.terminate();
    this.worker = null;
    for (const waiting of this.waiting.values()) waiting.reject(reason);
    this.waiting.clear();
  }
}

export type ScheduledTask<T> = { revision: number; run: () => Promise<T> };

/** Runs one worker request at a time and keeps only the newest edit while busy. */
export class LatestTaskQueue<T> {
  private active = false;
  private pending: ScheduledTask<T> | null = null;

  constructor(
    private readonly accept: (result: T, revision: number) => void,
    private readonly reject: (error: Error, revision: number) => void,
  ) {}

  push(task: ScheduledTask<T>) {
    if (this.active) {
      this.pending = task;
      return;
    }
    void this.execute(task);
  }

  clear() {
    this.pending = null;
  }

  get busy() {
    return this.active;
  }

  private async execute(task: ScheduledTask<T>) {
    this.active = true;
    try {
      this.accept(await task.run(), task.revision);
    } catch (error) {
      this.reject(error instanceof Error ? error : new Error(String(error)), task.revision);
    } finally {
      this.active = false;
      const pending = this.pending;
      this.pending = null;
      if (pending) void this.execute(pending);
    }
  }
}
