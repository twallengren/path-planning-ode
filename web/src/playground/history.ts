import { cloneSnapshot, type PlaygroundSnapshot } from './types';

export class GestureHistory {
  private undoStack: PlaygroundSnapshot[] = [];
  private redoStack: PlaygroundSnapshot[] = [];
  private before: PlaygroundSnapshot | null = null;

  constructor(private readonly limit = 50) {}

  begin(snapshot: PlaygroundSnapshot) {
    if (!this.before) this.before = cloneSnapshot(snapshot);
  }

  commit(changed: boolean) {
    if (changed && this.before) {
      this.undoStack.push(this.before);
      if (this.undoStack.length > this.limit) this.undoStack.shift();
      this.redoStack = [];
    }
    this.before = null;
  }

  cancel() {
    const snapshot = this.before;
    this.before = null;
    return snapshot ? cloneSnapshot(snapshot) : null;
  }

  undo(current: PlaygroundSnapshot) {
    const previous = this.undoStack.pop();
    if (!previous) return null;
    this.redoStack.push(cloneSnapshot(current));
    return cloneSnapshot(previous);
  }

  redo(current: PlaygroundSnapshot) {
    const next = this.redoStack.pop();
    if (!next) return null;
    this.undoStack.push(cloneSnapshot(current));
    return cloneSnapshot(next);
  }

  get canUndo() {
    return this.undoStack.length > 0;
  }

  get canRedo() {
    return this.redoStack.length > 0;
  }

  clear() {
    this.undoStack = [];
    this.redoStack = [];
    this.before = null;
  }
}
