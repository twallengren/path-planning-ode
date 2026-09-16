import katex from 'katex';
import 'katex/dist/katex.min.css';
import './derivation.css';

for (const node of document.querySelectorAll<HTMLElement>('[data-tex]')) {
  katex.render(node.dataset.tex ?? '', node, { displayMode: true, throwOnError: false });
}
