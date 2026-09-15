import katex from 'katex';
import 'katex/dist/katex.min.css';
import './style.css';
import './derivation.css';

// The article is static HTML with readable fallbacks. Only typesetting needs JS.
for (const element of document.querySelectorAll<HTMLElement>('[data-tex]')) {
  katex.render(element.dataset.tex!, element, {
    displayMode: true,
    throwOnError: true,
    output: 'htmlAndMathml',
  });
}
