import katex from 'katex';
const eq = (tex: string) => katex.renderToString(tex, { throwOnError: false, displayMode: true });
export const content = /* HTML */ ` <header class="masthead">
    <a class="wordmark" href="#">PATH / PLANNING <span>an exploration</span></a>
    <nav aria-label="Main">
      <a href="./derivation.html">Derivation</a><a href="./terrain.html">Terrain lab</a
      ><a href="#playground">Playground</a
      ><a href="https://github.com/twallengren/path-planning-ode">Source ↗</a>
    </nav>
  </header>
  <main>
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow"><span class="small-line"></span>A COMPUTATIONAL CASE STUDY · 01</p>
        <h1>The shape<br />of a <em>path.</em></h1>
        <p class="standfirst">
          The shortest route isn’t always the cheapest.<br class="desktop" />When is a detour worth
          it?
        </p>
        <p class="hero-detail">
          Give every place a cost. Compare ways through.<br />An interactive exploration of weighted
          distance.
        </p>
        <a class="primary-link" href="#playground">Explore the landscape <span>↓</span></a>
      </div>
      <div
        class="hero-art"
        aria-label="Illustration of three paths bending around regions of higher cost"
        role="img"
      >
        <svg viewBox="0 0 560 480" aria-hidden="true">
          <defs>
            <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
              <circle cx="1" cy="1" r="1" fill="#536858" opacity=".19" />
            </pattern>
            <radialGradient id="bump">
              <stop stop-color="#dc9c6b" stop-opacity=".38" />
              <stop offset="1" stop-color="#efd8b5" stop-opacity="0" />
            </radialGradient>
          </defs>
          <rect x="18" y="5" width="520" height="450" fill="url(#grid)" />
          <g transform="translate(289 225) rotate(-24)">
            <ellipse rx="155" ry="140" fill="url(#bump)" />
            <g stroke="#b98a5b" stroke-width=".8" fill="none" opacity=".4">
              <ellipse rx="120" ry="110" />
              <ellipse rx="94" ry="85" />
              <ellipse rx="68" ry="61" />
              <ellipse rx="40" ry="36" />
            </g>
          </g>
          <g transform="translate(154 122)">
            <circle r="84" fill="url(#bump)" />
            <g stroke="#b98a5b" stroke-width=".8" fill="none" opacity=".3">
              <circle r="55" />
              <circle r="32" />
            </g>
          </g>
          <path
            d="M62 399 C139 333 145 211 219 161 S369 165 481 63"
            stroke="#17695e"
            fill="none"
            stroke-width="3"
          />
          <path
            d="M62 399 C174 398 374 407 422 298 S431 136 481 63"
            stroke="#bd592f"
            fill="none"
            stroke-width="2.5"
            stroke-dasharray="8 5"
          />
          <path
            d="M62 399 C20 212 51 79 218 54 S409 61 481 63"
            stroke="#7566b0"
            fill="none"
            stroke-width="2.5"
            stroke-dasharray="3 5"
          />
          <g fill="#273d33">
            <circle cx="62" cy="399" r="6" />
            <circle cx="481" cy="63" r="6" />
          </g>
          <g font-family="monospace" font-size="10" fill="#536858">
            <text x="77" y="422">START</text>
            <text x="439" y="43">DESTINATION</text>
            <text x="266" y="271">HIGHER COST</text>
            <text x="356" y="431">THREE INITIAL GUESSES</text>
          </g>
          <g stroke="#96623d" stroke-width="1.5">
            <path d="M284 220 l10 10 m0 -10 l-10 10" />
            <path d="M150 118 l8 8 m0 -8 l-8 8" />
          </g>
        </svg>
        <p class="figure-caption">
          <span>FIG. 01</span>One landscape. More than one way through.<span
            class="illustration-note"
            >Illustration</span
          >
        </p>
      </div>
    </section>
    <section id="idea" class="intro-section">
      <div>
        <p class="eyebrow">01 / THE IDEA</p>
        <h2>Make the world a<br />little more expensive.</h2>
      </div>
      <div class="prose">
        <p>
          A straight line is a good start. But place a soft hill of cost around an obstacle, and a
          different route may become worthwhile. The path trades extra distance against the cost of
          the places it passes through.
        </p>
        <p>
          Imagine paying a toll for every small step: the local cost times the distance traveled.
          Add those tolls along the route. <em>Which way has the smallest total?</em>
        </p>
        <div class="equation">${eq("J[q]=\\int_0^1 c(q(t))\\,\\lVert q'(t)\\rVert\\,dt")}</div>
        <p class="caption">
          Total cost = local cost × distance, added up along the route. Walking the same route
          faster doesn’t change its cost. The endpoints stay fixed.
        </p>
      </div>
    </section>
    <section id="playground" class="lab-section">
      <div class="section-heading">
        <div>
          <p class="eyebrow">02 / THE EXPERIMENT</p>
          <h2>A path you can change.</h2>
        </div>
        <p>
          Move a hill. Change its strength or width.<br />Compare distance traveled with total cost.
        </p>
      </div>
      <div class="experiment-tabs" role="group" aria-label="Experiments">
        <button data-preset="central"><span>01</span>Worth the detour?</button
        ><button data-preset="asymmetric" class="active"><span>02</span>Which side?</button
        ><button data-preset="passage"><span>03</span>A narrow passage</button
        ><button data-preset="challenge"><span>04</span>Strong hills</button>
      </div>
      <div class="lab">
        <div class="visual-panel">
          <div class="plot-top">
            <span class="eyebrow">COST LANDSCAPE</span
            ><span id="iteration-label" class="mono">ITERATION 00</span>
          </div>
          <canvas
            id="landscape"
            aria-label="Interactive path landscape. Use the adjacent numeric controls to edit endpoints and obstacles."
            role="img"
          ></canvas>
          <div class="legend">
            <span><i style="background:#17695e"></i>Direct</span
            ><span><i class="dashed" style="color:#bd592f"></i>Right arc</span
            ><span><i class="dotted" style="color:#7566b0"></i>Left arc</span
            ><span class="cost-key">Low <i></i> High · relative scale</span>
          </div>
          <div class="transport">
            <button id="play" class="button primary" disabled>▶ Run solver</button
            ><button id="step" class="button" disabled>Step →</button
            ><button id="reset" class="button" disabled>↺ Reset</button
            ><span id="playback-label" class="caption">Drag points to reshape the scene</span>
          </div>
          <div class="timeline">
            <label for="timeline">Iteration</label
            ><input
              id="timeline"
              type="range"
              min="0"
              max="0"
              value="0"
              aria-label="Iteration history"
            /><output id="timeline-value">0 / 0</output>
          </div>
          <div class="runtime-row">
            <span id="runtime" role="status" aria-live="polite">Preparing an example…</span
            ><button id="retry" hidden class="text-button">Retry loading</button>
          </div>
        </div>
        <aside class="controls" aria-label="Experiment controls">
          <div class="control-section">
            <div class="control-heading">
              <h3>Your scene</h3>
              <button id="empty" class="text-button">Clear field</button>
            </div>
            <p id="experiment-note" class="caption">
              Try routes on either side of the hills. Which one is cheapest, and is it also the
              shortest?
            </p>
            <div class="endpoint-grid">
              <label
                >Start x<input id="start-x" type="number" min="-100" max="100" step="0.5" /></label
              ><label
                >Start y<input id="start-y" type="number" min="-100" max="100" step="0.5" /></label
              ><label>End x<input id="end-x" type="number" min="-100" max="100" step="0.5" /></label
              ><label
                >End y<input id="end-y" type="number" min="-100" max="100" step="0.5"
              /></label>
            </div>
          </div>
          <div class="control-section">
            <div class="control-heading">
              <h3>Obstacles <span id="obstacle-count"></span></h3>
              <button id="add" class="text-button">+ Add</button>
            </div>
            <label class="sr-only" for="obstacle-select">Selected obstacle</label
            ><select id="obstacle-select"></select>
            <div id="obstacle-editor">
              <div class="endpoint-grid two">
                <label
                  >x<input id="obstacle-x" type="number" min="-100" max="100" step="0.25" /></label
                ><label
                  >y<input id="obstacle-y" type="number" min="-100" max="100" step="0.25"
                /></label>
              </div>
              <label class="slider-label" for="weight"
                >Strength <output id="weight-value"></output></label
              ><input id="weight" type="range" min="0" max="100" step="0.5" /><label
                class="slider-label"
                for="width"
                >Width <output id="width-value"></output></label
              ><input id="width" type="range" min="0.1" max="10" step="0.1" /><button
                id="remove"
                class="text-button muted"
              >
                Remove selected obstacle
              </button>
            </div>
          </div>
          <details class="display-options">
            <summary>Solver settings</summary>
            <label class="field-label" for="mode">Newton method</label
            ><select id="mode">
              <option value="damped">Damped · backtracking</option>
              <option value="undamped">Undamped · full steps</option></select
            ><label class="slider-label" for="resolution"
              >Interior points <output id="resolution-value">30</output></label
            ><input id="resolution" type="range" min="1" max="100" step="1" value="30" />
            <fieldset class="guess-options">
              <legend>Initial guesses</legend>
              <label><input type="checkbox" data-guess="straight" checked /> Direct</label
              ><label><input type="checkbox" data-guess="bend-x" checked /> Right arc</label
              ><label><input type="checkbox" data-guess="bend-y" checked /> Left arc</label>
            </fieldset>
          </details>
          <details class="display-options">
            <summary>Display & export</summary>
            <label><input id="heatmap" type="checkbox" checked /> Cost heatmap</label
            ><label><input id="contours" type="checkbox" checked /> Contours</label
            ><label><input id="samples" type="checkbox" /> Sample points</label
            ><label for="rover-path">Rover path</label
            ><select id="rover-path"></select
            ><button id="rover" class="button">Play rover</button>
            <p class="caption">Constant-distance playback, not simulated robot dynamics.</p>
            <div class="export-actions">
              <button id="share" class="text-button">Copy scene link ↗</button
              ><button id="export" class="text-button">Export JSON ↓</button
              ><button id="import" class="text-button">Import JSON ↑</button
              ><button id="png" class="text-button">Save figure ↓</button>
            </div>
            <input id="file" type="file" accept="application/json,.json" hidden />
          </details>
        </aside>
        <div class="diagnostics">
          <div class="diagnostic-heading">
            <h3>What does each route cost?</h3>
            <span id="view-label" class="caption">Live Python solver</span>
          </div>
          <p id="cost-comparison" class="cost-comparison" aria-live="polite">
            Computing route costs…
          </p>
          <div id="metrics" class="metrics"></div>
          <div class="charts">
            <div>
              <h4>Total cost <span>of the displayed route · by iteration</span></h4>
              <canvas
                id="cost-chart"
                aria-label="Total weighted distance over iterations"
                role="img"
              ></canvas>
            </div>
            <details id="numerical-chart">
              <summary>Solver convergence</summary>
              <h4>ODE residual <span>log scale · lower is closer to a root</span></h4>
              <canvas
                id="residual-chart"
                aria-label="Residual norm over iterations"
                role="img"
              ></canvas>
            </details>
          </div>
          <p class="caption diagnostic-note">
            Route labels identify their starting shapes. “Lowest cost shown” compares these
            candidates, not every possible route. Converged means the numerical equations are
            satisfied, not that a minimum is proven. Cost can rise during a solve.
          </p>
        </div>
      </div>
      <p id="message" class="message" role="status" aria-live="polite"></p>
    </section>
    <section class="reading-section">
      <div>
        <p class="eyebrow">03 / THINGS TO NOTICE</p>
        <h2>A longer way.<br />A smaller bill.</h2>
      </div>
      <div class="observations">
        <article>
          <span class="observation-number">01</span>
          <h3>A hill isn’t a wall.</h3>
          <p>
            Gaussian bumps make a region expensive, not forbidden. Raise the strength or widen the
            hill and compare the routes. A path can still cross an obstacle.
          </p>
          <button class="text-button" data-preset="central">Try a single obstacle →</button>
        </article>
        <article>
          <span class="observation-number">02</span>
          <h3>Width and strength do different things.</h3>
          <p>
            Strength changes the price of crossing a hill. Width changes how far you must go to
            avoid it. Adjust them separately: does crossing or detouring become more attractive?
          </p>
          <button class="text-button" data-preset="central">Change the hill →</button>
        </article>
        <article>
          <span class="observation-number">03</span>
          <h3>There can be more than one answer.</h3>
          <p>
            Routes on opposite sides can both settle down, with very different costs. The starting
            curve affects what the solver finds. Compare candidates rather than trusting the first
            answer.
          </p>
          <button class="text-button" data-preset="asymmetric">Compare both sides →</button>
        </article>
      </div>
    </section>
    <section id="mathematics" class="math-section">
      <p class="eyebrow">04 / UNDER THE SURFACE</p>
      <h2>From a cost to a curve.</h2>
      <p class="math-intro">
        The geometry is the invitation. Here is the mathematics underneath.
        <a href="./derivation.html" class="derivation-link"
          >Read the full, step-by-step derivation →</a
        >
      </p>
      <details>
        <summary><span>01</span>The cost landscape</summary>
        <div class="detail-body">
          <p>For obstacle centres oᵢ, nonnegative strengths wᵢ, and positive widths sᵢ, define:</p>
          ${eq('c(q)=1+\\sum_i w_i\\exp\\left(-\\frac{\\lVert q-o_i\\rVert^2}{s_i^2}\\right)')}
          <p>
            The baseline is one. Width is the radial distance at which an isolated bump has decayed
            to 1/e of its peak. These contours show cost levels, not collision boundaries.
          </p>
        </div>
      </details>
      <details>
        <summary><span>02</span>How we find candidate routes</summary>
        <div class="detail-body">
          <p>
            Weighted distance depends on the route, not how quickly you trace it. To obtain a
            definite parameterization, we use an equivalent energy with cost squared. Minimizing it
            over paths and their parameterizations gives the same minimizing geometric routes.
          </p>
          ${eq("E[q]=\\int_0^1 c(q)^2\\,\\lVert q'\\rVert^2\\,dt")}
          <p>Euler–Lagrange gives the differential equation we solve:</p>
          ${eq("q''=\\frac{\\lVert q'\\rVert^2\\nabla c-2q'(\\nabla c\\cdot q')}{c}")}
          <p>
            We sample a curve, hold its endpoints fixed, and use damped Newton steps to reduce the
            errors in this equation. That finds stationary candidates, which we then compare using
            their actual weighted distance.
          </p>
        </div>
      </details>
      <details>
        <summary><span>03</span>What the numbers can tell you</summary>
        <div class="detail-body">
          <p>
            Total cost integrates the Gaussian field along every straight segment of the displayed
            path. Distance counts only geometric length. Their difference is the extra cost paid for
            passing through hills.
          </p>
          <p>
            The residual measures how well the sampled curve satisfies the ODE. A small residual is
            not a certificate of the cheapest possible route. Increase the resolution to check
            stability, especially near narrow hills. Soft costs do not enforce collision avoidance.
          </p>
          <p>
            The same Python implementation runs here and locally. The
            <a href="./derivation.html">full derivation</a> connects the objective,
            parameterization, differential equations, and numerical solver step by step.
          </p>
        </div>
      </details>
    </section>
    <section class="local-section">
      <div>
        <p class="eyebrow">05 / KEEP EXPLORING</p>
        <h2>Take the experiment<br />with you.</h2>
        <p>
          The exact same Python solver powers this page and your local experiments. Export a scene
          above, then reproduce it on your machine.
        </p>
        <a class="primary-link" href="https://github.com/twallengren/path-planning-ode"
          >Get the code ↗</a
        >
      </div>
      <div class="code-card">
        <div><span class="code-dot"></span>YOUR TERMINAL</div>
        <pre><code>git clone https://github.com/twallengren/path-planning-ode.git<br>cd path-planning-ode<br>uv sync --extra plot<br>uv run python examples/explore.py scene.json</code></pre>
        <p>Python · NumPy · a little calculus</p>
      </div>
    </section>
  </main>
  <footer>
    <a class="wordmark" href="#">PATH / PLANNING</a>
    <p>Every step has a cost. Built for curiosity.</p>
    <a href="https://github.com/twallengren/path-planning-ode">View on GitHub ↗</a>
  </footer>`;
