(() => {
  "use strict";

  const PROMPTS = [
    {
      num: "01",
      question: "Describe your ideal weekend from morning to night.",
      placeholder: "How do you choose to spend unstructured hours when nobody expects anything from you?",
    },
    {
      num: "02",
      question: "How do you usually approach a large project or tight deadline?",
      placeholder: "Do you map milestones upfront, adapt as you go, or wait for momentum?",
    },
    {
      num: "03",
      question: "When making a difficult decision, what weighs most heavily for you?",
      placeholder: "Think of a recent hard choice and how you arrived at your conclusion.",
    },
    {
      num: "04",
      question: "How do you recharge after a long, demanding week?",
      placeholder: "What reliably restores your energy, and what drains it fastest?",
    },
    {
      num: "05",
      question: "What subjects, hobbies, or conversations absorb your attention completely?",
      placeholder: "Write about the ideas, crafts, or problems you naturally lose track of time exploring.",
    },
    {
      num: "06",
      question: "How do you handle disagreement or tension with someone close to you?",
      placeholder: "What is your instinct when a conversation gets uncomfortable?",
    },
  ];

  const SAMPLES = {
    architect: [
      "I usually spend my weekends quietly working through a long-term side project or reading books on systems design, history, and philosophy. Crowded, noisy rooms drain my energy quickly, and I would rather have one focused conversation than circulate through small talk.",
      "Before starting any major project, I map out the full architecture, list each milestone, and set firm internal deadlines so nothing depends on last-minute improvisation. Having a clear logical plan frees my mind to focus on the hard technical questions.",
      "When a friend asks for advice on a difficult decision, my first instinct is to lay out the objective trade-offs, second-order consequences, and measurable criteria rather than reacting purely to how everyone feels in the moment.",
      "What excites me most is uncovering the underlying framework behind complex phenomena. Memorizing routine step-by-step procedures without understanding the theory behind them quickly loses my interest.",
    ],
    campaigner: [
      "Last weekend my friends and I spontaneously drove out toward the coast with zero itinerary, stopped at three tiny roadside cafes, and ended up talking for hours with a local ceramicist we met at a market!",
      "I get genuinely energized bouncing fresh ideas around with people. When someone I care about is going through a rough week, I want to sit down, listen with my whole heart, and help them feel truly understood and hopeful again.",
      "Honestly, rigid schedules and repetitive routines make me feel boxed in. I prefer leaving space in my days for curiosity, unexpected invitations, and creative detours that turn an ordinary afternoon into a story worth remembering.",
      "I love imagining what could be possible in the future and connecting people's personal stories, values, and passions. Even in disagreements, keeping warmth, sincerity, and mutual respect intact matters more to me than scoring a logical point.",
    ],
    logistician: [
      "My ideal weekend starts early with coffee, a walk around the neighborhood, and getting the house and errands squared away before noon. Once the practical tasks are finished, I enjoy cooking a familiar recipe, working on a tangible repair project, or catching up with family.",
      "On any work assignment I review the exact specifications first, verify the facts, and build a realistic step-by-step checklist. I do not like vague speculation when concrete numbers and proven methods are right in front of us.",
      "When making major life choices I look closely at reliability, budget, past experience, and long-term obligations. Keeping commitments and following through consistently matters far more to me than chasing untested trends.",
      "In a disagreement I prefer calm, matter-of-fact honesty. Laying out the specific facts clearly and agreeing on practical next steps resolves tension much faster than guessing at unspoken motives.",
    ],
  };

  const GROUP_ORDER = ["Analysts", "Diplomats", "Sentinels", "Explorers"];

  let currentView = "analyze";
  let inputMode = "single";
  let lastResult = null;
  let typesCache = null;

  // DOM references
  const navLinks = document.querySelectorAll(".nav-link");
  const views = {
    analyze: document.getElementById("view-analyze"),
    prompts: document.getElementById("view-prompts"),
    reference: document.getElementById("view-reference"),
  };
  const brandLink = document.getElementById("brand-link");

  // Analyze view
  const modeSingleBtn = document.getElementById("mode-single-btn");
  const modeMultiBtn = document.getElementById("mode-multi-btn");
  const singleInputWrap = document.getElementById("single-input-wrap");
  const multiInputWrap = document.getElementById("multi-input-wrap");
  const freeTextInput = document.getElementById("free-text-input");
  const entriesList = document.getElementById("entries-list");
  const addEntryBtn = document.getElementById("add-entry-btn");
  const analyzeWordCount = document.getElementById("analyze-word-count");
  const analyzeSignalFill = document.getElementById("analyze-signal-fill");
  const analyzeSignalText = document.getElementById("analyze-signal-text");
  const clearAnalyzeBtn = document.getElementById("clear-analyze-btn");
  const runAnalyzeBtn = document.getElementById("run-analyze-btn");
  const analyzeError = document.getElementById("analyze-error");

  // Prompts view
  const promptsList = document.getElementById("prompts-list");
  const promptsWordCount = document.getElementById("prompts-word-count");
  const promptsSignalFill = document.getElementById("prompts-signal-fill");
  const promptsSignalText = document.getElementById("prompts-signal-text");
  const clearPromptsBtn = document.getElementById("clear-prompts-btn");
  const runPromptsBtn = document.getElementById("run-prompts-btn");
  const promptsError = document.getElementById("prompts-error");

  // Results section
  const resultsSection = document.getElementById("results-section");
  const resWordCount = document.getElementById("res-word-count");
  const resReliability = document.getElementById("res-reliability");
  const resEngine = document.getElementById("res-engine");
  const resTypeCode = document.getElementById("res-type-code");
  const resTypeGroup = document.getElementById("res-type-group");
  const resTypeTitle = document.getElementById("res-type-title");
  const resFunctions = document.getElementById("res-functions");
  const resTypeSummary = document.getElementById("res-type-summary");
  const resAxesList = document.getElementById("res-axes-list");
  const resTop3 = document.getElementById("res-top3");
  const resComm = document.getElementById("res-comm");
  const resStrengths = document.getElementById("res-strengths");
  const resBlindspots = document.getElementById("res-blindspots");
  const resSignals = document.getElementById("res-signals");
  const resEngineNote = document.getElementById("res-engine-note");
  const copyReportBtn = document.getElementById("copy-report-btn");
  const editInputBtn = document.getElementById("edit-input-btn");

  // Reference view
  const referenceGroups = document.getElementById("reference-groups");
  const referenceDetail = document.getElementById("reference-detail");

  // Engine status & modal
  const engineDot = document.getElementById("engine-dot");
  const engineLabel = document.getElementById("engine-label");
  const openModelModalBtn = document.getElementById("open-model-modal");
  const closeModelModalBtn = document.getElementById("close-model-modal");
  const modelModal = document.getElementById("model-modal");
  const modalEngineMode = document.getElementById("modal-engine-mode");
  const modalEngineFile = document.getElementById("modal-engine-file");
  const modalMasking = document.getElementById("modal-masking");
  const uploadDropzone = document.getElementById("upload-dropzone");
  const modelFileInput = document.getElementById("model-file-input");
  const dropzoneLabel = document.getElementById("dropzone-label");
  const uploadFeedback = document.getElementById("upload-feedback");

  // Helpers
  function countWords(text) {
    const matches = String(text || "").match(/[A-Za-z]+(?:'[A-Za-z]+)?/g);
    return matches ? matches.length : 0;
  }

  function updateMeter(wordCount, countEl, fillEl, textEl) {
    countEl.textContent = `${wordCount} ${wordCount === 1 ? "word" : "words"}`;
    const pct = Math.min(100, Math.round((wordCount / 220) * 100));
    fillEl.style.width = `${pct}%`;
    fillEl.classList.remove("tier-low", "tier-moderate", "tier-high");

    if (wordCount === 0) {
      textEl.textContent = "Add 150+ words for best reliability";
    } else if (wordCount < 80) {
      fillEl.classList.add("tier-low");
      textEl.textContent = "Low signal — keep writing";
    } else if (wordCount < 200) {
      fillEl.classList.add("tier-moderate");
      textEl.textContent = "Moderate signal — good";
    } else {
      fillEl.classList.add("tier-high");
      textEl.textContent = "High signal — ready";
    }
  }

  function showError(el, msg) {
    el.textContent = msg;
    el.classList.remove("is-hidden");
  }

  function clearError(el) {
    el.textContent = "";
    el.classList.add("is-hidden");
  }

  // Navigation
  function switchView(target) {
    currentView = target;
    navLinks.forEach((btn) => {
      btn.classList.toggle("is-active", btn.dataset.view === target);
    });
    Object.entries(views).forEach(([key, section]) => {
      section.classList.toggle("is-active", key === target);
    });

    if (target === "reference") {
      resultsSection.classList.add("is-hidden");
      loadReferenceTypes();
    } else if (lastResult) {
      resultsSection.classList.remove("is-hidden");
    }
  }

  navLinks.forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  brandLink.addEventListener("click", (e) => {
    e.preventDefault();
    switchView("analyze");
  });

  // Multiple entries management
  function createEntryCard(initialValue = "") {
    const idx = entriesList.children.length + 1;
    const item = document.createElement("div");
    item.className = "entry-item";
    item.innerHTML = `
      <div class="entry-head">
        <span class="entry-label">Entry ${String(idx).padStart(2, "0")}</span>
        <button type="button" class="entry-remove">Remove</button>
      </div>
      <textarea class="entry-textarea" placeholder="Paste or write a post, email, or note..."></textarea>
    `;
    const ta = item.querySelector(".entry-textarea");
    ta.value = initialValue;
    ta.addEventListener("input", refreshAnalyzeMeter);

    item.querySelector(".entry-remove").addEventListener("click", () => {
      if (entriesList.children.length <= 1) {
        ta.value = "";
      } else {
        item.remove();
        renumberEntries();
      }
      refreshAnalyzeMeter();
    });

    entriesList.appendChild(item);
  }

  function renumberEntries() {
    Array.from(entriesList.children).forEach((el, i) => {
      const lbl = el.querySelector(".entry-label");
      if (lbl) lbl.textContent = `Entry ${String(i + 1).padStart(2, "0")}`;
    });
  }

  function getAnalyzeEntries() {
    if (inputMode === "single") {
      return freeTextInput.value.trim() ? [freeTextInput.value.trim()] : [];
    }
    return Array.from(entriesList.querySelectorAll(".entry-textarea"))
      .map((ta) => ta.value.trim())
      .filter(Boolean);
  }

  function refreshAnalyzeMeter() {
    const total = getAnalyzeEntries().reduce((sum, txt) => sum + countWords(txt), 0);
    updateMeter(total, analyzeWordCount, analyzeSignalFill, analyzeSignalText);
  }

  function setInputMode(mode) {
    inputMode = mode;
    modeSingleBtn.classList.toggle("is-active", mode === "single");
    modeSingleBtn.setAttribute("aria-selected", String(mode === "single"));
    modeMultiBtn.classList.toggle("is-active", mode === "multi");
    modeMultiBtn.setAttribute("aria-selected", String(mode === "multi"));
    singleInputWrap.classList.toggle("is-hidden", mode !== "single");
    multiInputWrap.classList.toggle("is-hidden", mode !== "multi");
    if (mode === "multi" && entriesList.children.length === 0) {
      createEntryCard();
      createEntryCard();
    }
    refreshAnalyzeMeter();
  }

  modeSingleBtn.addEventListener("click", () => setInputMode("single"));
  modeMultiBtn.addEventListener("click", () => setInputMode("multi"));
  freeTextInput.addEventListener("input", refreshAnalyzeMeter);
  addEntryBtn.addEventListener("click", () => {
    createEntryCard();
    refreshAnalyzeMeter();
  });

  // Sample loaders
  document.querySelectorAll("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sample = SAMPLES[btn.dataset.sample];
      if (!sample) return;
      clearError(analyzeError);
      if (inputMode === "single") {
        freeTextInput.value = sample.join("\n\n");
      } else {
        entriesList.innerHTML = "";
        sample.forEach((p) => createEntryCard(p));
      }
      refreshAnalyzeMeter();
    });
  });

  clearAnalyzeBtn.addEventListener("click", () => {
    clearError(analyzeError);
    freeTextInput.value = "";
    entriesList.innerHTML = "";
    if (inputMode === "multi") {
      createEntryCard();
      createEntryCard();
    }
    refreshAnalyzeMeter();
  });

  // Guided prompts setup
  function buildPrompts() {
    promptsList.innerHTML = "";
    PROMPTS.forEach((p, idx) => {
      const item = document.createElement("div");
      item.className = "prompt-item";
      item.innerHTML = `
        <div class="prompt-header">
          <label class="prompt-label" for="prompt-input-${idx}">
            <span class="prompt-num">${p.num}</span>
            <span>${p.question}</span>
          </label>
          <span class="prompt-wc" id="prompt-wc-${idx}">0 words</span>
        </div>
        <textarea
          id="prompt-input-${idx}"
          class="prompt-textarea"
          placeholder="${p.placeholder}"
        ></textarea>
      `;
      const ta = item.querySelector("textarea");
      const wc = item.querySelector(`#prompt-wc-${idx}`);
      ta.addEventListener("input", () => {
        const w = countWords(ta.value);
        wc.textContent = `${w} ${w === 1 ? "word" : "words"}`;
        refreshPromptsMeter();
      });
      promptsList.appendChild(item);
    });
  }

  function getPromptResponses() {
    return Array.from(promptsList.querySelectorAll(".prompt-textarea"))
      .map((ta) => ta.value.trim())
      .filter(Boolean);
  }

  function refreshPromptsMeter() {
    const total = getPromptResponses().reduce((sum, txt) => sum + countWords(txt), 0);
    updateMeter(total, promptsWordCount, promptsSignalFill, promptsSignalText);
  }

  clearPromptsBtn.addEventListener("click", () => {
    clearError(promptsError);
    promptsList.querySelectorAll(".prompt-textarea").forEach((ta, idx) => {
      ta.value = "";
      const wc = document.getElementById(`prompt-wc-${idx}`);
      if (wc) wc.textContent = "0 words";
    });
    refreshPromptsMeter();
  });

  // Run Prediction
  async function executePrediction(entries, buttonEl, errorEl) {
    clearError(errorEl);
    if (!entries.length) {
      showError(errorEl, "Enter some writing first so the model has text to analyze.");
      return;
    }

    const originalLabel = buttonEl.textContent;
    buttonEl.disabled = true;
    buttonEl.textContent = "Analyzing...";

    try {
      const payload = entries.length === 1 ? { text: entries[0] } : { posts: entries };
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(errorEl, data.detail || "Could not analyze this input.");
        return;
      }

      lastResult = data;
      renderResult(data);
      resultsSection.classList.remove("is-hidden");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError(errorEl, "Unable to reach the local backend server. Make sure app.py is running.");
    } finally {
      buttonEl.disabled = false;
      buttonEl.textContent = originalLabel;
    }
  }

  runAnalyzeBtn.addEventListener("click", () => {
    executePrediction(getAnalyzeEntries(), runAnalyzeBtn, analyzeError);
  });

  runPromptsBtn.addEventListener("click", () => {
    executePrediction(getPromptResponses(), runPromptsBtn, promptsError);
  });

  // Render Dossier
  function renderResult(data) {
    resWordCount.textContent = `${data.word_count} words`;
    resReliability.textContent = data.reliability_label;
    resEngine.textContent = data.engine.mode === "ml_ensemble" ? "ML Ensemble" : "Built-in Estimator";

    resTypeCode.textContent = data.predicted_type;
    resTypeGroup.textContent = data.profile.group;
    resTypeTitle.textContent = data.profile.title;
    resTypeSummary.textContent = data.profile.summary;

    // Cognitive function pills
    resFunctions.innerHTML = "";
    data.profile.functions.forEach((fn, i) => {
      const span = document.createElement("span");
      span.className = `fn-pill${i === 0 ? " is-dominant" : ""}`;
      span.textContent = fn;
      resFunctions.appendChild(span);
    });

    // Diverging spectrum bars
    resAxesList.innerHTML = "";
    data.axes.forEach((ax) => {
      const posPct = (ax.pos_prob * 100).toFixed(1);
      const negPct = (ax.neg_prob * 100).toFixed(1);
      const isPos = ax.predicted_letter === ax.pos_letter;
      const offsetPct = Math.abs(ax.pos_prob - 0.5) * 100; // 0 to 50% of track width

      const row = document.createElement("div");
      row.className = "axis-row";
      row.innerHTML = `
        <div class="axis-pole left">
          <span class="pole-title${isPos ? " is-winner" : ""}">${ax.pos_letter} &middot; ${ax.pos_name}</span>
          <span class="pole-pct">${posPct}%</span>
        </div>
        <div class="spectrum-track" aria-label="${ax.label}: ${posPct}% ${ax.pos_name}, ${negPct}% ${ax.neg_name}">
          <div class="spectrum-center"></div>
          <div class="spectrum-fill ${isPos ? "side-left" : "side-right"}" style="width: ${offsetPct.toFixed(1)}%"></div>
        </div>
        <div class="axis-pole right">
          <span class="pole-title${!isPos ? " is-winner" : ""}">${ax.neg_name} &middot; ${ax.neg_letter}</span>
          <span class="pole-pct">${negPct}%</span>
        </div>
      `;
      resAxesList.appendChild(row);
    });

    // Top 3 probable types
    resTop3.innerHTML = "";
    data.top_3_types.forEach((item, idx) => {
      const row = document.createElement("div");
      row.className = `top3-row${idx === 0 ? " is-first" : ""}`;
      row.innerHTML = `
        <span class="top3-code">${item.type}</span>
        <span class="top3-title">${item.title}</span>
        <span class="top3-prob">${(item.probability * 100).toFixed(1)}%</span>
      `;
      resTop3.appendChild(row);
    });

    // Profile details
    resComm.textContent = data.profile.communication;
    resStrengths.innerHTML = data.profile.strengths.map((s) => `<li>${s}</li>`).join("");
    resBlindspots.innerHTML = data.profile.blind_spots.map((s) => `<li>${s}</li>`).join("");

    // Linguistic signals
    const s = data.signals;
    const signalRows = [
      ["Average word length", `${s.avg_word_length} chars`],
      ["Words per sentence", `${s.words_per_sentence}`],
      ["Lexical diversity (MATTR)", `${(s.lexical_diversity * 100).toFixed(1)}%`],
      ["Exclamation rate", `${s.exclamation_rate} / 100w`],
      ["Question rate", `${s.question_rate} / 100w`],
      ["First-person focus (I/my)", `${s.first_person_rate} / 100w`],
      ["Social pronouns (we/you)", `${s.social_pronoun_rate} / 100w`],
      ["Analytical vs. feeling index", `${s.analytical_vs_feeling >= 0 ? "+" : ""}${s.analytical_vs_feeling}`],
      ["Planning vs. spontaneous index", `${s.planning_vs_spontaneous >= 0 ? "+" : ""}${s.planning_vs_spontaneous}`],
    ];
    resSignals.innerHTML = signalRows
      .map(([k, v]) => `<div class="signal-row"><dt>${k}</dt><dd>${v}</dd></div>`)
      .join("");

    if (data.engine.mode !== "ml_ensemble") {
      resEngineNote.className = "caveat-box is-warn";
      resEngineNote.textContent =
        "Currently running on the built-in psycholinguistic estimator. For predictions powered by the full 8,675-user TF-IDF + Stylometry + LightGBM ensemble, click the engine indicator in the header to upload mbti_predictor.joblib.";
    } else {
      resEngineNote.className = "caveat-box";
      resEngineNote.textContent =
        "Predicted with your trained 4-axis balanced ML ensemble (Word & Character TF-IDF + 16 stylometric signals + SVD/GBDT). Explicit MBTI keywords are ignored to ensure real-world generalization.";
    }
  }

  copyReportBtn.addEventListener("click", async () => {
    if (!lastResult) return;
    const lines = [
      `Typology Assessment Summary`,
      `Predicted Type: ${lastResult.predicted_type} (${lastResult.profile.title})`,
      `Cognitive Stack: ${lastResult.profile.functions.join(" - ")}`,
      `Input: ${lastResult.word_count} words (${lastResult.reliability_label})`,
      ``,
      `Axis Breakdown:`,
      ...lastResult.axes.map(
        (ax) =>
          `  ${ax.pos_letter} ${(ax.pos_prob * 100).toFixed(1)}% / ${(ax.neg_prob * 100).toFixed(1)}% ${ax.neg_letter} -> ${ax.predicted_name}`
      ),
      ``,
      `Top 3 Probable Types: ${lastResult.top_3_types
        .map((t) => `${t.type} (${(t.probability * 100).toFixed(1)}%)`)
        .join(", ")}`,
    ];
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      copyReportBtn.textContent = "Copied";
      setTimeout(() => {
        copyReportBtn.textContent = "Copy Summary";
      }, 1800);
    } catch {
      copyReportBtn.textContent = "Copy failed";
      setTimeout(() => {
        copyReportBtn.textContent = "Copy Summary";
      }, 1800);
    }
  });

  editInputBtn.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // 16 Types Reference
  async function loadReferenceTypes() {
    if (!typesCache) {
      try {
        const resp = await fetch("/api/types");
        const data = await resp.json();
        typesCache = data.types;
      } catch {
        referenceGroups.innerHTML = "<p class='lead'>Unable to load type reference.</p>";
        return;
      }
    }
    if (referenceGroups.children.length > 0) return;

    GROUP_ORDER.forEach((groupName) => {
      const section = document.createElement("div");
      const heading = document.createElement("h2");
      heading.className = "ref-group-title";
      heading.textContent = groupName;
      section.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "ref-grid";

      Object.entries(typesCache)
        .filter(([, info]) => info.group === groupName)
        .forEach(([code, info]) => {
          const card = document.createElement("button");
          card.type = "button";
          card.className = "ref-card";
          card.dataset.code = code;
          card.innerHTML = `
            <div class="ref-card-code">${code}</div>
            <div class="ref-card-title">${info.title}</div>
            <div class="ref-card-fn">${info.functions.join(" · ")}</div>
          `;
          card.addEventListener("click", () => selectReferenceType(code, info));
          grid.appendChild(card);
        });

      section.appendChild(grid);
      referenceGroups.appendChild(section);
    });
  }

  function selectReferenceType(code, info) {
    document.querySelectorAll(".ref-card").forEach((c) => {
      c.classList.toggle("is-selected", c.dataset.code === code);
    });

    referenceDetail.classList.remove("is-hidden");
    referenceDetail.innerHTML = `
      <div class="ref-detail-top">
        <div>
          <p class="eyebrow">${info.group} &middot; ${info.functions.join(" - ")}</p>
          <h3 class="ref-detail-heading">
            <span class="ref-detail-code">${code}</span>${info.title}
          </h3>
        </div>
        <button type="button" class="close-btn" id="close-ref-detail">Dismiss</button>
      </div>
      <p class="type-summary">${info.summary}</p>
      <div class="divider-rule"></div>
      <div class="two-list-grid">
        <div>
          <h4 style="margin:0 0 6px 0;font-size:0.82rem;">Communication Style</h4>
          <p style="margin:0 0 14px 0;font-size:0.88rem;color:var(--ink-secondary);">${info.communication}</p>
          <h4 style="margin:0 0 6px 0;font-size:0.82rem;">Core Strengths</h4>
          <ul class="clean-list">${info.strengths.map((s) => `<li>${s}</li>`).join("")}</ul>
        </div>
        <div>
          <h4 style="margin:0 0 6px 0;font-size:0.82rem;">Potential Blind Spots</h4>
          <ul class="clean-list">${info.blind_spots.map((s) => `<li>${s}</li>`).join("")}</ul>
        </div>
      </div>
    `;
    document.getElementById("close-ref-detail").addEventListener("click", () => {
      referenceDetail.classList.add("is-hidden");
      document.querySelectorAll(".ref-card").forEach((c) => c.classList.remove("is-selected"));
    });
    referenceDetail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Model status & upload modal
  async function refreshEngineStatus() {
    try {
      const resp = await fetch("/api/status");
      const st = await resp.json();
      const isML = st.mode === "ml_ensemble";
      engineDot.classList.toggle("is-ml", isML);
      engineLabel.textContent = isML ? "ML Ensemble" : "Estimator Mode";
      modalEngineMode.textContent = st.mode_label;
      modalEngineFile.textContent = st.model_file || "None loaded";
      modalMasking.textContent = st.remove_mbti_words ? "Enabled (anti-leakage)" : "Disabled";
    } catch {
      engineLabel.textContent = "Offline";
    }
  }

  function openModal() {
    refreshEngineStatus();
    uploadFeedback.classList.add("is-hidden");
    modelModal.classList.remove("is-hidden");
  }

  function closeModal() {
    modelModal.classList.add("is-hidden");
  }

  openModelModalBtn.addEventListener("click", openModal);
  closeModelModalBtn.addEventListener("click", closeModal);
  modelModal.addEventListener("click", (e) => {
    if (e.target === modelModal) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modelModal.classList.contains("is-hidden")) {
      closeModal();
    }
  });

  async function uploadModelFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".joblib")) {
      uploadFeedback.className = "upload-feedback is-err";
      uploadFeedback.textContent = "File must end with .joblib.";
      uploadFeedback.classList.remove("is-hidden");
      return;
    }

    dropzoneLabel.textContent = `Uploading ${file.name}...`;
    uploadFeedback.classList.add("is-hidden");

    try {
      const resp = await fetch("/api/upload-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-Filename": file.name,
        },
        body: file,
      });
      const data = await resp.json();
      if (!resp.ok) {
        uploadFeedback.className = "upload-feedback is-err";
        uploadFeedback.textContent = data.detail || "Upload failed.";
      } else {
        uploadFeedback.className = "upload-feedback is-ok";
        uploadFeedback.textContent = data.message;
        await refreshEngineStatus();
      }
      uploadFeedback.classList.remove("is-hidden");
    } catch {
      uploadFeedback.className = "upload-feedback is-err";
      uploadFeedback.textContent = "Network error while uploading model.";
      uploadFeedback.classList.remove("is-hidden");
    } finally {
      dropzoneLabel.textContent = "Choose mbti_predictor.joblib or drop it here";
      modelFileInput.value = "";
    }
  }

  modelFileInput.addEventListener("change", () => {
    if (modelFileInput.files && modelFileInput.files[0]) {
      uploadModelFile(modelFileInput.files[0]);
    }
  });

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("is-dragover");
  });

  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("is-dragover");
  });

  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("is-dragover");
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadModelFile(e.dataTransfer.files[0]);
    }
  });

  // Initialize
  buildPrompts();
  refreshAnalyzeMeter();
  refreshPromptsMeter();
  refreshEngineStatus();
})();
