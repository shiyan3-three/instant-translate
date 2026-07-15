/**
 * instant-translate GUI mock — phase switch + light interactions.
 * No backend. Open index.html in a browser.
 */

(function () {
  const PHASES = [
    { id: "1", label: "1 主窗骨架" },
    { id: "2", label: "2 任务流" },
    { id: "3", label: "3 弹窗" },
    { id: "4", label: "4 Overlay" },
  ];

  const PAGE_TITLES = {
    template: "翻译模板",
    model: "模型配置",
    feedback: "优化翻译",
    settings: "设置",
  };

  const phaseNav = document.getElementById("phaseNav");
  const panels = document.querySelectorAll(".phase-panel");
  const titlePage = document.getElementById("titlePage");
  const statusText = document.getElementById("statusText");
  const progress = document.getElementById("progress");

  function setPhase(id) {
    // Phase 2 reuses the phase-1 desktop shell; show shell + phase-2 notes together.
    panels.forEach((p) => {
      const show =
        p.dataset.phase === id || (id === "2" && p.dataset.phase === "1");
      p.classList.toggle("is-active", show);
    });
    phaseNav.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.phase === id);
    });
    location.hash = "phase-" + id;
  }

  PHASES.forEach((ph, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = ph.label;
    btn.dataset.phase = ph.id;
    if (i === 0) btn.classList.add("is-active");
    btn.addEventListener("click", () => setPhase(ph.id));
    phaseNav.appendChild(btn);
  });

  // Sidebar pages
  const navBtns = document.querySelectorAll(".nav-btn");
  const pages = document.querySelectorAll(".page");

  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const page = btn.dataset.page;
      navBtns.forEach((b) => b.classList.toggle("is-active", b === btn));
      pages.forEach((p) => p.classList.toggle("is-active", p.dataset.page === page));
      if (titlePage) titlePage.textContent = PAGE_TITLES[page] || page;
      if (statusText) {
        statusText.textContent =
          page === "feedback"
            ? "选择框: 1/3  |  模式: 普通"
            : "就绪 — Ctrl+Shift+Z 新建框选";
      }
    });
  });

  // Feedback tabs
  const feedbackTabs = document.getElementById("feedbackTabs");
  if (feedbackTabs) {
    feedbackTabs.addEventListener("click", (e) => {
      const tab = e.target.closest(".tab");
      if (!tab) return;
      const name = tab.dataset.tab;
      feedbackTabs.querySelectorAll(".tab").forEach((t) => {
        t.classList.toggle("is-active", t === tab);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("is-active", panel.dataset.tab === name);
      });
    });
  }

  // Dialogs
  function openDialog(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add("is-open");
  }

  function closeDialog(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove("is-open");
  }

  document.getElementById("openPreviewDialog")?.addEventListener("click", () => {
    openDialog("previewDialog");
  });
  document.getElementById("demoPreview")?.addEventListener("click", () => {
    openDialog("previewDialog");
  });
  document.getElementById("openReviewDialog")?.addEventListener("click", () => {
    if (progress) {
      progress.classList.add("is-on");
      setTimeout(() => progress.classList.remove("is-on"), 1200);
    }
    setTimeout(() => openDialog("reviewDialog"), 400);
  });
  document.getElementById("demoReview")?.addEventListener("click", () => {
    openDialog("reviewDialog");
  });

  document.querySelectorAll("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => closeDialog(btn.dataset.close));
  });

  document.querySelectorAll(".dialog-layer").forEach((layer) => {
    layer.addEventListener("click", (e) => {
      if (e.target === layer) layer.classList.remove("is-open");
    });
  });

  // Preview dialog tabs
  const previewTabs = document.getElementById("previewTabs");
  if (previewTabs) {
    previewTabs.addEventListener("click", (e) => {
      const tab = e.target.closest(".tab");
      if (!tab) return;
      const name = tab.dataset.previewTab;
      previewTabs.querySelectorAll(".tab").forEach((t) => {
        t.classList.toggle("is-active", t === tab);
      });
      document.querySelectorAll("[data-preview-panel]").forEach((panel) => {
        const show = panel.dataset.previewPanel === name;
        panel.hidden = !show;
      });
    });
  }

  // Review nav + mock actions
  const reviewSide = document.getElementById("reviewSide");
  reviewSide?.addEventListener("click", (e) => {
    const nav = e.target.closest(".review-nav");
    if (!nav) return;
    reviewSide.querySelectorAll(".review-nav").forEach((n) => {
      n.classList.toggle("is-active", n === nav);
    });
  });

  document.querySelectorAll("[data-review-act]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const act = btn.dataset.reviewAct;
      const active = reviewSide?.querySelector(".review-nav.is-active");
      if (!active) return;
      if (act === "skip") active.dataset.state = "skipped";
      if (act === "ai" || act === "user") active.dataset.state = "done";
      if (act === "done") closeDialog("reviewDialog");
      // advance to next pending
      if (act !== "done" && reviewSide) {
        const items = [...reviewSide.querySelectorAll(".review-nav")];
        const idx = items.indexOf(active);
        const next =
          items.slice(idx + 1).find((n) => n.dataset.state === "pending") ||
          items.find((n) => n.dataset.state === "pending");
        if (next) {
          items.forEach((n) => n.classList.toggle("is-active", n === next));
        }
      }
    });
  });

  // —— Overlay phase 4 ——
  const overlayStage = document.getElementById("overlayStage");
  const overlayModeHint = document.getElementById("overlayModeHint");
  const showOcrWindow = document.getElementById("showOcrWindow");
  const showMultiGroup = document.getElementById("showMultiGroup");

  function setOverlayMode(mode) {
    if (!overlayStage) return;
    const isEdit = mode === "edit";
    overlayStage.classList.toggle("is-edit", isEdit);
    overlayStage.classList.toggle("is-normal", !isEdit);
    document.querySelectorAll("[data-overlay-mode]").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.overlayMode === mode);
    });
    overlayStage.querySelectorAll("[data-edit-only]").forEach((el) => {
      el.hidden = !isEdit;
    });
    if (overlayModeHint) {
      overlayModeHint.textContent = isEdit
        ? "编辑：可拖选择框 / 译文 / OCR · 工具栏显示"
        : "普通：细边框 · 无工具栏 · 示意点击穿透";
    }
  }

  document.querySelectorAll("[data-overlay-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setOverlayMode(btn.dataset.overlayMode));
  });

  function syncOcrVisibility() {
    const on = !showOcrWindow || showOcrWindow.checked;
    document.querySelectorAll("[data-ocr-card]").forEach((el) => {
      el.hidden = !on;
    });
    overlayStage?.classList.toggle("ocr-hidden", !on);
  }

  function syncMultiGroup() {
    const on = showMultiGroup && showMultiGroup.checked;
    document.querySelectorAll(".ov-group.multi").forEach((el) => {
      el.hidden = !on;
    });
  }

  showOcrWindow?.addEventListener("change", syncOcrVisibility);
  showMultiGroup?.addEventListener("change", syncMultiGroup);

  // corner bars + main toolbar interactions
  overlayStage?.addEventListener("click", (e) => {
    const collapse = e.target.closest("[data-collapse]");
    if (collapse) {
      const bar = collapse.closest(".corner-bar");
      bar?.classList.toggle("is-collapsed");
      // expanded: › (point into tools / collapse), collapsed: ‹ (point to expand)
      const collapsed = bar?.classList.contains("is-collapsed");
      collapse.textContent = collapsed ? "‹" : "›";
      return;
    }

    const pauseBtn = e.target.closest("[data-pause]");
    if (pauseBtn) {
      const paused = pauseBtn.dataset.paused === "1";
      pauseBtn.dataset.paused = paused ? "0" : "1";
      pauseBtn.textContent = paused ? "❚❚" : "▶";
      pauseBtn.title = paused ? "暂停" : "继续";
      return;
    }

    const dockBtn = e.target.closest("[data-dock]");
    if (dockBtn) {
      const bar = dockBtn.closest(".corner-tools");
      bar?.querySelectorAll("[data-dock]").forEach((b) => {
        b.classList.toggle("is-on", b === dockBtn);
      });
      return;
    }

    const bodyInv = e.target.closest("[data-body-invisible]");
    if (bodyInv) {
      const which = bodyInv.dataset.bodyInvisible;
      const group = bodyInv.closest(".ov-group");
      const card =
        which === "tx"
          ? group?.querySelector("[data-tx-card]")
          : group?.querySelector("[data-ocr-card]");
      card?.classList.toggle("is-body-hidden");
      const on = card?.classList.contains("is-body-hidden");
      bodyInv.classList.toggle("is-on", !!on);
      bodyInv.classList.remove("is-dim");
      bodyInv.title = on
        ? "退出沉浸：显示框体"
        : "沉浸：隐藏框体，保留文字";
      return;
    }

    const groupInv = e.target.closest("[data-group-invisible]");
    if (groupInv) {
      const group = groupInv.closest(".ov-group");
      group?.classList.toggle("is-group-hidden");
      const on = group?.classList.contains("is-group-hidden");
      groupInv.classList.toggle("is-on", !!on);
      groupInv.classList.remove("is-dim");
      groupInv.title = on
        ? "退出沉浸：显示框体与角工具栏"
        : "沉浸：隐藏框体与角工具栏，保留文字";
      return;
    }

    const toggleOcr = e.target.closest("[data-toggle-ocr]");
    if (toggleOcr && showOcrWindow) {
      showOcrWindow.checked = !showOcrWindow.checked;
      syncOcrVisibility();
      return;
    }

    // mock copy feedback
    if (e.target.closest("[data-copy-tx], [data-copy-ocr]")) {
      const btn = e.target.closest("button");
      const prev = btn.textContent;
      btn.textContent = "✓";
      setTimeout(() => {
        btn.textContent = prev;
      }, 600);
    }
  });

  /**
   * Drag ov-group (selection+cards move together) or free cards in edit mode.
   * - selection-box: moves whole .ov-group
   * - translation-card / ocr-card: free position relative to group
   */
  function setupOverlayDrag() {
    if (!overlayStage) return;

    let drag = null;

    function isEdit() {
      return overlayStage.classList.contains("is-edit");
    }

    function onPointerDown(e) {
      if (!isEdit() || e.button !== 0) return;
      if (e.target.closest("button, select, input, label")) return;

      const group = e.target.closest(".ov-group");
      if (!group || group.hidden) return;

      const freeCard = e.target.closest(".translation-card, .ocr-card");
      const onBox = e.target.closest(".selection-box");

      let target = null;
      let mode = null;
      if (onBox) {
        target = group;
        mode = "group";
      } else if (freeCard) {
        target = freeCard;
        mode = "card";
      } else {
        return;
      }

      const stageRect = overlayStage.getBoundingClientRect();
      const left = parseFloat(target.style.left || getComputedStyle(target).left) || 0;
      const top = parseFloat(target.style.top || getComputedStyle(target).top) || 0;

      // For cards, style.left/top may be empty — use offset within group math:
      let baseLeft = left;
      let baseTop = top;
      if (mode === "card" && !target.style.left) {
        baseLeft = target.offsetLeft;
        baseTop = target.offsetTop;
      }
      if (mode === "group" && !target.style.left) {
        baseLeft = target.offsetLeft;
        baseTop = target.offsetTop;
      }

      drag = {
        mode,
        target,
        group,
        startX: e.clientX,
        startY: e.clientY,
        baseLeft,
        baseTop,
        stageW: stageRect.width,
        stageH: stageRect.height,
      };
      target.classList.add("is-dragging");
      group.classList.add("is-dragging");
      target.setPointerCapture?.(e.pointerId);
      e.preventDefault();
    }

    function onPointerMove(e) {
      if (!drag) return;
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      let nx = drag.baseLeft + dx;
      let ny = drag.baseTop + dy;

      if (drag.mode === "group") {
        // clamp group origin roughly inside stage
        nx = Math.max(8, Math.min(nx, drag.stageW - 40));
        ny = Math.max(40, Math.min(ny, drag.stageH - 40));
        drag.target.style.left = nx + "px";
        drag.target.style.top = ny + "px";
      } else {
        // free card within stage relative to group — allow generous range
        drag.target.style.left = nx + "px";
        drag.target.style.top = ny + "px";
      }
    }

    function onPointerUp(e) {
      if (!drag) return;
      drag.target.classList.remove("is-dragging");
      drag.group.classList.remove("is-dragging");
      try {
        drag.target.releasePointerCapture?.(e.pointerId);
      } catch (_) {
        /* ignore */
      }
      drag = null;
    }

    overlayStage.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
  }

  setOverlayMode("normal");
  syncOcrVisibility();
  syncMultiGroup();
  setupOverlayDrag();

  // Hash restore
  const hash = (location.hash || "").replace("#phase-", "");
  if (hash && PHASES.some((p) => p.id === hash)) {
    setPhase(hash);
  }
})();
