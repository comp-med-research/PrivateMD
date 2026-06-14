import gradio as gr

from private_md import analyze_patient, answer_question, patient_choices


CSS = """
:root {
  --ink: #111827;
  --muted: #667085;
  --line: rgba(17, 24, 39, 0.12);
  --panel: rgba(255, 255, 255, 0.86);
  --panel-strong: rgba(255, 255, 255, 0.96);
  --brand: #0f766e;
  --brand-dark: #0b4f4a;
}
html, body, .gradio-container {
  min-height: 100%;
  background: #f8fafc !important;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.gradio-container {
  padding: 0 0 36px 76px !important;
}
footer, .built-with, .api-docs {
  display: none !important;
}
.pm-ambient {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;
  background:
    radial-gradient(circle at 50% 28%, rgba(15, 118, 110, 0.12), transparent 34%),
    radial-gradient(circle at 72% 70%, rgba(34, 211, 238, 0.10), transparent 30%),
    linear-gradient(180deg, #fbfdff 0%, #eef7f6 48%, #f8fafc 100%);
}
#pm-mesh {
  width: 100%;
  height: 100%;
  opacity: 0.48;
  filter: saturate(0.9);
}
.pm-rail {
  position: fixed;
  inset: 0 auto 0 0;
  width: 68px;
  z-index: 3;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 22px;
  padding: 24px 0 20px;
  border-right: 1px solid rgba(17, 24, 39, 0.08);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(18px);
}
.pm-mark {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: #fff;
  background: radial-gradient(circle at 30% 20%, #2dd4bf, #0f766e 68%, #0b4f4a);
  font-weight: 760;
  letter-spacing: 0;
  box-shadow: 0 12px 26px rgba(15, 118, 110, 0.26);
}
.pm-rail-icon {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  color: #111827;
  font-size: 18px;
}
.pm-rail-spacer {
  flex: 1;
}
.pm-user {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: #c75d2c;
  color: white;
  font-size: 13px;
  font-weight: 700;
}
.pm-shell {
  position: relative;
  z-index: 1;
  width: min(1120px, calc(100vw - 116px));
  margin: 0 auto;
  padding: 26px 0 0;
}
.pm-topbar {
  align-items: end;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(18px);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 16px 50px rgba(15, 23, 42, 0.07);
}
.pm-topbar .wrap {
  gap: 10px !important;
}
.pm-topbar button {
  min-height: 42px !important;
}
.pm-hero {
  min-height: 360px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 42px 0 18px;
}
.pm-kicker {
  margin: 0 0 12px;
  color: var(--brand-dark);
  font-size: 13px;
  font-weight: 760;
  letter-spacing: 0;
}
.pm-title {
  margin: 0;
  font-size: 34px;
  line-height: 1.16;
  font-weight: 670;
  letter-spacing: 0;
}
.pm-subtitle {
  margin: 12px auto 0;
  max-width: 720px;
  color: var(--muted);
  font-size: 15px;
  line-height: 1.55;
}
.pm-prompt-card {
  margin: -92px auto 28px;
  max-width: 820px;
  border-radius: 8px;
  border: 1px solid rgba(17, 24, 39, 0.14);
  background: var(--panel-strong);
  box-shadow: 0 24px 70px rgba(15, 23, 42, 0.16);
  padding: 10px 10px 10px 18px;
}
.pm-prompt-row {
  align-items: stretch;
}
#pm-question textarea {
  min-height: 52px !important;
  max-height: 130px;
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  font-size: 16px !important;
  padding: 14px 8px !important;
}
#pm-question label {
  display: none !important;
}
#pm-question .wrap {
  border: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
}
#pm-ask {
  align-self: center;
  min-width: 54px !important;
  height: 54px !important;
  border-radius: 50% !important;
  padding: 0 !important;
  font-size: 0 !important;
  background: #050505 !important;
  border: 0 !important;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.20);
}
#pm-ask::after {
  content: "↑";
  font-size: 25px;
  line-height: 1;
  color: white;
}
.pm-chips {
  display: flex;
  gap: 10px;
  justify-content: center;
  flex-wrap: wrap;
  margin: -12px 0 28px;
}
.pm-chip {
  border: 1px solid rgba(17, 24, 39, 0.13);
  background: rgba(255, 255, 255, 0.72);
  color: #525866;
  border-radius: 999px;
  padding: 9px 14px;
  font-size: 14px;
  backdrop-filter: blur(12px);
}
.pm-results {
  border-radius: 8px;
  border: 1px solid rgba(17, 24, 39, 0.12);
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(18px);
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.10);
  padding: 16px;
}
.pm-visual iframe {
  min-height: 520px;
}
.pm-answer {
  margin-top: 14px;
  padding: 18px 20px;
  border: 1px solid rgba(17, 24, 39, 0.10);
  background: rgba(255, 255, 255, 0.86);
  border-radius: 8px;
}
.pm-context {
  margin-top: 18px;
}
.pm-context .label-wrap,
.pm-results .label-wrap {
  color: var(--muted) !important;
}
.pm-context textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
  font-size: 12px !important;
}
.metric-band {
  background: rgba(255, 255, 255, 0.76);
  border: 1px solid rgba(17, 24, 39, 0.10);
  border-radius: 8px;
  padding: 14px 16px;
}
button.primary {
  background: var(--brand) !important;
  border-color: var(--brand) !important;
}
button.secondary {
  border-radius: 8px !important;
}
@media (max-width: 760px) {
  .gradio-container {
    padding-left: 0 !important;
  }
  .pm-rail {
    display: none;
  }
  .pm-shell {
    width: calc(100vw - 24px);
    padding-top: 12px;
  }
  .pm-title {
    font-size: 27px;
  }
  .pm-hero {
    min-height: 300px;
  }
  .pm-prompt-card {
    margin-top: -70px;
  }
}
"""


BACKGROUND_HTML = """
<div class="pm-ambient"><canvas id="pm-mesh"></canvas></div>
<aside class="pm-rail">
  <div class="pm-mark">PM</div>
  <div class="pm-rail-icon" title="New review">✎</div>
  <div class="pm-rail-icon" title="Search">⌕</div>
  <div class="pm-rail-icon" title="Evidence">◌</div>
  <div class="pm-rail-spacer"></div>
  <div class="pm-user">HA</div>
</aside>
"""


MESH_JS = """
() => {
  const startMesh = () => {
    const canvas = document.getElementById("pm-mesh");
    if (!canvas || canvas.dataset.ready) return;
    canvas.dataset.ready = "1";
    const ctx = canvas.getContext("2d");
    let w = 0;
    let h = 0;
    let points = [];
    const palette = ["#0f766e", "#0891b2", "#22c55e", "#94a3b8"];

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.max(54, Math.floor((w * h) / 22000));
      points = Array.from({ length: count }, (_, i) => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.34,
        vy: (Math.random() - 0.5) * 0.34,
        r: 1.2 + Math.random() * 2.2,
        c: palette[i % palette.length],
      }));
    }

    function draw() {
      ctx.clearRect(0, 0, w, h);
      for (const p of points) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -20 || p.x > w + 20) p.vx *= -1;
        if (p.y < -20 || p.y > h + 20) p.vy *= -1;
      }
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const a = points[i];
          const b = points[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 145) {
            ctx.globalAlpha = (1 - dist / 145) * 0.32;
            ctx.strokeStyle = "#0f766e";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
      for (const p of points) {
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = p.c;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(draw);
    }

    resize();
    window.addEventListener("resize", resize);
    draw();
  };
  startMesh();
  setTimeout(startMesh, 600);
  setTimeout(startMesh, 1500);
}
"""


EMPTY_VISUALIZATION = (
    "<div style='border:1px solid rgba(17,24,39,0.12);border-radius:8px;padding:24px;"
    "background:rgba(255,255,255,0.78);color:#667085;text-align:center;'>"
    "Ask a question to see LangExtract highlight the source document here.</div>"
)


choices = patient_choices()
choice_labels = [choice.label for choice in choices]
choice_lookup = {choice.label: choice.path for choice in choices}
PREFERRED_DEMO_PATIENT_ID = "c04befa5-e835-3f4c-2b90-88da0af27c0d"
default_choice = next(
    (label for label in choice_labels if PREFERRED_DEMO_PATIENT_ID in label),
    choice_labels[0] if choice_labels else None,
)


def select_patient(label):
    if not label:
        return "No patient selected.", None, None, ""
    return analyze_patient(choice_lookup[label])


def ask(label, question):
    if not label:
        return "<p>Select a patient first.</p>", "Select a patient first.", "", None, None
    answer, answer_document, extracted, retrieved, visualization = answer_question(choice_lookup[label], question)
    return visualization, answer, answer_document, extracted, retrieved


with gr.Blocks(title="PrivateMD") as demo:
    gr.HTML(BACKGROUND_HTML)
    with gr.Column(elem_classes=["pm-shell"]):
        with gr.Row(elem_classes=["pm-topbar"]):
            patient = gr.Dropdown(
                choices=choice_labels,
                value=default_choice,
                label="Patient record",
                filterable=True,
                scale=5,
            )
            load = gr.Button("Analyze", variant="secondary", scale=1)

        gr.HTML(
            """
            <section class="pm-hero">
              <p class="pm-kicker">PrivateMD</p>
              <h1 class="pm-title">What should we inspect in this chart?</h1>
              <p class="pm-subtitle">Local LangExtract over synthetic FHIR, imaging, and genomics records. Built for clinician review, not autonomous diagnosis.</p>
            </section>
            """
        )

        with gr.Column(elem_classes=["pm-prompt-card"]):
            with gr.Row(elem_classes=["pm-prompt-row"]):
                question = gr.Textbox(
                    label="Ask about this patient",
                    placeholder="Ask about age, anticoagulation, recent medications, stroke history...",
                    lines=1,
                    scale=12,
                    elem_id="pm-question",
                )
                ask_button = gr.Button("Ask", variant="primary", scale=1, elem_id="pm-ask")

        gr.HTML(
            """
            <div class="pm-chips">
              <span class="pm-chip">Demographics</span>
              <span class="pm-chip">Medication status</span>
              <span class="pm-chip">Stroke history</span>
              <span class="pm-chip">Source highlights</span>
            </div>
            """
        )

        with gr.Column(elem_classes=["pm-results"]):
            visualization = gr.HTML(value=EMPTY_VISUALIZATION, elem_classes=["pm-visual"])
            answer = gr.Markdown(elem_classes=["pm-answer"])
            with gr.Accordion("LangExtract Source Trace", open=False, elem_classes=["pm-context"]):
                answer_document = gr.Textbox(
                    label="Original chart source document sent to LangExtract",
                    lines=10,
                    interactive=False,
                )
                extracted = gr.Dataframe(
                    label="LangExtract source entities and relations",
                    wrap=True,
                    interactive=False,
                )
                retrieved = gr.Dataframe(
                    label="Grounding evidence",
                    wrap=True,
                    interactive=False,
                )
            with gr.Accordion("Patient Context", open=False, elem_classes=["pm-context"]):
                snapshot = gr.Markdown(elem_classes=["metric-band"])
                timeline = gr.Dataframe(
                    label="Longitudinal patient journey",
                    wrap=True,
                    interactive=False,
                )
                opportunities = gr.Dataframe(
                    label="Rule-based review prompts",
                    wrap=True,
                    interactive=False,
                )
                sources = gr.Markdown()

    load.click(select_patient, inputs=patient, outputs=[snapshot, timeline, opportunities, sources])
    patient.change(select_patient, inputs=patient, outputs=[snapshot, timeline, opportunities, sources])
    ask_button.click(ask, inputs=[patient, question], outputs=[visualization, answer, answer_document, extracted, retrieved])
    demo.load(select_patient, inputs=patient, outputs=[snapshot, timeline, opportunities, sources])


if __name__ == "__main__":
    demo.launch(css=CSS, js=MESH_JS)
