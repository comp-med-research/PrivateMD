import gradio as gr

from private_md import analyze_patient, answer_question, extract_medications, patient_choices


CSS = """
:root {
  --brand: #0f766e;
  --ink: #172033;
  --muted: #5b6475;
  --line: #dbe3ea;
  --surface: #f7faf9;
}
.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
}
.app-header {
  border-bottom: 1px solid var(--line);
  padding: 18px 0 14px;
  margin-bottom: 12px;
}
.app-header h1 {
  font-size: 30px;
  line-height: 1.1;
  margin: 0 0 6px;
  letter-spacing: 0;
}
.app-header p {
  color: var(--muted);
  margin: 0;
  max-width: 880px;
}
.metric-band {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 14px;
}
button.primary {
  background: var(--brand) !important;
  border-color: var(--brand) !important;
}
"""


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
        return "Select a patient first.", None
    return answer_question(choice_lookup[label], question)


def run_medication_extract(label):
    if not label:
        return "Select a patient first.", "", None, None, None, None
    return extract_medications(choice_lookup[label])


with gr.Blocks(title="PrivateMD") as demo:
    gr.HTML(
        """
        <div class="app-header">
          <h1>PrivateMD</h1>
          <p>Evidence-grounded medical copilot for local synthetic FHIR, imaging metadata, and genomics records. Built for clinician review, not autonomous diagnosis.</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            patient = gr.Dropdown(
                choices=choice_labels,
                value=default_choice,
                label="Patient record",
                filterable=True,
            )
            load = gr.Button("Analyze Local Record", variant="primary")
            gr.Markdown(
                "All analysis is computed from files inside this Space. Demo records are synthetic Coherent/Synthea patients."
            )
        with gr.Column(scale=2):
            snapshot = gr.Markdown(elem_classes=["metric-band"])

    with gr.Tabs():
        with gr.Tab("Journey"):
            timeline = gr.Dataframe(
                label="Longitudinal patient journey",
                wrap=True,
                interactive=False,
            )
        with gr.Tab("Treatment Opportunities"):
            opportunities = gr.Dataframe(
                label="Rule-based, evidence-cited review prompts",
                wrap=True,
                interactive=False,
            )
            sources = gr.Markdown()
        with gr.Tab("LangExtract Medications"):
            med_extract = gr.Button("Extract Medication Relationships", variant="primary")
            med_summary = gr.Markdown()
            med_narrative = gr.Textbox(
                label="FHIR medication narrative sent to LangExtract",
                lines=8,
                interactive=False,
            )
            med_fhir = gr.Dataframe(
                label="FHIR MedicationRequest reference",
                wrap=True,
                interactive=False,
            )
            med_groups = gr.Dataframe(
                label="Grouped medications and attributes",
                wrap=True,
                interactive=False,
            )
            med_spans = gr.Dataframe(
                label="Source-aligned extraction spans",
                wrap=True,
                interactive=False,
            )
            med_visualization = gr.File(label="LangExtract visualization HTML")
        with gr.Tab("Copilot (RAG Experimental)"):
            question = gr.Textbox(
                label="Ask about this patient",
                placeholder="e.g. What should I review before changing anticoagulation?",
                lines=2,
            )
            ask_button = gr.Button("Search Local Evidence", variant="primary")
            answer = gr.Markdown()
            retrieved = gr.Dataframe(
                label="Retrieved evidence chunks",
                wrap=True,
                interactive=False,
            )

    load.click(select_patient, inputs=patient, outputs=[snapshot, timeline, opportunities, sources])
    patient.change(select_patient, inputs=patient, outputs=[snapshot, timeline, opportunities, sources])
    med_extract.click(
        run_medication_extract,
        inputs=patient,
        outputs=[med_summary, med_narrative, med_fhir, med_groups, med_spans, med_visualization],
    )
    ask_button.click(ask, inputs=[patient, question], outputs=[answer, retrieved])
    demo.load(select_patient, inputs=patient, outputs=[snapshot, timeline, opportunities, sources])


if __name__ == "__main__":
    demo.launch(css=CSS)
