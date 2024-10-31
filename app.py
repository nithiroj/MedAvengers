import gradio as gr

from dotenv import load_dotenv

load_dotenv()

from avengers import avengers_graph as graph

thread_id = 0
thread = {"configurable": {"thread_id": str(thread_id)}}

report_md = '# ---REPORT---'

analysts = []

def stream_graph(topic, max_analysts):
    global analysts, thread_id, thread
    # Run the graph until the first interruption

    #  New thread evertimes getting start
    thread_id += 1
    thread = {"configurable": {"thread_id": str(thread_id)}}

    inputs = {"topic": topic, "max_analysts": max_analysts}
    partial_message = ""
    for event in graph.stream(inputs,
                              thread,
                              stream_mode="values"):
        
        analysts = event.get('analysts', '')
        if analysts:
          for i, analyst in enumerate(analysts):
              partial_message += (f"Name: {analyst['name']}\n")
              partial_message += (f"Affiliation: {analyst['affiliation']}\n")
              partial_message += (f"Role: {analyst['role']}\n")
              partial_message += (f"Description: {analyst['description']}\n")
              partial_message += ("-" * 50 + '\n')
              if i == len(analysts) - 1 : partial_message += ("#" * 100)
              yield partial_message
        print(analysts)
    return

def update_state(feedback=None):
    global analysts
    partial_message = ""
    
    if feedback:
        graph.update_state(
            thread,
            {"human_analyst_feedback": feedback},
            as_node="human_feedback"
        )
        # Adjust Team
        for event in graph.stream(None,
                                  thread,
                                  stream_mode="values"):
            
            analysts = event.get('analysts', '')
            if analysts:
              for i, analyst in enumerate(analysts):
                  partial_message += (f"Name: {analyst['name']}\n")
                  partial_message += (f"Affiliation: {analyst['affiliation']}\n")
                  partial_message += (f"Role: {analyst['role']}\n")
                  partial_message += (f"Description: {analyst['description']}\n")
                  partial_message += ("-" * 50 + '\n')
                  if i == len(analysts) - 1 : partial_message += ("#" * 100) 
                  yield partial_message
            print(analysts)        

    return

def continue_state():
    global report_md
    # Continue
    partial_message = ""

    graph.update_state(
            thread,
            {"human_analyst_feedback": None},
            as_node="human_feedback"
        )

    for event in graph.stream(None, thread, stream_mode="updates"):
        partial_message += ("--Node--\n")
        node_name = next(iter(event.keys()))
        partial_message += (node_name + '\n')
        yield partial_message

    print(partial_message)

    final_state = graph.get_state(thread)
    report_md = final_state.values.get('final_report')

    return

def update_ui():
    return gr.Textbox(thread_id), gr.Row(visible=True)

def display_report():
    return gr.Markdown(report_md, visible=True)

dummy_topic = "Exploring Mental Health Stigma Among Youth: Causes, Effects, and Reduction Strategies"
dummy_adjustment = "Add in the profesionals in public media sector"

with gr.Blocks(theme=gr.themes.Default(spacing_size='sm',text_size="md")) as demo:
    
    gr.Markdown("# MedAvengers")
    gr.Markdown("## (1) Inputs")

    with gr.Row():
        topic = gr.Textbox(label="Medical Research Topic", value=dummy_topic, interactive=True)
        max_analysts = gr.Dropdown(
            [1, 2, 3],
            value=3,
            label="Max Number of Analysts", 
            info="Number of analysts for the research",
            interactive=True
            )
        thread_dsp = gr.Textbox(label="Thread", scale=0,min_width=80)
        start_btn = gr.Button("Start", scale=0,min_width=80)
    
    live = gr.Textbox(label="Live Agent Output", lines=5)

    gr.Markdown("## (2) Adjust Analyst Members")

    with gr.Row(visible=False) as adjusment:
        feedback = gr.Textbox(label="You can adjust your analyst team memebers here", value=dummy_adjustment, interactive=True)
    
        adjust_btn = gr.Button("Adjust", scale=0,min_width=80)
        cont_btn = gr.Button("Continue", scale=0,min_width=100)

    gr.Markdown("## (3) REPORT")
    report_dsp = gr.Markdown(report_md, visible=False)

    start_btn.click(fn=stream_graph, 
                    inputs=[topic,
                            max_analysts],
                    outputs=[live]
                    ).then(fn=update_ui, inputs=None, outputs=[thread_dsp, adjusment])
    
    adjust_btn.click(fn=update_state, 
                    inputs=[feedback],
                    outputs=[live]
                    )
    
    cont_btn.click(fn=continue_state, 
                    outputs=[live]
                    ).then(fn=display_report, inputs=None, outputs=[report_dsp])
        
demo.launch()