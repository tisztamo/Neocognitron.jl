# Neocognitron.jl

Neocognitron by ChatGPT

The main goal here is to explore the limitations
when developing with ChatGPT (GPT-4)
and find workarounds.

initial session: https://chat.openai.com/share/930b0668-0511-409b-92bd-c655cd73e286

## Findings

 - ChatGPT starts to loose its oversight and generates incoherent code when the full project is ~100 lines.
 - Re-Showing previously generated files in the prompt is not enough to get it back on track
 - Maybe some history-compressing is at work, because it can answer broad questions about the chat history