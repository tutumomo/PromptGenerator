import streamlit as st
from openai import OpenAI  # Added import for OpenAI

def main():
    st.title("Prompt Generator")
    
    # Sidebar for OpenAI API key input
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    # Model selection
    model_options = ["gpt-3.5-turbo", "gpt-4o"]
    selected_model = st.sidebar.selectbox("Choose a model:", model_options, index=1)  # Default to gpt-4o

    # Parameter presets
    presets = {
        "Creative": {
            "temperature": 0.8,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "max_tokens": 512,
        },
        "Balanced": {
            "temperature": 0.5,
            "top_p": 0.85,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "max_tokens": 512,
        },
        "Precise": {
            "temperature": 0.2,
            "top_p": 0.75,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "max_tokens": 512,
        },
    }

    # Select box for choosing parameter preset
    selected_preset = st.sidebar.selectbox("Choose a parameter preset:", list(presets.keys()), index=1)  # Default to Balanced

    # Get parameters based on selected preset
    parameters = presets[selected_preset]
    
    # Display parameters and allow tuning
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=parameters["temperature"], step=0.1)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=parameters["top_p"], step=0.1)
    max_completion_tokens = st.sidebar.number_input("Max Completion Tokens", min_value=1, max_value=2048, value=parameters["max_tokens"])
    presence_penalty = st.sidebar.slider("Presence Penalty", min_value=0.0, max_value=1.0, value=parameters["presence_penalty"], step=0.1)
    frequency_penalty = st.sidebar.slider("Frequency Penalty", min_value=0.0, max_value=1.0, value=parameters["frequency_penalty"], step=0.1)

    # Initialize OpenAI client with the provided API key if available
    client = None
    if api_key:
        client = OpenAI(api_key=api_key)  # Pass the API key to the client

        META_PROMPT = """
        Given a task description or existing prompt, produce a detailed system prompt to guide a language model in completing the task effectively.

        # Guidelines

        - Understand the Task: Grasp the main objective, goals, requirements, constraints, and expected output.
        - Minimal Changes: If an existing prompt is provided, improve it only if it's simple. For complex prompts, enhance clarity and add missing elements without altering the original structure.
        - Reasoning Before Conclusions**: Encourage reasoning steps before any conclusions are reached. ATTENTION! If the user provides examples where the reasoning happens afterward, REVERSE the order! NEVER START EXAMPLES WITH CONCLUSIONS!
            - Reasoning Order: Call out reasoning portions of the prompt and conclusion parts (specific fields by name). For each, determine the ORDER in which this is done, and whether it needs to be reversed.
            - Conclusion, classifications, or results should ALWAYS appear last.
        - Examples: Include high-quality examples if helpful, using placeholders [in brackets] for complex elements.
           - What kinds of examples may need to be included, how many, and whether they are complex enough to benefit from placeholders.
        - Clarity and Conciseness: Use clear, specific language. Avoid unnecessary instructions or bland statements.
        - Formatting: Use markdown features for readability. DO NOT USE ``` CODE BLOCKS UNLESS SPECIFICALLY REQUESTED.
        - Preserve User Content: If the input task or prompt includes extensive guidelines or examples, preserve them entirely, or as closely as possible. If they are vague, consider breaking down into sub-steps. Keep any details, guidelines, examples, variables, or placeholders provided by the user.
        - Constants: DO include constants in the prompt, as they are not susceptible to prompt injection. Such as guides, rubrics, and examples.
        - Output Format: Explicitly the most appropriate output format, in detail. This should include length and syntax (e.g. short sentence, paragraph, JSON, etc.)
            - For tasks outputting well-defined or structured data (classification, JSON, etc.) bias toward outputting a JSON.
            - JSON should never be wrapped in code blocks (```) unless explicitly requested.

        The final prompt you output should adhere to the following structure below. Do not include any additional commentary, only output the completed system prompt. SPECIFICALLY, do not include any additional messages at the start or end of the prompt. (e.g. no "---")

        [Concise instruction describing the task - this should be the first line in the prompt, no section header]

        [Additional details as needed.]

        [Optional sections with headings or bullet points for detailed steps.]

        # Steps [optional]

        [optional: a detailed breakdown of the steps necessary to accomplish the task]

        # Output Format

        [Specifically call out how the output should be formatted, be it response length, structure e.g. JSON, markdown, etc]

        # Examples [optional]

        [Optional: 1-3 well-defined examples with placeholders if necessary. Clearly mark where examples start and end, and what the input and output are. User placeholders as necessary.]
        [If the examples are shorter than what a realistic example is expected to be, make a reference with () explaining how real examples should be longer / shorter / different. AND USE PLACEHOLDERS! ]

        # Notes [optional]

        [optional: edge cases, details, and an area to call or repeat out specific important considerations]
        """.strip()

        def generate_prompt(task_or_prompt: str):
            completion = client.chat.completions.create(
                model=selected_model,  # Use the selected model
                messages=[
                    {
                        "role": "system",
                        "content": META_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt,
                    },
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_completion_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            return completion.choices[0].message.content

        # Add a text box for user input
    original_prompt = st.text_area("Enter your original prompt:")

    # Example of how to call the generate_prompt function and display the result
    if st.button("Generate Prompt"):
        if original_prompt:
            if client:  # Check if the client is initialized
                generated_prompt = generate_prompt(original_prompt)
                st.text_area("Generated Prompt:", value=generated_prompt, height=1200)
            else:
                st.error("Please enter a valid OpenAI API Key to generate a prompt.")
        else:
            st.error("Please enter an original prompt.")

if __name__ == "__main__":
    main()
