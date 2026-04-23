import ollama
import json

def generate_prompt(differences, rpm):
    """
    Construct a prompt for the LLM based on extracted differences.
    """
    f_1x = round(rpm / 60.0, 2)
    f_2x = round(2 * f_1x, 2)
    f_3x = round(3 * f_1x, 2)
    
    prompt = (
        f"You are an expert vibration analyst specializing in intelligent fault diagnosis for rotating machinery.\n"
        f"A machine is operating at {rpm} RPM. The fundamental rotational frequency (1X) is {f_1x} Hz.\n"
        f"Harmonics are: 2X = {f_2x} Hz, 3X = {f_3x} Hz.\n\n"
        f"Compared to a healthy baseline, the current operation shows the following abnormal frequency peaks in the FFT analysis:\n"
        f"{json.dumps(differences, indent=2)}\n\n"
        f"Based on these findings, please identify the most likely condition or fault.\n"
        f"Keep your diagnostic reasoning concise and point out whether there is no fault detected, or if there is an imbalance (typified by strong 1X), "
        f"misalignment (strong 2X or 3X), or a high-frequency specific bearing fault.\n"
        f"Provide your final determination at the end."
    )
    return prompt

def generate_vision_prompt(rpm):
    """
    Construct a prompt for the multimodal LLM based on the FFT image.
    """
    f_1x = round(rpm / 60.0, 2)
    f_2x = round(2 * f_1x, 2)
    f_3x = round(3 * f_1x, 2)
    
    prompt = (
        f"look at this image. do you think there is a fault in the machine or not. Its oper"
    )
    return prompt

def diagnose_fault_with_llm(differences, rpm=1800, model_name="gemma4", image_path=None):
    """
    Send the prompt and/or image to Ollama using the requested gemma4 model.
    """
    if image_path:
        prompt = generate_vision_prompt(rpm)
        message = {
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }
    else:
        prompt = generate_prompt(differences, rpm)
        message = {
            'role': 'user',
            'content': prompt,
        }
    
    try:
        response = ollama.chat(model=model_name, messages=[message])
        return response['message']['content']
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}\nPlease ensure standard ollama is running, '{model_name}' is installed, and it supports multimodal/vision tasks."

if __name__ == "__main__":
    # Test block
    dummy_diffs = [
        {"freq": 30.0, "mag": 6.2, "status": "new peak"}
    ]
    print(diagnose_fault_with_llm(dummy_diffs, rpm=1800))
