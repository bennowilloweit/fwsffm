import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

class LLM:
    
    def __init__(self):
        vertexai.init(
            project="gpt-2-393809",
            location="europe-west4",
            api_endpoint="europe-west4-aiplatform.googleapis.com"
        )
        
        self.default_generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
        
        self.default_safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]

    def generate(self, model, content, generation_config=None, safety_settings=None):
        if generation_config is None:
            generation_config = self.default_generation_config
        if safety_settings is None:
            safety_settings = self.default_safety_settings
        model_ref = GenerativeModel(model)
        responses = model_ref.generate_content(
        [content],
        generation_config=generation_config,
        safety_settings=safety_settings)
        return responses


