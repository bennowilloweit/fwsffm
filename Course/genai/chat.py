import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

class ChatSession:
    """ stores a single scalar value and its gradient """
    
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

    def start(self, system_instruction, model):
        self.model = GenerativeModel(
            model,
            system_instruction=[system_instruction]
        )
        self.chatSession = self.model.start_chat()

    def sendMessage(self, message, generation_config=None, safety_settings=None):
        if generation_config is None:
            generation_config = self.default_generation_config
        if safety_settings is None:
            safety_settings = self.default_safety_settings
        response = self.chatSession.send_message(content=message, generation_config=generation_config, safety_settings=safety_settings)
        return response

    def chat(self, message, generation_config=None, safety_settings=None):
        response = self.sendMessage(message, generation_config=generation_config, safety_settings=safety_settings)
        print(response.text)

