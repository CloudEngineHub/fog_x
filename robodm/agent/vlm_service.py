"""
Shared Vision-Language Model service for RoboDM Agent.

Provides a singleton VLM instance that can be shared across multiple components
to avoid redundant model loading and improve batch inference efficiency.
"""

import threading
from typing import Union, Optional, List
import numpy as np
import base64
import io

try:
    from PIL import Image
except ImportError:
    class Image:
        @staticmethod
        def fromarray(array, mode=None):
            return MockImage()

class MockImage:
    def save(self, buffer, format=None):
        buffer.write(b"mock_image_data")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


class VLMService:
    """Singleton vision-language model service."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._client = None
            self._model = None
            self._config = {}
            self._initialized = True
    
    def initialize(self, 
                   model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
                   temperature: float = 0.1,
                   max_tokens: int = 256,
                   base_url: str = "http://localhost:30000/v1",
                   api_key: str = "EMPTY",
                   **kwargs):
        """Initialize the VLM service with OpenAI client configuration."""
        if self._client is not None and self._model == model:
            return  # Already initialized with same model
            
        self._model = model
        self._config = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }
        
        if OPENAI_AVAILABLE:
            try:
                self._client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )
                
                # Test connection with a simple request
                try:
                    self._client.models.list()
                except Exception as e:
                    print(f"Failed to connect to SGLang server ({e}), falling back to mock VLM")
                    self._client = None
                    
            except Exception as e:
                print(f"Failed to initialize OpenAI client ({e}), falling back to mock VLM")
                self._client = None
        else:
            print("OpenAI client not available, using mock VLM")
            self._client = None
    
    def get_client(self):
        """Get the OpenAI client instance."""
        if self._client is None and OPENAI_AVAILABLE:
            # Auto-initialize with defaults if not done
            self.initialize()
        return self._client
    
    def _convert_to_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Convert image to PIL Image."""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            if len(image.shape) == 3 and image.shape[2] == 3:
                return Image.fromarray(image, mode="RGB")
            elif len(image.shape) == 3 and image.shape[2] == 4:
                return Image.fromarray(image, mode="RGBA")
            elif len(image.shape) == 2:
                return Image.fromarray(image, mode="L")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        elif isinstance(image, Image.Image):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _encode_image_to_base64(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Encode image to base64 string for OpenAI API."""
        pil_image = self._convert_to_pil(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_image(self, frame: Union[np.ndarray, Image.Image], prompt: str) -> str:
        """Analyze image with vision-language model."""
        if not OPENAI_AVAILABLE or self._client is None:
            return f"Mock VLM response for: {prompt}"
        
        try:
            client = self.get_client()
            image_base64 = self._encode_image_to_base64(frame)
            
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self._config.get('max_tokens', 256),
                temperature=self._config.get('temperature', 0.1)
            )
            
            # Check if response content is None
            content = response.choices[0].message.content
            if content is None:
                return f"Mock VLM response for: {prompt} (model returned None content)"
            
            return content.strip()
            
        except Exception as e:
            return f"Error in VLM analysis: {str(e)}"

    def analyze_images(self, frames: List[Union[np.ndarray, Image.Image]], prompt: str) -> str:
        """Analyze multiple images together with a single prompt."""
        if not OPENAI_AVAILABLE or self._client is None:
            return f"Mock VLM response for multi-image prompt: {prompt}"

        try:
            client = self.get_client()

            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            for frame in frames:
                image_base64 = self._encode_image_to_base64(frame)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                )

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=self._config.get('max_tokens', 256),
                temperature=self._config.get('temperature', 0.1)
            )

            content_text = response.choices[0].message.content
            if content_text is None:
                return f"Mock VLM response for multi-image prompt: {prompt} (model returned None content)"

            return content_text.strip()

        except Exception as e:
            return f"Error in multi-image VLM analysis: {str(e)}"
    
    def generate_code(self, prompt: str) -> str:
        """Generate code using the language model."""
        if not OPENAI_AVAILABLE or self._client is None:
            return "    # Mock code generation - OpenAI client not available\n    return True"
        
        try:
            client = self.get_client()
            
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.1,
                stop=["# End of function"]  # Removed "```" as it was causing premature stopping
            )
            
            print(f"Generated code for prompt: {prompt} -> {response.choices}")
            
            # Check if response content is None
            content = response.choices[0].message.content
            if content is None:
                print("Warning: Model returned None content, using fallback")
                return "    # Fallback code - model returned empty response\n    return trajectory.get('is_success_labeled', False)"
            
            return content.strip()
            
        except Exception as e:
            print(f"Error in code generation: {str(e)}")
            return "    # Error in code generation - using fallback\n    return trajectory.get('is_success_labeled', False)"
        


# Global service instance
vlm_service = VLMService()


def get_vlm_service() -> VLMService:
    """Get the global VLM service instance."""
    return vlm_service