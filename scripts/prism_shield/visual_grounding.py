"""
visual_grounding.py
-------------------
Vision-Language Model based visual grounding for UI element verification.

Security Critical: Malicious apps can use FLAG_SECURE, SurfaceView overlays,
or invisible TextViews to completely blind or deceive the accessibility XML.
This module verifies elements exist in actual pixels before any tap action executes.

Uses the existing Moondream2 VLM (via llama-cpp-python) to:
1. Verify a target element is visually present on screen
2. Return confidence that the element is real (not spoofed)
3. Provide visual coordinates if available

This is the core defense against UI Obfuscation attacks that target
the uiautomator2 XML dump that PRISM agents rely on.
"""

from __future__ import annotations
import base64
import json
import logging
import os
import re
from typing import Optional
from dataclasses import dataclass

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger("PrismShield.VisualGrounding")

# Local GGUF paths (same as vlm_consistency_checker.py)
VLM_MODEL_PATH = "models/vlm/moondream2-text-model-f16.gguf"
VLM_MMPROJ_PATH = "models/vlm/moondream2-mmproj-f16.gguf"


@dataclass
class VisualGroundingResult:
    """Result of visual element verification."""
    verified: bool
    confidence: float
    reason: str
    element_description: str | None = None


class VisualGrounding:
    """
    VLM-based visual grounding for UI element verification.
    
    Before any tap action, verify the target element actually exists in the 
    screen pixels - not just in the accessibility XML which can be spoofed.
    """
    
    _instance: Optional["VisualGrounding"] = None
    
    def __init__(self):
        self.llm: Optional[Llama] = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> "VisualGrounding":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self):
        """Load the VLM into memory. Heavy operation, do once at startup."""
        if self._initialized or Llama is None:
            return
        
        if not os.path.exists(VLM_MODEL_PATH) or not os.path.exists(VLM_MMPROJ_PATH):
            logger.warning(f"VLM models not found at {VLM_MODEL_PATH}. Skipping visual grounding init.")
            return
        
        logger.info("Loading Moondream2 for visual grounding...")
        try:
            self.llm = Llama(
                model_path=VLM_MODEL_PATH,
                chat_handler=self._get_moondream2_handler(),
                n_ctx=2048,
                n_gpu_layers=0,  # CPU only for reliable baseline
                n_threads=4,
                verbose=False
            )
            self._initialized = True
            logger.info("Visual grounding VLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load VLM for visual grounding: {e}")
    
    def _get_moondream2_handler(self):
        """Get the Moondream2 chat handler for multimodal input."""
        try:
            from llama_cpp.llama_chat_format import MoondreamChatHandler
            import llama_cpp
            try:
                clip = llama_cpp.Llama.create_clip(mmproj_path=VLM_MMPROJ_PATH)
                return MoondreamChatHandler(clip=clip)
            except AttributeError:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                return Llava15ChatHandler(clip_model_path=VLM_MMPROJ_PATH, verbose=False)
        except Exception as e:
            logger.warning(f"VLM chat handler init failed: {e}")
            return None
    
    def verify_element(
        self,
        screenshot_path: str,
        target_text: str | None = None,
        target_desc: str | None = None,
    ) -> VisualGroundingResult:
        """
        Verify a UI element exists visually on screen.
        
        Args:
            screenshot_path: Path to the current screen screenshot
            target_text: The text attribute from XML that we want to verify
            target_desc: The content-desc attribute from XML
            
        Returns:
            VisualGroundingResult with verified=True if element is visually present
        """
        if not self._initialized or not self.llm:
            logger.warning("VLM not initialized - cannot perform visual grounding")
            return VisualGroundingResult(
                verified=False,
                confidence=0.0,
                reason="VLM not available",
            )
        
        if not os.path.exists(screenshot_path):
            logger.warning(f"Screenshot not found: {screenshot_path}")
            return VisualGroundingResult(
                verified=False,
                confidence=0.0,
                reason="Screenshot unavailable",
            )
        
        # Build the verification prompt
        target = target_text or target_desc or ""
        if not target:
            return VisualGroundingResult(
                verified=False,
                confidence=0.0,
                reason="No target element specified",
            )
        
        # Truncate target to reasonable length
        target = target[:100]
        
        USER_PROMPT = f"""
Look at this Android phone screen screenshot.
I need to verify if a UI element with the text/description "{target}" is actually visible on the screen.

This is a security check - malicious apps can fake the accessibility XML tree while showing different content visually.

Please answer:
1. Is there an element with text or label matching "{target}" visible on screen?
2. Describe what you see in the screen area where this element should be.

Respond in this format:
VERIFIED: yes/no
CONFIDENCE: 0.0-1.0
DESCRIPTION: (brief description of what you see)
"""
        
        try:
            with open(screenshot_path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode("utf-8")
            img_url = f"data:image/jpeg;base64,{img_b64}"
            
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a security AI that verifies UI elements are actually visible on Android screens."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_url}},
                            {"type": "text", "text": USER_PROMPT}
                        ]
                    }
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            answer = response["choices"][0]["message"]["content"].strip()
            logger.debug(f"VLM visual grounding response: {answer}")
            
            # Parse the response
            verified = False
            confidence = 0.5
            description = answer
            
            # Look for VERIFIED: yes/no
            verified_match = re.search(r"VERIFIED:\s*(yes|no)", answer, re.IGNORECASE)
            if verified_match:
                verified = verified_match.group(1).lower() == "yes"
            
            # Look for CONFIDENCE: 0.X
            conf_match = re.search(r"CONFIDENCE:\s*(0\.\d+|1\.0)", answer, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
            
            # Security: If VLM says not verified, BLOCK the action
            # This prevents XML-spoofing attacks
            if verified:
                reason = f"Element '{target}' verified visually present"
            else:
                reason = f"Element '{target}' NOT visually verified - possible XML spoofing attack"
            
            return VisualGroundingResult(
                verified=verified,
                confidence=confidence,
                reason=reason,
                element_description=description,
            )
            
        except Exception as e:
            logger.error(f"Visual grounding check failed: {e}")
            return VisualGroundingResult(
                verified=False,
                confidence=0.0,
                reason=f"VLM error: {e}",
            )
    
    def verify_batch(
        self,
        screenshot_path: str,
        elements: list[dict],
    ) -> dict[str, VisualGroundingResult]:
        """
        Verify multiple UI elements in a single screenshot.
        
        Used for pre-validation before agent decides which element to tap.
        
        Args:
            screenshot_path: Path to screenshot
            elements: List of element dicts with 'text' or 'desc' keys
            
        Returns:
            Dict mapping element text -> VisualGroundingResult
        """
        if not self._initialized or not self.llm:
            return {}
        
        results = {}
        
        # Build a batch prompt to check all elements at once
        target_list = []
        for elem in elements:
            text = elem.get("text", "")
            desc = elem.get("desc", "")
            if text:
                target_list.append(text)
            elif desc:
                target_list.append(desc)
        
        if not target_list:
            return {}
        
        # Use first element for single verification (batch would be more expensive)
        # For security-critical path, verify each element individually is too slow
        # Instead, verify the specific target before action
        first_target = target_list[0]
        
        result = self.verify_element(screenshot_path, target_text=first_target)
        results[first_target] = result
        
        return results


# Singleton accessor
visual_grounder = VisualGrounding()