#!/bin/bash
# Patch vLLM for LightOnOCR support with transformers 5.0

echo "Upgrading transformers to 5.0..."
pip install --quiet --upgrade transformers

echo "Patching vLLM mistral3.py for LightOnOcrProcessor..."
python3 -c '
with open("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mistral3.py", "r") as f:
    content = f.read()

# Add LightOnOcrProcessor import
old_import = "from transformers.models.pixtral import PixtralProcessor"
new_import = """from transformers.models.pixtral import PixtralProcessor
try:
    from transformers import LightOnOcrProcessor
except ImportError:
    LightOnOcrProcessor = None"""

if "LightOnOcrProcessor" not in content:
    content = content.replace(old_import, new_import)

# Patch get_hf_processor method
old_method = """    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)"""

new_method = """    def get_hf_processor(self, **kwargs: object):
        # Use LightOnOcrProcessor for LightOnOCR models
        if LightOnOcrProcessor is not None and "lightonocr" in str(self.ctx.model_config.model).lower():
            return self.ctx.get_hf_processor(LightOnOcrProcessor, **kwargs)
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)"""

if "LightOnOcrProcessor for LightOnOCR" not in content:
    content = content.replace(old_method, new_method)

with open("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mistral3.py", "w") as f:
    f.write(content)
'

echo "Patches applied!"

# Run the original vLLM command
exec vllm serve "$@"
