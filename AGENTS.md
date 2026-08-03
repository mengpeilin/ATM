## Project Instructions

Always begin every response by addressing the user as **Peilin**. This applies without exception.

Do not write backward-compatibility code unless Peilin explicitly requests it. Do not add compatibility layers, preserve legacy interfaces, or implement fallbacks for deprecated behavior. Assume the current technology stack and prefer clean, modern implementations.

All dataset imagery is RGB and must remain RGB throughout the pipeline. Do not add automatic BGR/RGB conversions, including `cv2.BGR2RGB` or `cv2.RGB2BGR`, unless Peilin explicitly specifies one.

File modification should be restricted to project folder. Files outside the folder can only be read but not be modified or deleted, unless I ask you to.
