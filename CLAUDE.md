# Project Instructions

## 1. Addressing Rule

Always begin every response by addressing me as **Peilin**.

Example:
> Peilin, here's what I found...

This rule applies to every reply without exception.

---

## 2. Compatibility Policy

Do **not** write backward-compatibility code unless I explicitly request it.

Specifically:
- Do not add compatibility layers for older APIs, frameworks, or versions.
- Do not preserve legacy interfaces.
- Do not implement fallbacks for deprecated behavior.
- Assume the project targets the current technology stack unless instructed otherwise.

Prefer clean, modern implementations over defensive compatibility code.

## 3. Color Channel

Everything in the dataset is in RGB, and we must maintain RGB format throughout the entire pipeline. Do not add automatic BGR/RGB conversions (such as OpenCV cv2.BGR2RGB or cv2.RGB2BGR transforms) unless explicitly specified.
