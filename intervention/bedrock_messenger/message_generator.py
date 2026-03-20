"""
intervention/bedrock_messenger/message_generator.py
──────────────────────────────────────────────────────────────────────────────
Generates empathetic, constraint-compliant outreach messages using
Amazon Bedrock (Claude). Falls back to template-based messages if
Bedrock is unavailable (for local development without AWS credentials).
 
Message constraints (from Sentinel design):
  - 1 sentence only
  - No word "risk" or "delinquency"
  - Clear YES/NO CTA
  - One emoji
  - Under 140 characters
  - Empathetic, not alarming tone
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import json
import re
from typing import Optional
 
from config.logging_config import get_logger
from config.settings import get_settings
 
logger = get_logger(__name__)
settings = get_settings()
 
# Template-based fallback messages (used when Bedrock is unavailable)
TEMPLATES = {
    "payment_holiday": {
        "salary_delayed":  "We noticed your salary came in late this month — need a 7-day EMI extension? Reply YES 🙏",
        "failed_debit":    "Your recent auto-payment didn't go through — want us to defer it to next week? Reply YES 🙏",
        "high_lending":    "We see you've been managing a lot this month — can we pause your EMI for 30 days? Reply YES 🙏",
        "default":         "It looks like you might need some breathing room this month — want a payment holiday? Reply YES 🙏",
    },
    "flexible_emi": {
        "salary_delayed":  "Salary delay affecting your EMI date? We can move it forward by 10 days for you — Reply YES 💙",
        "utility_delay":   "Need to spread your payments differently this month? We can offer a flexible EMI plan — Reply YES 💙",
        "default":         "Want to adjust your EMI schedule to fit this month better? We can help — Reply YES 💙",
    },
    "digital_nudge": {
        "default": "Quick tip: Setting up a monthly savings auto-transfer can help you always be EMI-ready 💡",
    },
}
 
PROHIBITED_WORDS = ["risk", "delinquency", "default", "overdue", "missed", "credit score", "alert"]
 
 
def _select_template(intervention_type: str, top_factor: str) -> str:
    """Select best-match template based on intervention type and top SHAP factor."""
    templates = TEMPLATES.get(intervention_type, TEMPLATES["digital_nudge"])
    if "salary" in top_factor.lower():
        return templates.get("salary_delayed", templates.get("default", ""))
    if "auto_debit" in top_factor.lower() or "failed" in top_factor.lower():
        return templates.get("failed_debit", templates.get("default", ""))
    if "lending" in top_factor.lower() or "upi" in top_factor.lower():
        return templates.get("high_lending", templates.get("default", ""))
    if "utility" in top_factor.lower():
        return templates.get("utility_delay", templates.get("default", ""))
    return templates.get("default", "We have a flexible option for you this month — Reply YES")
 
 
def _is_compliant(message: str) -> bool:
    """Validate message meets all constraints."""
    if len(message) > 140:
        return False
    msg_lower = message.lower()
    for word in PROHIBITED_WORDS:
        if word in msg_lower:
            return False
    return True
 
 
def generate_message(
    customer_id: str,
    intervention_type: str,
    top_factor: str,
    customer_segment: str = "mass_retail",
) -> str:
    """
    Generate an empathetic outreach message.
    Tries Bedrock first, falls back to template if unavailable.
    """
    # Try Bedrock (requires real AWS credentials)
    try:
        import boto3
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )
        prompt = f"""You are a banking assistant generating a single SMS message for a customer.
Context:
- Top financial stress signal: {top_factor}
- Offer type: {intervention_type.replace('_', ' ')}
- Customer segment: {customer_segment}
 
Rules (ALL must be followed):
1. Maximum 140 characters total
2. Exactly 1 sentence
3. Must include "Reply YES" as the call to action
4. Include exactly 1 emoji at the end
5. NEVER use any of these words: {', '.join(PROHIBITED_WORDS)}
6. Tone must be warm, empathetic, supportive — not alarming
 
Output only the SMS message text, nothing else."""
 
        response = bedrock.invoke_model(
            modelId=settings.bedrock_model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}],
            }),
            contentType="application/json",
        )
        raw = json.loads(response["body"].read())
        message = raw["content"][0]["text"].strip()
        # Remove any quotes the model may add
        message = message.strip("''")
 
        if _is_compliant(message):
            logger.info("Bedrock message generated", customer_id=customer_id, length=len(message))
            return message
        else:
            logger.warning("Bedrock message failed compliance check — using template", message=message)
    except Exception as e:
        logger.info("Bedrock unavailable, using template", reason=str(e)[:100])
 
    # Fallback: template-based message
    message = _select_template(intervention_type, top_factor)
    if not _is_compliant(message):
        message = "We have a flexible payment option for you this month — Reply YES 💙"
    return message