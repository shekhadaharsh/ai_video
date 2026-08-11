"""
cost_tracker.py — Track Gemini API token usage and calculate cost in USD + INR.

Pricing reference (gemini-3.1-pro-preview):
  Input:  $2.00 per 1 million tokens  (<=200K context)
  Output: $12.00 per 1 million tokens (<=200K context)

  Long context (>200K tokens) doubles the price:
  Input:  $4.00 / 1M | Output: $18.00 / 1M
"""

import logging
from modules.utils import save_json

logger = logging.getLogger(__name__)

# ── Pricing Constants ──────────────────────────────────────────────────────────
MODEL_NAME = "gemini-3.1-pro-preview"  # Gemini 3.1 Pro

# Standard context pricing (<=200K tokens)
PRICE_INPUT_PER_1M  = 2.00   # USD per million input tokens
PRICE_OUTPUT_PER_1M = 12.00  # USD per million output tokens

# Long context pricing (>200K tokens)
PRICE_INPUT_LONG_PER_1M  = 4.00
PRICE_OUTPUT_LONG_PER_1M = 18.00

# Context threshold for long-context pricing
LONG_CONTEXT_THRESHOLD = 200_000

# INR conversion rate (approximate)
USD_TO_INR = 84.0


# ── Main Cost Calculator ───────────────────────────────────────────────────────
def calculate_cost(usage_metadata) -> dict:
    """
    Calculate cost from Gemini API usage_metadata object.
    Handles both old (google.generativeai) and new (google.genai) SDK formats.

    Args:
        usage_metadata: response.usage_metadata from Gemini API

    Returns:
        dict with token counts, USD costs, and INR cost
    """
    # New google-genai SDK uses prompt_token_count / candidates_token_count
    # Fallback to total_token_count if individual counts not available
    input_tokens = (
        getattr(usage_metadata, "prompt_token_count", None)
        or getattr(usage_metadata, "input_token_count", None)
        or 0
    )
    output_tokens = (
        getattr(usage_metadata, "candidates_token_count", None)
        or getattr(usage_metadata, "output_token_count", None)
        or 0
    )

    # Determine pricing tier based on input tokens
    is_long_context = input_tokens > LONG_CONTEXT_THRESHOLD
    price_in  = PRICE_INPUT_LONG_PER_1M  if is_long_context else PRICE_INPUT_PER_1M
    price_out = PRICE_OUTPUT_LONG_PER_1M if is_long_context else PRICE_OUTPUT_PER_1M

    input_cost_usd  = (input_tokens  / 1_000_000) * price_in
    output_cost_usd = (output_tokens / 1_000_000) * price_out
    total_cost_usd  = input_cost_usd + output_cost_usd
    total_cost_inr  = total_cost_usd * USD_TO_INR

    report = {
        "model":              MODEL_NAME,
        "pricing_tier":       "long_context" if is_long_context else "standard",
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "total_tokens":       input_tokens + output_tokens,
        "input_cost_usd":     round(input_cost_usd,  6),
        "output_cost_usd":    round(output_cost_usd, 6),
        "total_cost_usd":     round(total_cost_usd,  6),
        "total_cost_inr":     round(total_cost_inr,  4),
        "usd_to_inr_rate":    USD_TO_INR,
    }


def calculate_combined_cost(usage_metadata_list: list) -> dict:
    """
    Combine token usage and cost across multiple Gemini API calls
    (e.g. Pass 1 = Task A+B+D, Pass 2 = Task C) into one report.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    for usage_metadata in usage_metadata_list:
        single_report = calculate_cost(usage_metadata)
        total_input_tokens += single_report["input_tokens"]
        total_output_tokens += single_report["output_tokens"]
        total_cost_usd += single_report["total_cost_usd"]

    total_cost_inr = total_cost_usd * USD_TO_INR

    # Check if context threshold was breached by either call
    is_long_context = total_input_tokens > LONG_CONTEXT_THRESHOLD

    return {
        "model":              MODEL_NAME,
        "pricing_tier":       "long_context" if is_long_context else "standard",
        "calls_combined":     len(usage_metadata_list),
        "input_tokens":       total_input_tokens,
        "output_tokens":      total_output_tokens,
        "total_tokens":       total_input_tokens + total_output_tokens,
        "input_cost_usd":     round(total_cost_usd * (total_input_tokens / max(1, total_input_tokens + total_output_tokens)), 6),
        "output_cost_usd":    round(total_cost_usd * (total_output_tokens / max(1, total_input_tokens + total_output_tokens)), 6),
        "total_cost_usd":     round(total_cost_usd, 6),
        "total_cost_inr":     round(total_cost_inr, 4),
        "usd_to_inr_rate":    USD_TO_INR,
    }


    logger.info(
        f"Cost: {input_tokens} input + {output_tokens} output tokens = "
        f"${total_cost_usd:.4f} USD (₹{total_cost_inr:.2f})"
    )
    return report


def save_cost_report(cost_report: dict, cost_report_path: str) -> None:
    """Save cost report to JSON file."""
    save_json(cost_report, cost_report_path)
    logger.info(f"Cost report saved: {cost_report_path}")


def format_cost_display(cost_report: dict) -> dict:
    """
    Format cost report values for nice display in Streamlit UI.
    Returns dict with formatted strings.
    """
    return {
        "model":         cost_report.get("model", "N/A"),
        "pricing_tier":  cost_report.get("pricing_tier", "standard").replace("_", " ").title(),
        "input_tokens":  f"{cost_report.get('input_tokens', 0):,}",
        "output_tokens": f"{cost_report.get('output_tokens', 0):,}",
        "total_tokens":  f"{cost_report.get('total_tokens', 0):,}",
        "input_cost":    f"${cost_report.get('input_cost_usd', 0):.4f}",
        "output_cost":   f"${cost_report.get('output_cost_usd', 0):.4f}",
        "total_usd":     f"${cost_report.get('total_cost_usd', 0):.4f}",
        "total_inr":     f"₹{cost_report.get('total_cost_inr', 0):.2f}",
    }
