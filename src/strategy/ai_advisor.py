"""Optional LLM layer — never required for the agent to run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

from strategy.types import Signal, Side


@dataclass
class AiVerdict:
    confidence_multiplier: float
    veto: bool
    note: str


class AiAdvisor:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def review(self, symbol: str, base: Signal, context: pd.DataFrame) -> AiVerdict:
        if not self.enabled:
            return AiVerdict(1.0, False, "rules_only")

        last = context.iloc[-1]
        summary = {
            "symbol": symbol,
            "proposed": base.side.value,
            "rule_confidence": base.confidence,
            "close": float(last["close"]),
            "rsi": float(last["rsi"]) if not pd.isna(last["rsi"]) else None,
            "atr_pct": float(last["atr"] / last["close"]) if last["close"] else None,
            "return_24h": float(context["close"].pct_change(24).iloc[-1])
            if len(context) > 24
            else None,
        }
        prompt = (
            "You are a conservative crypto trading risk reviewer. "
            "Given JSON market snapshot and a proposed direction, respond ONLY with JSON: "
            '{"confidence_multiplier": 0.0-1.5, "veto": true/false, "note": "short reason"}. '
            "Veto if volatility is extreme or direction fights strong trend. "
            f"Input: {json.dumps(summary)}"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=120,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            mult = float(data.get("confidence_multiplier", 1.0))
            veto = bool(data.get("veto", False))
            note = str(data.get("note", "ok"))[:120]
            mult = max(0.0, min(1.5, mult))
            if base.side == Side.SHORT and mult > 1.2:
                mult = 1.2
            return AiVerdict(mult, veto, note)
        except Exception as exc:  # noqa: BLE001 — fallback to rules
            return AiVerdict(1.0, False, f"ai_error:{type(exc).__name__}")
