import json
import os
import anthropic
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
print(sys.path[0])
from config import *


BACKTEST_FILE   = os.path.join(RESULTS_DIR, "backtest_summary.json")     

with open(BACKTEST_FILE, 'r') as f:
    backtest = json.load(f)

s = backtest["strategy_metrics"]
b = backtest["benchmark_metrics"]


comparisons = {
    "period":                   backtest["period"],
    "n_stocks":                 backtest["n_stocks"],

    # Returns
    "strategy_ann_return":      round(s["annualized_return"] * 100, 2),
    "benchmark_ann_return":     round(b["annualized_return"] * 100, 2),
    "return_outperformance":    round((s["annualized_return"] - b["annualized_return"]) * 100, 2),

    # Risk
    "strategy_volatility":      round(s["volatility"] * 100, 2),
    "benchmark_volatility":     round(b["volatility"] * 100, 2),
    "excess_volatility":        round((s["volatility"] - b["volatility"]) * 100, 2),

    # Risk-adjusted
    "strategy_sharpe":          round(s["sharpe_ratio"], 4),
    "benchmark_sharpe":         round(b["sharpe_ratio"], 4),
    "sharpe_difference":        round(s["sharpe_ratio"] - b["sharpe_ratio"], 4),
    "strategy_sortino":         round(s["sortino_ratio"], 4),
    "benchmark_sortino":        round(b["sortino_ratio"], 4),

    # Drawdown
    "strategy_max_drawdown":    round(s["max_drawdown"] * 100, 2),
    "benchmark_max_drawdown":   round(b["max_drawdown"] * 100, 2),
    "strategy_recovery_days":   s["max_drawdown_recovery_days"],
    "benchmark_recovery_days":  b["max_drawdown_recovery_days"],
    "faster_recovery_days":     b["max_drawdown_recovery_days"] - s["max_drawdown_recovery_days"],

    # Alpha / factor metrics
    "alpha":                    round(s["alpha_vs_benchmark"] * 100, 2),
    "beta":                     round(s["beta_vs_benchmark"], 4),
    "information_ratio":        round(s["information_ratio"], 4),
    "tracking_error":           round(s["tracking_error"] * 100, 2),
    "correlation":              round(s["correlation_vs_benchmark"], 4),
}

prompt = f"""
You are a quantitative investment analyst writing a performance attribution 
summary for a systematic momentum factor strategy. Be precise, cite specific 
numbers throughout, and maintain a balanced tone — acknowledge both strengths 
and limitations. Do not use bullet points. Write in flowing prose only.

Here are the strategy details and performance metrics vs the S&P 500 benchmark:

STRATEGY OVERVIEW:
- Universe: {comparisons['n_stocks']} S&P 500 stocks
- Backtest period: {comparisons['period']}
- Strategy type: Cross-sectional momentum with regime-based volatility filter
- Portfolio construction: Long-short quantile-based signal, monthly rebalancing

PERFORMANCE METRICS (Strategy vs Benchmark):
- Annualized return:    {comparisons['strategy_ann_return']}% vs {comparisons['benchmark_ann_return']}%  (outperformance: +{comparisons['return_outperformance']}%)
- Annualized volatility: {comparisons['strategy_volatility']}% vs {comparisons['benchmark_volatility']}%  (excess: +{comparisons['excess_volatility']}%)
- Sharpe ratio:        {comparisons['strategy_sharpe']} vs {comparisons['benchmark_sharpe']}  (difference: +{comparisons['sharpe_difference']})
- Sortino ratio:       {comparisons['strategy_sortino']} vs {comparisons['benchmark_sortino']}
- Max drawdown:        {comparisons['strategy_max_drawdown']}% vs {comparisons['benchmark_max_drawdown']}%
- Drawdown recovery:   {comparisons['strategy_recovery_days']} days vs {comparisons['benchmark_recovery_days']} days  ({comparisons['faster_recovery_days']} days faster)
- Alpha (annualized):  {comparisons['alpha']}%
- Beta:                {comparisons['beta']}
- Information ratio:   {comparisons['information_ratio']}
- Tracking error:      {comparisons['tracking_error']}%
- Correlation vs benchmark: {comparisons['correlation']}

Please write a 3-paragraph performance summary covering:
1. Overall return and risk-adjusted performance vs the benchmark — was the outperformance meaningful?
2. Drawdown behavior and recovery — how did the strategy hold up during stress periods?
3. Key risks, limitations, and what the metrics suggest about robustness of this momentum strategy.
"""


def generate_report(prompt: str) -> str:

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found.\n"
            "Set it with: export ANTHROPIC_API_KEY='...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


def validate_output(text: str, comparisons: dict) -> bool:
    """
    Simple sanity check: confirm the LLM mentioned key figures.
    Returns True if valid, False if suspicious.
    """
    checks = [
        str(comparisons["strategy_ann_return"]),
        str(comparisons["benchmark_ann_return"]),
        str(comparisons["strategy_sharpe"]),
    ]
    return all(c in text for c in checks)


def save_report(text: str, output_path: str = "backtest_report.txt"):
    """Save the narrative to a text file."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MOMENTUM STRATEGY — PERFORMANCE NARRATIVE\n")
        f.write(f"Generated from: {BACKTEST_FILE}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":

    print("Calling Claude API...")
    narrative = generate_report(prompt)

    if validate_output(narrative, comparisons):
        print("Validation passed — key figures found in output.\n")
    else:
        print("WARNING: Some expected numbers not found in output.")
        print("Review the narrative carefully before using it.\n")

    print("-" * 60)
    print(narrative)
    print("-" * 60)

    save_report(narrative, RESULTS_DIR)