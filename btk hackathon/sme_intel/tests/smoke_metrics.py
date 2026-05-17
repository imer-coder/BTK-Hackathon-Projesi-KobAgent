import os
os.environ['OPENAI_API_KEY'] = 'sk-test-dummy'
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-test-dummy'

from src.security import DataMasker
from src.data_loader import load_and_anonymise
from src.agents import _compute_customer_metrics, _metrics_to_prompt_str

masker = DataMasker()
df = load_and_anonymise('data/raw/sales_data.xlsx', masker)

metrics = _compute_customer_metrics(df)
print('=== Pre-computed Metrics (what the LLM receives) ===')
print(_metrics_to_prompt_str(metrics))

print()
print('=== Raw metric values ===')
for cid, m in metrics.items():
    margin = m["avg_margin_ratio"]
    velocity = m["order_velocity_change_pct"]
    print(f'{cid}: margin={margin:.1%}, velocity={velocity:+.1f}%')
