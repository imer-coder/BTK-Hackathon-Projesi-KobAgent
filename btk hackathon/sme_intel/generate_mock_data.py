"""
generate_mock_data.py — One-time script to produce the sample Excel file.

Run from the project root:
    python generate_mock_data.py

Output: data/raw/sales_data.xlsx
"""

import pandas as pd
from pathlib import Path

MOCK_ROWS = [
    {"Date": "2024-01-05", "Customer Name": "Anadolu Gıda", "Category": "Tahıl Ürünleri",    "Quantity": 150, "Unit Sales Price": 12.50, "Unit Cost": 8.00,  "Payment Term": "Net 30"},
    {"Date": "2024-01-12", "Customer Name": "Karadeniz Market", "Category": "Süt Ürünleri",  "Quantity": 80,  "Unit Sales Price": 18.00, "Unit Cost": 11.50, "Payment Term": "Immediate"},
    {"Date": "2024-01-20", "Customer Name": "Anadolu Gıda", "Category": "İçecek",            "Quantity": 200, "Unit Sales Price": 7.00,  "Unit Cost": 4.20,  "Payment Term": "Net 30"},
    {"Date": "2024-02-03", "Customer Name": "Ege Distribütör", "Category": "Tahıl Ürünleri", "Quantity": 300, "Unit Sales Price": 11.00, "Unit Cost": 7.50,  "Payment Term": "Net 60"},
    {"Date": "2024-02-14", "Customer Name": "Karadeniz Market", "Category": "Atıştırmalık",  "Quantity": 50,  "Unit Sales Price": 25.00, "Unit Cost": 16.00, "Payment Term": "Immediate"},
    {"Date": "2024-02-28", "Customer Name": "Doğu Ticaret", "Category": "İçecek",            "Quantity": 120, "Unit Sales Price": 9.50,  "Unit Cost": 5.80,  "Payment Term": "Net 30"},
    {"Date": "2024-03-10", "Customer Name": "Ege Distribütör", "Category": "Süt Ürünleri",   "Quantity": 90,  "Unit Sales Price": 20.00, "Unit Cost": 13.00, "Payment Term": "Net 60"},
    {"Date": "2024-03-22", "Customer Name": "Doğu Ticaret", "Category": "Tahıl Ürünleri",    "Quantity": 400, "Unit Sales Price": 10.50, "Unit Cost": 6.90,  "Payment Term": "Net 30"},
    {"Date": "2024-04-01", "Customer Name": "Marmara Toptancı", "Category": "Atıştırmalık",  "Quantity": 70,  "Unit Sales Price": 22.00, "Unit Cost": 14.50, "Payment Term": "Net 45"},
    {"Date": "2024-04-15", "Customer Name": "Marmara Toptancı", "Category": "İçecek",        "Quantity": 180, "Unit Sales Price": 8.00,  "Unit Cost": 4.80,  "Payment Term": "Net 45"},
]

def main() -> None:
    out_path = Path("data/raw/sales_data.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(MOCK_ROWS)
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"[OK] Mock data written to '{out_path}' ({len(df)} rows).")

if __name__ == "__main__":
    main()
