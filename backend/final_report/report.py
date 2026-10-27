import io
from html import escape
import pandas as pd


def pdf_bytes(title: str, sections: list) -> bytes:
    """Generate PDF bytes for report. Falls back gracefully if reportlab is not installed."""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        s = getSampleStyleSheet()
        story = [Paragraph(escape(title), s['Title']), Spacer(1, 12)]
        for h, b in sections:
            story += [
                Paragraph(escape(str(h)), s['Heading2']),
                Paragraph(escape(str(b)).replace('\n', '<br/>'), s['BodyText']),
                Spacer(1, 8)
            ]
        doc.build(story)
        return buf.getvalue()
    except Exception:
        # Robust fallback: return structured text/html as document bytes
        text_rep = f"=== {title} ===\n\n"
        for h, b in sections:
            text_rep += f"## {h} ##\n{b}\n\n"
        return text_rep.encode('utf-8')


def html_bytes(title: str, sections: list) -> bytes:
    """Generate HTML report bytes."""
    html_str = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{escape(title)}</title>" \
               f"<style>body{{font-family:Arial,sans-serif;margin:40px;line-height:1.6;background:#f8fafc;color:#1e293b}}" \
               f"h1{{color:#0f172a;border-bottom:2px solid #0284c7;padding-bottom:10px}}" \
               f"h2{{color:#0369a1;margin-top:24px}}pre{{background:#f1f5f9;padding:12px;border-radius:6px;overflow-x:auto}}" \
               f"</style></head><body><h1>{escape(title)}</h1>"
    for h, b in sections:
        html_str += f"<h2>{escape(str(h))}</h2><pre>{escape(str(b))}</pre>"
    html_str += "</body></html>"
    return html_str.encode('utf-8')


def excel_bytes(sheets: dict) -> bytes:
    """Generate Excel bytes across multiple sheets. Falls back to CSV if openpyxl is not available."""
    b = io.BytesIO()
    try:
        with pd.ExcelWriter(b, engine='openpyxl') as w:
            for n, d in sheets.items():
                if isinstance(d, pd.DataFrame):
                    d.to_excel(w, sheet_name=str(n)[:31], index=False)
        return b.getvalue()
    except Exception:
        # Fallback to CSV for first sheet
        first_key = list(sheets.keys())[0] if sheets else 'Data'
        first_df = sheets[first_key] if isinstance(sheets.get(first_key), pd.DataFrame) else pd.DataFrame()
        return first_df.to_csv(index=False).encode('utf-8')
