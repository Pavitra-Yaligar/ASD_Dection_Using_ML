from fpdf import FPDF
from datetime import datetime
import os

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Autism Detection Report', border=False, ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'C')

def generate_pdf_report(user_name, behavior_result, behavior_confidence,
                        image_result, image_confidence, combined_result=None, output_path='static/reports'):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = f"{user_name.replace(' ', '_')}_autism_report.pdf"
    filepath = os.path.join(output_path, filename)

    pdf = PDFReport()
    pdf.add_page()

    # Body content
    pdf.set_font("Arial", '', 12)

    pdf.cell(0, 10, f"Name/ID: {user_name}", ln=True)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, "Behavioral Model Result:", ln=True)
    pdf.cell(0, 10, f"Prediction: {behavior_result}", ln=True)
    pdf.cell(0, 10, f"Confidence: {behavior_confidence}%", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, "Image Model Result:", ln=True)
    pdf.cell(0, 10, f"Prediction: {image_result}", ln=True)
    pdf.cell(0, 10, f"Confidence: {image_confidence}%", ln=True)
    pdf.ln(5)

    if combined_result:
        pdf.set_text_color(0, 102, 204)
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(0, 10, f"Combined Result: {combined_result}", ln=True)
        pdf.set_text_color(0, 0, 0)

    pdf.output(filepath)
    return filepath
