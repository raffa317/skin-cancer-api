from xhtml2pdf import pisa
import os

def convert_html_to_pdf(source_html_path, output_pdf_path):
    # 1. Read HTML content
    print(f"Reading HTML from: {source_html_path}")
    with open(source_html_path, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

    # 2. Fix Image Paths for xhtml2pdf
    # xhtml2pdf doesn't like file:/// C:/ syntax sometimes. It prefers local paths.
    # We will replace 'file:///C:/' with 'C:/' just to be safe, or ensure it's handled.
    # Actually, let's just try providing the file path directly.
    # But wait, the HTML might contain 'file:///C:/Users/...'
    # Let's clean it up to be safe local paths 'C:/Users/...'
    # Remove style block entirely or simplify it
    # Pandoc generates some CSS that xhtml2pdf doesn't like. 
    # We'll just remove the <style>...</style> block or specific problematic lines.
    # Simple approach: remove the entire <style> block if it exists, or just catch the error.
    # Better: just strip the <style> tag content.
    import re
    html_content = re.sub(r'<style>.*?</style>', '', html_content, flags=re.DOTALL)
    
    html_content = html_content.replace("file:///C:/", "C:/")
    html_content = html_content.replace("file:///c:/", "c:/") 
    
    # 3. Create PDF
    print(f"Writing PDF to: {output_pdf_path}")
    with open(output_pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(
            html_content,                # the HTML to convert
            dest=pdf_file                # file handle to recieve result
        )

    # 4. Check for errors
    if pisa_status.err:
        print(f"Error: Failed to generate PDF: {pisa_status.err}")
        return False
    else:
        print("Success: PDF generated!")
        return True

if __name__ == "__main__":
    # Define paths
    source_html = r"C:\Users\s0u1r\.gemini\antigravity\brain\8939432f-00f7-4080-9418-f47f034be708\AiDerm_Research_Paper.html"
    output_pdf = r"C:\Users\s0u1r\Downloads\AiDerm_Research_Paper.pdf"
    
    if not os.path.exists(source_html):
        print(f"Error: Source HTML not found at {source_html}")
    else:
        convert_html_to_pdf(source_html, output_pdf)
