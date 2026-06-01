from pypdf import PdfReader

pdf_path = r"data/pdfs/sample.pdf"

reader = PdfReader(pdf_path)

print("Number of Pages:", len(reader.pages))

text = ""

for page in reader.pages:
    text += page.extract_text()

print("\nExtracted Text:\n")
print(text[:1000])