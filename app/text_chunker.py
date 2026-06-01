from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = r"data/pdfs/sample.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("Number of Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("-" * 30)
    print(chunk)