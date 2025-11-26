import pypdf

pdf_path = "circuit-sparsity-paper.pdf"
output_path = "circuit-sparsity-paper.txt"

with open(pdf_path, 'rb') as file:
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
with open(output_path, 'w', encoding='utf-8') as output:
    output.write(text)

print(f"PDF内容已提取到 {output_path}")
print(f"总页数: {len(reader.pages)}")
