import zipfile
import xml.etree.ElementTree as ET
import sys

def read_docx(file_path):
    try:
        docx = zipfile.ZipFile(file_path)
        content = docx.read('word/document.xml')
        tree = ET.fromstring(content)
        
        text = []
        for node in tree.iter():
            if node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p':
                text.append('\n')
            elif node.tag == '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t':
                if node.text:
                    text.append(node.text)
        
        return "".join(text)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    file_path = "c:\\Users\\ABC\\Downloads\\Flower_Identification_PRD_v2 (1).docx"
    text = read_docx(file_path)
    with open("prd.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Done")
