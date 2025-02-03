import xml.etree.ElementTree as ET

def extract_relevant_info(file_path):
    """
    Recursively extract text-based content from an XML file.
    Returns a string of meaningful text.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # Attempt UTF-8, fallback to UTF-16
        try:
            root = ET.fromstring(content.decode("utf-8"))
        except UnicodeDecodeError:
            root = ET.fromstring(content.decode("utf-16"))

        text_chunks = []

        def traverse(node):
            if node.text and node.text.strip():
                text_chunks.append(node.text.strip())
            for child in node:
                traverse(child)

        traverse(root)
        return "\n".join(text_chunks)

    except Exception as e:
        raise RuntimeError(f"Failed to process file '{file_path}': {str(e)}")
