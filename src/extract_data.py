import xml.etree.ElementTree as ET

def extract_relevant_info(file_path):
    """
    Recursively extracts text from an XML file, ignoring tags.
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Attempt UTF-8 decode, fallback to UTF-16
        try:
            root = ET.fromstring(raw_data.decode("utf-8"))
        except UnicodeDecodeError:
            root = ET.fromstring(raw_data.decode("utf-16"))

        text_parts = []

        def traverse(node):
            if node.text and node.text.strip():
                text_parts.append(node.text.strip())
            for child in node:
                traverse(child)

        traverse(root)
        return "\n".join(text_parts)

    except Exception as e:
        raise RuntimeError(f"Failed to process '{file_path}': {str(e)}")
