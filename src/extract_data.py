import xml.etree.ElementTree as ET

def extract_relevant_info(file_path):
    """
    Recursively extracts the text content from an XML file, removing tags.
    Returns a plain string of meaningful text.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # Decode
        try:
            root = ET.fromstring(content.decode("utf-8"))
        except UnicodeDecodeError:
            root = ET.fromstring(content.decode("utf-16"))

        text_list = []

        def traverse(node):
            # Capture node text if present
            if node.text and node.text.strip():
                text_list.append(node.text.strip())
            # Recurse
            for child in node:
                traverse(child)
            # Possibly capture node.tail if needed

        traverse(root)

        return "\n".join(text_list)

    except Exception as e:
        raise RuntimeError(f"Failed to process file '{file_path}': {str(e)}")
