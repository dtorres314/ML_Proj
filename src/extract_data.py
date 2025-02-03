import xml.etree.ElementTree as ET


def extract_relevant_info(file_path):
    """
    Extract relevant information from an XML file.

    Args:
        file_path (str): Path to the XML file.

    Returns:
        dict: A dictionary containing extracted information such as ProblemID, Title, and Statement.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # Parse the XML file
        try:
            root = ET.fromstring(content.decode("utf-8"))
        except UnicodeDecodeError:
            root = ET.fromstring(content.decode("utf-16"))

        # Extract relevant fields
        problem_id = root.findtext("ProblemID", default="").strip()
        title = root.findtext("Title", default="").strip()
        statement = root.findtext("Statement", default="").strip()

        return {
            "ProblemID": problem_id,
            "Title": title,
            "Statement": statement,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to process file {file_path}: {str(e)}")
